"""
Online Inference Script — Next-Item Recommendations
====================================================
Reads features from the online Feature Store (Postgres) and sends them
to the SPCS-hosted model inference API for scoring.

Flow:
  1. Read user behavioral features from USER_EVENTS_STREAM_FV (stream, online)
  2. Read user profile from USER_PROFILE_FV (batch, online)
  3. Read candidate item features from ITEM_PROFILE_FV (batch, online)
  4. Assemble feature matrix (raw columns — pipeline handles encoding)
  5. POST to model inference API
  6. Rank by predicted score, return top-K
  7. Log results to INFERENCE_RESULTS table

Usage:
  python run_inference.py --user U00042 --top-k 5
  python run_inference.py  # random user, default top-5

Env vars:
  INFERENCE_URL  — model service endpoint (SPCS)
  SNOWFLAKE_PAT  — Snowflake personal access token
"""

import os
import argparse
import json
from decimal import Decimal
from datetime import datetime, timezone

from dotenv import load_dotenv
import pandas as pd
import requests
from snowflake.snowpark import Session
from snowflake.ml.feature_store import FeatureStore, CreationMode

load_dotenv()

INFERENCE_URL = os.getenv("INFERENCE_URL", "")
SNOWFLAKE_PAT = os.getenv("SNOWFLAKE_PAT", "")


def get_random_user(session) -> str:
    """Pick a random active user from recent events."""
    row = session.sql("""
        SELECT user_id FROM ML_DEMOS.ONLINE_W_STREAMING.RAW_EVENTS
        WHERE event_ts > DATEADD('hour', -1, CURRENT_TIMESTAMP())
        ORDER BY RANDOM() LIMIT 1
    """).collect()
    if row:
        return row[0][0]
    row = session.sql("SELECT user_id FROM ML_DEMOS.ONLINE_W_STREAMING.USERS ORDER BY RANDOM() LIMIT 1").collect()
    return row[0][0]


def get_candidate_items(session, n=20) -> list[str]:
    """Get a pool of candidate items to rank."""
    rows = session.sql(f"""
        SELECT item_id FROM ML_DEMOS.ONLINE_W_STREAMING.ITEMS
        WHERE is_available = TRUE
        ORDER BY RANDOM() LIMIT {n}
    """).collect()
    return [r[0] for r in rows]


def call_inference_api(scoring_df: pd.DataFrame) -> list[float]:
    """POST features to the model inference API and return scores."""
    # Convert Decimal columns to float for JSON serialization
    for col in scoring_df.columns:
        if scoring_df[col].apply(lambda x: isinstance(x, Decimal)).any():
            scoring_df[col] = scoring_df[col].astype(float)

    # Fill NaN with appropriate defaults
    scoring_df = scoring_df.fillna(0)

    # Reorder columns to match model signature exactly
    model_col_order = [
        "PRICE", "AVG_RATING",
        "VIEW_COUNT_1H", "VIEW_COUNT_24H", "CART_COUNT_1H",
        "PURCHASE_COUNT_24H", "TOTAL_DWELL_SEC_1H", "UNIQUE_ITEMS_VIEWED_1H",
        "COUNTRY", "USER_SEGMENT", "DEVICE_TYPE", "AGE_GROUP",
        "CATEGORY", "SUBCATEGORY", "BRAND",
    ]
    scoring_df = scoring_df[model_col_order]

    # SPCS inference format: each row is [row_index, col1, col2, ...]
    rows = []
    for idx, (_, row) in enumerate(scoring_df.iterrows()):
        rows.append([idx] + row.tolist())
    payload = {"data": rows}

    response = requests.post(
        f"{INFERENCE_URL}/predict",
        headers={
            "Authorization": f'Snowflake Token="{SNOWFLAKE_PAT}"',
            "Content-Type": "application/json",
            "sf-custom-input-columns": ",".join(model_col_order),
        },
        json=payload,
    )
    response.raise_for_status()
    result = response.json()

    # Response format: {"data": [[row_idx, {"output_feature_0": predicted_class}], ...]}
    scores = []
    for row in result["data"]:
        prediction = row[1]  # second element is the prediction dict
        if isinstance(prediction, dict):
            scores.append(float(prediction.get("output_feature_0", prediction.get("predict", 0))))
        else:
            scores.append(float(prediction))
    return scores


def run_inference(user_id: str, top_k: int = 5, n_candidates: int = 20):
    """Run a single inference pass for one user."""
    # Connect
    session = Session.builder.config("connection_name", "demo").create()
    session.use_database("ML_DEMOS")
    session.use_schema("ONLINE_W_STREAMING")
    session.use_warehouse("ML_DEMO_WH")

    fs = FeatureStore(
        session=session, database="ML_DEMOS", name="ONLINE_W_STREAMING",
        default_warehouse="ML_DEMO_WH", creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
    )

    # Pick user if not specified
    if not user_id:
        user_id = get_random_user(session)
    print(f"\nRunning inference for user: {user_id}")

    # --- Read features from the online store ---

    # 1. Stream FV: real-time behavioral aggregations
    # Stream FV has entities [USER, ITEM] — query with user + first candidate item
    # (aggregations are user-level, item key is required but doesn't filter results)
    stream_fv = fs.get_feature_view("USER_EVENTS_STREAM_FV", "V1")
    candidate_items = get_candidate_items(session, n_candidates)
    user_stream_features = fs.read_feature_view(
        stream_fv, keys=[[user_id, candidate_items[0]]], store_type="online"
    )
    print(f"  Stream features: {user_stream_features.columns.tolist()}")

    # 2. Batch FV: user profile
    profile_fv = fs.get_feature_view("USER_PROFILE_FV", "V1")
    user_profile_features = fs.read_feature_view(
        profile_fv, keys=[[user_id]], store_type="online"
    )
    print(f"  Profile features: {user_profile_features.columns.tolist()}")

    # 3. Batch FV: candidate item features
    item_fv = fs.get_feature_view("ITEM_PROFILE_FV", "V1")
    item_features = fs.read_feature_view(
        item_fv, keys=[[item_id] for item_id in candidate_items], store_type="online"
    )
    print(f"  Item features: {item_features.shape[0]} candidates")

    # --- Assemble scoring DataFrame ---

    # Merge user features into a single row
    user_row = pd.concat([
        user_profile_features.drop(columns=["USER_ID"], errors="ignore"),
        user_stream_features.drop(columns=["USER_ID", "ITEM_ID"], errors="ignore"),
    ], axis=1)

    # Cross-join: repeat user row for each candidate item
    scoring_df = item_features.merge(user_row, how="cross")

    print(f"  Scoring matrix: {scoring_df.shape[0]} rows x {scoring_df.shape[1]} cols")

    # --- Call inference API (no encoding needed — pipeline handles it) ---
    # Drop entity keys — the model only sees feature columns from the FVs
    feature_df = scoring_df.drop(columns=["ITEM_ID", "USER_ID"], errors="ignore")

    scores = call_inference_api(feature_df)
    scoring_df["score"] = scores
    ranked = scoring_df.sort_values("score", ascending=False).head(top_k)

    # --- Display results ---
    print(f"\n  Top-{top_k} recommendations for {user_id}:")
    print(f"  {'ITEM_ID':<10} {'CATEGORY':<12} {'PRICE':>8} {'SCORE':>8}")
    print(f"  {'-'*42}")
    for _, row in ranked.iterrows():
        print(f"  {row.get('ITEM_ID', '?'):<10} {str(row.get('CATEGORY', '?')):<12} {row.get('PRICE', 0):>8.2f} {row['score']:>8.3f}")

    return ranked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run online inference for next-item recommendations")
    parser.add_argument("--user", type=str, default=None, help="User ID (e.g. U00042). Random if not specified.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations to return")
    parser.add_argument("--candidates", type=int, default=20, help="Number of candidate items to score")
    args = parser.parse_args()

    run_inference(args.user, args.top_k, args.candidates)
