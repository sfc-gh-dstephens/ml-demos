"""
Online Inference Script — Next-Item Recommendations
====================================================
Reads features from the online Feature Store (Postgres) and scores
candidate items to produce ranked recommendations.

Flow:
  1. Read user behavioral features from USER_EVENTS_STREAM_FV (stream, online)
  2. Read user profile from USER_PROFILE_FV (batch, online)
  3. Read candidate item features from ITEM_PROFILE_FV (batch, online)
  4. Assemble feature matrix, score with the trained model
  5. Return top-K ranked items
  6. Log results to INFERENCE_RESULTS table

Usage:
  python run_inference.py --user U00042 --top-k 5
  python run_inference.py  # random user, default top-5

Requires:
  - model_artifacts.pkl (produced by 02_train_and_deploy.ipynb)
  - SNOWFLAKE_PAT env var for online store access
"""

import os
import sys
import argparse
import pickle
import random
import json
from datetime import datetime, timezone

import pandas as pd
from snowflake.snowpark import Session
from snowflake.ml.feature_store import FeatureStore, CreationMode
from snowflake.ml.registry import Registry


def load_artifacts():
    """Load feature columns and label encoders from training."""
    artifacts_path = os.path.join(os.path.dirname(__file__), "model_artifacts.pkl")
    with open(artifacts_path, "rb") as f:
        return pickle.load(f)


def get_random_user(session) -> str:
    """Pick a random active user from recent events."""
    row = session.sql("""
        SELECT user_id FROM ML_DEMOS.ONLINE_W_STREAMING.RAW_EVENTS
        WHERE event_ts > DATEADD('hour', -1, CURRENT_TIMESTAMP())
        ORDER BY RANDOM() LIMIT 1
    """).collect()
    if row:
        return row[0][0]
    # Fallback to any user
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

    # Load model from registry
    registry = Registry(session=session, database_name="ML_DEMOS", schema_name="ONLINE_W_STREAMING")
    model = registry.get_model("RECOMMENDATION_RANKER").version("V1").load()

    # Load training artifacts
    artifacts = load_artifacts()
    FEATURE_COLUMNS = artifacts["feature_columns"]
    CATEGORICAL_FEATURES = artifacts["categorical_features"]
    NUMERIC_FEATURES = artifacts["numeric_features"]
    encoders = artifacts["encoders"]

    # Pick user if not specified
    if not user_id:
        user_id = get_random_user(session)
    print(f"\nRunning inference for user: {user_id}")

    # --- Read features from the online store ---

    # 1. Stream FV: real-time behavioral aggregations
    stream_fv = fs.get_feature_view("USER_EVENTS_STREAM_FV", "V1")
    user_stream_features = fs.read_feature_view(
        stream_fv, keys=[[user_id]], store_type="online"
    )
    print(f"  Stream features: {user_stream_features.columns.tolist()}")

    # 2. Batch FV: user profile
    profile_fv = fs.get_feature_view("USER_PROFILE_FV", "V1")
    user_profile_features = fs.read_feature_view(
        profile_fv, keys=[[user_id]], store_type="online"
    )
    print(f"  Profile features: {user_profile_features.columns.tolist()}")

    # 3. Batch FV: candidate item features
    candidate_items = get_candidate_items(session, n_candidates)
    item_fv = fs.get_feature_view("ITEM_PROFILE_FV", "V1")
    item_features = fs.read_feature_view(
        item_fv, keys=[[item_id] for item_id in candidate_items], store_type="online"
    )
    print(f"  Item features: {item_features.shape[0]} candidates")

    # --- Assemble scoring DataFrame ---

    # Merge user features into a single row
    user_row = pd.concat([
        user_profile_features.drop(columns=["USER_ID"], errors="ignore"),
        user_stream_features.drop(columns=["USER_ID"], errors="ignore"),
    ], axis=1)

    # Cross-join: repeat user row for each candidate item
    scoring_df = item_features.merge(user_row, how="cross")

    # --- Prepare features (same encoding as training) ---

    for col in CATEGORICAL_FEATURES:
        if col in scoring_df.columns:
            le = encoders[col]
            scoring_df[col] = scoring_df[col].fillna("UNKNOWN").astype(str)
            # Handle unseen labels
            scoring_df[col] = scoring_df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    for col in NUMERIC_FEATURES:
        if col in scoring_df.columns:
            scoring_df[col] = scoring_df[col].fillna(0)

    # Only keep columns the model expects
    X = scoring_df[FEATURE_COLUMNS].fillna(0)

    # --- Score and rank ---
    # Model predicts probability of each relevance class (0-3)
    # Use weighted sum: higher weight for purchase/cart probabilities
    proba = model.predict_proba(X)
    # Score = 0*P(view) + 1*P(click) + 2*P(cart) + 3*P(purchase)
    weights = [0, 1, 2, 3]
    scores = sum(proba[:, i] * w for i, w in enumerate(weights))

    scoring_df["score"] = scores
    ranked = scoring_df.sort_values("score", ascending=False).head(top_k)

    # --- Display results ---
    print(f"\n  Top-{top_k} recommendations for {user_id}:")
    print(f"  {'ITEM_ID':<10} {'CATEGORY':<12} {'PRICE':>8} {'SCORE':>8}")
    print(f"  {'-'*42}")
    for _, row in ranked.iterrows():
        # Decode category back
        cat_idx = int(row.get("CATEGORY", 0))
        cat_name = encoders["CATEGORY"].inverse_transform([cat_idx])[0] if cat_idx >= 0 else "?"
        print(f"  {row.get('ITEM_ID', '?'):<10} {cat_name:<12} {row.get('PRICE', 0):>8.2f} {row['score']:>8.3f}")

    # --- Log results ---
    try:
        session.sql("""
            CREATE TABLE IF NOT EXISTS INFERENCE_RESULTS (
                user_id VARCHAR(20),
                recommended_items VARIANT,
                scores VARIANT,
                inference_ts TIMESTAMP_NTZ
            )
        """).collect()

        result_items = ranked["ITEM_ID"].tolist() if "ITEM_ID" in ranked.columns else []
        result_scores = ranked["score"].tolist()
        now = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")

        session.sql(f"""
            INSERT INTO INFERENCE_RESULTS (user_id, recommended_items, scores, inference_ts)
            SELECT '{user_id}',
                   PARSE_JSON('{json.dumps(result_items)}'),
                   PARSE_JSON('{json.dumps([round(s, 4) for s in result_scores])}'),
                   '{now}'::TIMESTAMP_NTZ
        """).collect()
        print(f"\n  Results logged to INFERENCE_RESULTS table.")
    except Exception as e:
        print(f"\n  Warning: could not log results: {e}")

    session.close()
    return ranked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run online inference for next-item recommendations")
    parser.add_argument("--user", type=str, default=None, help="User ID (e.g. U00042). Random if not specified.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations to return")
    parser.add_argument("--candidates", type=int, default=20, help="Number of candidate items to score")
    args = parser.parse_args()

    run_inference(args.user, args.top_k, args.candidates)
