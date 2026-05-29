"""
Streaming Recommendation Demo — Event Simulator
================================================
Simulates realistic user browsing behavior and continuously inserts
events into ML_DEMOS.ONLINE_W_STREAMING.RAW_EVENTS.

Behavior:
  - Seeds USERS and ITEMS tables on first run if empty.
  - Maintains a pool of concurrent active sessions (default 15).
  - Each session follows a realistic flow:
      land → view sequence → maybe add-to-cart → maybe purchase → end
  - Dwell time drawn from a log-normal distribution (median ~15s).
  - Item bursts: every 60–90s, 1–2 items trend and get 5–10x more traffic.
  - Inserts in micro-batches every ~1.5s to simulate a live stream.

Usage:
  pip install snowflake-connector-python faker
  python simulate_events.py

Connection: reads from env vars:
  SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD (or SNOWFLAKE_PRIVATE_KEY_PATH)
  Optional: SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE
  Or edit the CONNECTION dict below directly.
"""

import os
import uuid
import time
import random
import json
import math
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

import snowflake.connector

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONNECTION = {
    "account":   os.getenv("SNOWFLAKE_ACCOUNT", ""),
    "user":      os.getenv("SNOWFLAKE_USER", ""),
    "password":  os.getenv("SNOWFLAKE_PASSWORD", ""),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    "role":      os.getenv("SNOWFLAKE_ROLE", ""),
    "database":  "ML_DEMOS",
    "schema":    "ONLINE_W_STREAMING",
}

BATCH_INTERVAL_SEC    = 1.5    # seconds between inserts
BATCH_SIZE            = 12     # events per insert batch
ACTIVE_SESSION_TARGET = 15     # concurrent sessions to maintain
BURST_INTERVAL_MIN    = 60     # seconds between trending item bursts
BURST_INTERVAL_MAX    = 90
BURST_DURATION_SEC    = 120    # how long an item stays trending
BURST_MULTIPLIER      = 7      # how much more likely trending items are

# ---------------------------------------------------------------------------
# Synthetic catalog data
# ---------------------------------------------------------------------------

CATEGORIES = {
    "Running":        ["Road Shoes", "Trail Shoes", "Racing Flats", "Socks", "Shorts", "GPS Watches"],
    "Cycling":        ["Road Bikes", "Mountain Bikes", "Helmets", "Jerseys", "Pedals", "Lights"],
    "Yoga":           ["Mats", "Blocks", "Straps", "Leggings", "Sports Bras", "Towels"],
    "Camping":        ["Tents", "Sleeping Bags", "Headlamps", "Stoves", "Water Filters", "Packs"],
    "Swimming":       ["Goggles", "Swimsuits", "Caps", "Fins", "Pull Buoys", "Kickboards"],
    "Basketball":     ["Shoes", "Balls", "Jerseys", "Shorts", "Knee Pads", "Bags"],
    "Nutrition":      ["Protein Powder", "Energy Gels", "Electrolytes", "Bars", "Pre-Workout"],
    "Recovery":       ["Foam Rollers", "Massage Guns", "Ice Baths", "Compression Sleeves", "Bands"],
}

BRANDS = [
    "Apex", "Strider", "PeakForm", "IronCore", "SwiftGear",
    "Elevate", "ProPulse", "NovaSport", "ZenAthletics", "FieldEdge",
]

COUNTRIES = ["US", "US", "US", "US", "CA", "UK", "AU", "DE", "FR", "JP"]
SEGMENTS  = ["new", "new", "returning", "returning", "returning", "vip"]
DEVICES   = ["mobile", "mobile", "desktop", "desktop", "tablet"]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]

NUM_USERS = 200
NUM_ITEMS = 500

# Event type transition weights within a session:
#   state -> {next_event_type: weight}
# 'view' is the dominant event; purchases are rare.
SESSION_TRANSITIONS = {
    "start":        {"view": 0.70, "click": 0.20, "search": 0.10},
    "view":         {"view": 0.55, "click": 0.20, "add_to_cart": 0.12, "search": 0.08, "end": 0.05},
    "click":        {"view": 0.60, "add_to_cart": 0.20, "search": 0.10, "end": 0.10},
    "search":       {"view": 0.65, "click": 0.25, "end": 0.10},
    "add_to_cart":  {"view": 0.40, "purchase": 0.30, "add_to_cart": 0.15, "end": 0.15},
    "purchase":     {"view": 0.30, "end": 0.70},
}

SEARCH_QUERIES = [
    "running shoes", "trail shoes", "road bike", "yoga mat", "protein powder",
    "basketball shoes", "camping tent", "swim goggles", "foam roller", "energy gels",
    "lightweight backpack", "compression socks", "massage gun", "sleeping bag",
    "cycling helmet", "workout shorts", "sports bra",
]

logging.basicConfig(
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def _rand_dwell() -> int:
    """Log-normal dwell time: median ~15s, mean ~35s, long tail up to ~300s."""
    return max(2, min(300, int(math.exp(random.gauss(2.7, 0.9)))))


def _weighted_choice(options: dict) -> str:
    keys   = list(options.keys())
    weights = list(options.values())
    return random.choices(keys, weights=weights, k=1)[0]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Catalog generation
# ---------------------------------------------------------------------------

def generate_users() -> list[dict]:
    users = []
    base_ts = datetime(2020, 1, 1)
    for i in range(NUM_USERS):
        days_ago = random.randint(0, 1825)
        signup = base_ts.replace(
            month=random.randint(1, 12),
            day=random.randint(1, 28),
            year=random.randint(2020, 2025),
        )
        users.append({
            "user_id":      f"U{i+1:05d}",
            "signup_ts":    signup.isoformat(sep=" "),
            "country":      random.choice(COUNTRIES),
            "user_segment": random.choice(SEGMENTS),
            "device_type":  random.choice(DEVICES),
            "age_group":    random.choice(AGE_GROUPS),
        })
    return users


def generate_items() -> list[dict]:
    items = []
    idx = 1
    per_category = NUM_ITEMS // len(CATEGORIES)
    for category, subcats in CATEGORIES.items():
        for _ in range(per_category):
            subcat = random.choice(subcats)
            # Price ranges vary by category
            base_price = {"Cycling": 300, "Camping": 80, "Running": 120}.get(category, 40)
            price = round(base_price * random.uniform(0.5, 4.0), 2)
            items.append({
                "item_id":      f"I{idx:05d}",
                "category":     category,
                "subcategory":  subcat,
                "brand":        random.choice(BRANDS),
                "price":        price,
                "avg_rating":   round(random.uniform(3.2, 5.0), 1),
                "is_available": random.random() > 0.05,  # 95% in stock
                "created_at":   "2024-01-01 00:00:00",
            })
            idx += 1
    return items


# ---------------------------------------------------------------------------
# Session model
# ---------------------------------------------------------------------------

@dataclass
class Session:
    session_id:    str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id:       str = ""
    state:         str = "start"
    last_item_id:  Optional[str] = None
    events_count:  int = 0
    max_events:    int = field(default_factory=lambda: random.randint(3, 25))
    category_bias: Optional[str] = None  # once a user enters a category, they tend to stay

    def is_done(self) -> bool:
        return self.state == "end" or self.events_count >= self.max_events


# ---------------------------------------------------------------------------
# Burst / trending logic
# ---------------------------------------------------------------------------

class TrendingManager:
    def __init__(self, all_item_ids: list[str]):
        self.all_items   = all_item_ids
        self.trending    : dict[str, float] = {}  # item_id -> expiry timestamp
        self.next_burst  = time.time() + random.uniform(BURST_INTERVAL_MIN, BURST_INTERVAL_MAX)

    def tick(self):
        now = time.time()
        # Expire old trending items
        self.trending = {k: v for k, v in self.trending.items() if v > now}
        # Maybe start a new burst
        if now >= self.next_burst:
            new_items = random.sample(self.all_items, k=random.randint(1, 2))
            expiry = now + BURST_DURATION_SEC
            for item in new_items:
                self.trending[item] = expiry
            log.info(f"  ** TRENDING: {new_items} (for {BURST_DURATION_SEC}s)")
            self.next_burst = now + random.uniform(BURST_INTERVAL_MIN, BURST_INTERVAL_MAX)

    def pick_item(self, category_filter: Optional[str], all_items_by_cat: dict) -> str:
        if category_filter and random.random() < 0.75:
            pool = all_items_by_cat.get(category_filter, self.all_items)
        else:
            pool = self.all_items

        if self.trending and random.random() < 0.3:
            # 30% chance to pick a trending item (regardless of category)
            candidates = list(self.trending.keys())
            return random.choice(candidates)

        # Weighted draw: trending items get BURST_MULTIPLIER weight, others get 1
        weights = [BURST_MULTIPLIER if i in self.trending else 1 for i in pool]
        return random.choices(pool, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Event generation
# ---------------------------------------------------------------------------

def generate_event(
    session: Session,
    users: list[dict],
    trending: TrendingManager,
    items_by_cat: dict[str, list[str]],
    all_items: list[str],
) -> Optional[dict]:
    """Advance session state and produce one event dict, or None if session ended."""
    if session.is_done():
        return None

    user = next(u for u in users if u["user_id"] == session.user_id)
    transitions = SESSION_TRANSITIONS.get(session.state, {"end": 1.0})
    next_state = _weighted_choice(transitions)
    session.state = next_state
    session.events_count += 1

    if next_state == "end":
        return None

    now = _now_utc()
    event = {
        "event_id":          str(uuid.uuid4()),
        "session_id":        session.session_id,
        "user_id":           session.user_id,
        "item_id":           None,
        "event_type":        next_state,
        "event_ts":          now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "dwell_time_sec":    None,
        "search_query":      None,
        "referrer_item_id":  None,
        "properties":        json.dumps({
            "device":   user["device_type"],
            "country":  user["country"],
        }),
    }

    if next_state == "search":
        event["search_query"] = random.choice(SEARCH_QUERIES)
        # Bias future items toward a matching category
        query = event["search_query"].lower()
        for cat in items_by_cat:
            if cat.lower() in query or any(w in query for w in cat.lower().split()):
                session.category_bias = cat
                break

    elif next_state in ("view", "click", "add_to_cart", "purchase"):
        item_id = trending.pick_item(session.category_bias, items_by_cat)
        event["item_id"] = item_id
        session.last_item_id = item_id

        # Infer category from item and set bias
        for cat, cat_items in items_by_cat.items():
            if item_id in cat_items:
                session.category_bias = cat
                break

        if next_state == "view":
            event["dwell_time_sec"] = _rand_dwell()
        if next_state == "click" and session.last_item_id:
            event["referrer_item_id"] = session.last_item_id

    return event


# ---------------------------------------------------------------------------
# Snowflake I/O
# ---------------------------------------------------------------------------

INSERT_EVENTS_SQL = """
INSERT INTO RAW_EVENTS (
    event_id, session_id, user_id, item_id, event_type, event_ts,
    dwell_time_sec, search_query, referrer_item_id, properties
) VALUES (
    %(event_id)s, %(session_id)s, %(user_id)s, %(item_id)s, %(event_type)s,
    %(event_ts)s::TIMESTAMP_NTZ, %(dwell_time_sec)s, %(search_query)s,
    %(referrer_item_id)s, PARSE_JSON(%(properties)s)
)
"""

INSERT_USERS_SQL = """
INSERT INTO USERS (user_id, signup_ts, country, user_segment, device_type, age_group)
VALUES (%(user_id)s, %(signup_ts)s::TIMESTAMP_NTZ, %(country)s,
        %(user_segment)s, %(device_type)s, %(age_group)s)
"""

INSERT_ITEMS_SQL = """
INSERT INTO ITEMS (item_id, category, subcategory, brand, price,
                   avg_rating, is_available, created_at)
VALUES (%(item_id)s, %(category)s, %(subcategory)s, %(brand)s, %(price)s,
        %(avg_rating)s, %(is_available)s, %(created_at)s::TIMESTAMP_NTZ)
"""


def seed_catalog(cur) -> tuple[list[dict], list[dict]]:
    """Insert users and items if not already seeded."""
    cur.execute("SELECT COUNT(*) FROM USERS")
    user_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM ITEMS")
    item_count = cur.fetchone()[0]

    users = generate_users()
    items = generate_items()

    if user_count == 0:
        log.info(f"Seeding {len(users)} users...")
        cur.executemany(INSERT_USERS_SQL, users)
        log.info("  Users seeded.")
    else:
        log.info(f"Users already seeded ({user_count} rows). Loading from DB...")
        cur.execute("SELECT user_id, device_type, country FROM USERS")
        rows = cur.fetchall()
        users = [{"user_id": r[0], "device_type": r[1], "country": r[2]} for r in rows]

    if item_count == 0:
        log.info(f"Seeding {len(items)} items...")
        cur.executemany(INSERT_ITEMS_SQL, items)
        log.info("  Items seeded.")
    else:
        log.info(f"Items already seeded ({item_count} rows). Loading from DB...")
        cur.execute("SELECT item_id, category FROM ITEMS")
        rows = cur.fetchall()
        items = [{"item_id": r[0], "category": r[1]} for r in rows]

    return users, items


def insert_events(cur, events: list[dict]):
    cur.executemany(INSERT_EVENTS_SQL, events)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    log.info("Connecting to Snowflake...")
    conn = snowflake.connector.connect(**{k: v for k, v in CONNECTION.items() if v})
    cur  = conn.cursor()

    log.info("Seeding catalog...")
    users, items = seed_catalog(cur)

    # Build lookup structures
    user_ids = [u["user_id"] for u in users]

    items_by_cat: dict[str, list[str]] = {}
    for item in items:
        cat = item.get("category", "Unknown")
        items_by_cat.setdefault(cat, []).append(item["item_id"])
    all_items = [item["item_id"] for item in items]

    trending = TrendingManager(all_items)
    sessions: list[Session] = []

    total_inserted = 0
    run_start      = time.time()
    loop_count     = 0

    log.info(f"Starting event stream. Target: {ACTIVE_SESSION_TARGET} concurrent sessions.")
    log.info("Press Ctrl+C to stop.\n")

    try:
        while True:
            loop_count += 1
            trending.tick()

            # Top up sessions
            while len(sessions) < ACTIVE_SESSION_TARGET:
                s = Session(user_id=random.choice(user_ids))
                sessions.append(s)

            # Generate one event per active session
            batch = []
            for session in sessions:
                event = generate_event(session, users, trending, items_by_cat, all_items)
                if event:
                    batch.append(event)

            # Remove finished sessions
            sessions = [s for s in sessions if not s.is_done()]

            # Insert batch
            if batch:
                insert_events(cur, batch)
                total_inserted += len(batch)

            # Stats every 10 loops
            if loop_count % 10 == 0:
                elapsed   = time.time() - run_start
                rate      = total_inserted / elapsed if elapsed > 0 else 0
                trending_items = list(trending.trending.keys())
                log.info(
                    f"Events: {total_inserted:,}  |  "
                    f"Rate: {rate:.1f}/s  |  "
                    f"Active sessions: {len(sessions)}  |  "
                    f"Trending: {trending_items or 'none'}"
                )

            time.sleep(BATCH_INTERVAL_SEC)

    except KeyboardInterrupt:
        elapsed = time.time() - run_start
        log.info(f"\nStopped. Inserted {total_inserted:,} events in {elapsed:.0f}s "
                 f"({total_inserted/elapsed:.1f} events/sec)")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
