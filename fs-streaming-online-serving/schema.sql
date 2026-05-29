-- ============================================================
-- Streaming Recommendation Demo — Data Model
-- Database: ML_DEMOS  |  Schema: ONLINE_W_STREAMING
-- ============================================================

CREATE DATABASE IF NOT EXISTS ML_DEMOS;
CREATE SCHEMA IF NOT EXISTS ML_DEMOS.ONLINE_W_STREAMING;

USE DATABASE ML_DEMOS;
USE SCHEMA ONLINE_W_STREAMING;

-- ------------------------------------------------------------
-- USER DIMENSION
-- Seeded once. Represents registered users.
-- Slow-changing — batch refresh is fine.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS USERS (
    user_id         VARCHAR(20)     NOT NULL,
    signup_ts       TIMESTAMP_NTZ,
    country         VARCHAR(50),
    user_segment    VARCHAR(20),    -- 'new' | 'returning' | 'vip'
    device_type     VARCHAR(20),    -- 'mobile' | 'desktop' | 'tablet'
    age_group       VARCHAR(20),    -- '18-24' | '25-34' | '35-44' | '45-54' | '55+'
    PRIMARY KEY (user_id)
);

-- ------------------------------------------------------------
-- ITEM / PRODUCT CATALOG
-- Seeded once. Represents the product catalog.
-- Slow-changing — batch refresh is fine.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ITEMS (
    item_id         VARCHAR(20)     NOT NULL,
    category        VARCHAR(50)     NOT NULL,
    subcategory     VARCHAR(50),
    brand           VARCHAR(50),
    price           FLOAT,
    avg_rating      FLOAT,          -- 1.0 – 5.0
    is_available    BOOLEAN         DEFAULT TRUE,
    created_at      TIMESTAMP_NTZ,
    PRIMARY KEY (item_id)
);

-- ------------------------------------------------------------
-- RAW EVENTS — the hot streaming table
-- Every user interaction writes here in near real-time.
-- This is what the Python simulator continuously appends to.
--
-- event_type values:
--   view        user viewed an item detail page
--   click       user clicked an item in a list/search result
--   add_to_cart user added item to cart
--   purchase    user completed a purchase
--   search      user submitted a search query
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS RAW_EVENTS (
    event_id            VARCHAR(36)     NOT NULL,   -- UUID
    session_id          VARCHAR(36)     NOT NULL,
    user_id             VARCHAR(20)     NOT NULL,
    item_id             VARCHAR(20),                -- NULL for search events
    event_type          VARCHAR(20)     NOT NULL,
    event_ts            TIMESTAMP_NTZ   NOT NULL,
    dwell_time_sec      INT,                        -- seconds on item page (view events only)
    search_query        VARCHAR(200),               -- search events only
    referrer_item_id    VARCHAR(20),                -- item that led the user here (click-through)
    properties          VARIANT,                    -- flexible bag: device, page, position, etc.
    PRIMARY KEY (event_id)
);

-- ------------------------------------------------------------
-- CDC STREAM — consumed by the Feature Store stream feature view
-- Captures every new row appended to RAW_EVENTS.
-- The Feature Store's StreamSource will point at this stream.
-- ------------------------------------------------------------
CREATE STREAM IF NOT EXISTS RAW_EVENTS_STREAM
    ON TABLE RAW_EVENTS
    APPEND_ONLY = TRUE
    COMMENT = 'CDC stream for Feature Store streaming feature views';

-- ------------------------------------------------------------
-- Verification queries (run after the simulator has been
-- running for a few minutes)
-- ------------------------------------------------------------

-- Event volume and freshness
-- SELECT COUNT(*) AS total_events, MAX(event_ts) AS latest_event FROM RAW_EVENTS;

-- Event type distribution (expect ~65% view, ~15% click, ~12% add_to_cart, ~8% purchase)
-- SELECT event_type, COUNT(*) AS cnt,
--        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
-- FROM RAW_EVENTS
-- GROUP BY 1 ORDER BY 2 DESC;

-- Top trending items in the last 5 minutes
-- SELECT item_id, COUNT(*) AS recent_views
-- FROM RAW_EVENTS
-- WHERE event_type = 'view'
--   AND event_ts > DATEADD('minute', -5, CURRENT_TIMESTAMP)
-- GROUP BY 1 ORDER BY 2 DESC LIMIT 10;

-- Confirm stream is capturing changes
-- SELECT COUNT(*) AS unread_stream_rows FROM RAW_EVENTS_STREAM;

-- Active sessions right now
-- SELECT COUNT(DISTINCT session_id) AS active_sessions
-- FROM RAW_EVENTS
-- WHERE event_ts > DATEADD('minute', -5, CURRENT_TIMESTAMP);
