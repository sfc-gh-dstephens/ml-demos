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
