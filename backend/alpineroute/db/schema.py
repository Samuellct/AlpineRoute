# schema SQLite -- source: A09_sqlite_schema.py

import os
import sqlite3
import logging

from alpineroute.config import DB_PATH

logger = logging.getLogger(__name__)


SCHEMA_SQL = """
-- routes calculees
CREATE TABLE IF NOT EXISTS routes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    name TEXT,
    start_lat REAL NOT NULL,
    start_lon REAL NOT NULL,
    end_lat REAL NOT NULL,
    end_lon REAL NOT NULL,
    dem_source TEXT NOT NULL DEFAULT 'lidar_hd',
    dem_resolution REAL NOT NULL DEFAULT 1.0,
    season_month INTEGER NOT NULL DEFAULT 7,
    acclimatized INTEGER NOT NULL DEFAULT 1,
    load_kg REAL DEFAULT 10,
    landcover_source TEXT,
    distance_m REAL,
    dplus_m REAL,
    dminus_m REAL,
    time_tobler_h REAL,
    glacier_pct REAL,
    cost_total REAL,
    computation_time_s REAL,
    geojson TEXT,
    elevation_profile TEXT
);

-- zones utilisateur
CREATE TABLE IF NOT EXISTS user_zones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    name TEXT NOT NULL,
    zone_type TEXT NOT NULL,
    cost_multiplier REAL DEFAULT 100.0,
    geojson TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);

-- cache DEM
CREATE TABLE IF NOT EXISTS dem_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    tile_name TEXT NOT NULL,
    xmin REAL, ymin REAL, xmax REAL, ymax REAL,
    resolution REAL NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER,
    downloaded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(source, tile_name, resolution)
);

-- preferences
CREATE TABLE IF NOT EXISTS preferences (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_routes_created ON routes(created_at);
CREATE INDEX IF NOT EXISTS idx_dem_cache_source ON dem_cache(source, tile_name);
CREATE INDEX IF NOT EXISTS idx_user_zones_type ON user_zones(zone_type, active);
"""


def init_db(db_path=None):
    """Cree la base et les tables si elles n'existent pas."""
    if db_path is None:
        db_path = DB_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.close()
    logger.info("db initialisee: %s", db_path)
    return db_path


def get_connection(db_path=None):
    """Retourne une connexion SQLite."""
    if db_path is None:
        db_path = DB_PATH
    return sqlite3.connect(db_path)
