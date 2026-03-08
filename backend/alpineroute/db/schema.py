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

-- traces alpine indexees depuis index.json
CREATE TABLE IF NOT EXISTS alpine_routes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gpx_path TEXT NOT NULL UNIQUE,
    massif TEXT,
    summit TEXT,
    voie TEXT,
    grade TEXT,
    grade_ord INTEGER,
    distance_m REAL,
    dplus_m REAL,
    start_lat REAL, start_lon REAL,
    end_lat REAL, end_lon REAL,
    geojson TEXT,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- segments terrain (Phase 7)
CREATE TABLE IF NOT EXISTS terrain_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gpx_path TEXT NOT NULL UNIQUE,
    start_name TEXT,
    end_name TEXT,
    segment_type TEXT,
    trail_cost REAL,
    distance_m REAL,
    dplus_m REAL,
    start_lat REAL,
    start_lon REAL,
    end_lat REAL,
    end_lon REAL,
    geojson TEXT,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_routes_created ON routes(created_at);
CREATE INDEX IF NOT EXISTS idx_dem_cache_source ON dem_cache(source, tile_name);
CREATE INDEX IF NOT EXISTS idx_user_zones_type ON user_zones(zone_type, active);
CREATE INDEX IF NOT EXISTS idx_alpine_routes_massif ON alpine_routes(massif);
CREATE INDEX IF NOT EXISTS idx_alpine_routes_summit ON alpine_routes(summit);
CREATE INDEX IF NOT EXISTS idx_terrain_segments_type ON terrain_segments(segment_type);
"""


def init_db(db_path=None):
    """Cree la base et les tables si elles n'existent pas."""
    if db_path is None:
        db_path = DB_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)

    # migration Phase 7: colonnes spatiales sur terrain_segments
    for col in ("start_lat", "start_lon", "end_lat", "end_lon"):
        try:
            conn.execute(f"ALTER TABLE terrain_segments ADD COLUMN {col} REAL")
        except Exception:
            pass  # colonne deja presente
    conn.commit()

    conn.close()
    logger.info("db initialisee: %s", db_path)
    return db_path


def get_connection(db_path=None):
    """Retourne une connexion SQLite."""
    if db_path is None:
        db_path = DB_PATH
    return sqlite3.connect(db_path)
