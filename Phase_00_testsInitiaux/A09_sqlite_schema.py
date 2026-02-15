# A09 - schema SQLite pour la V1

import sqlite3
import json
import time


SCHEMA_SQL = """
-- routes calculees
CREATE TABLE IF NOT EXISTS routes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    name TEXT,

    -- coords depart/arrivee (WGS84)
    start_lat REAL NOT NULL,
    start_lon REAL NOT NULL,
    end_lat REAL NOT NULL,
    end_lon REAL NOT NULL,

    -- parametres du calcul
    dem_source TEXT NOT NULL DEFAULT 'lidar_hd',  -- lidar_hd | copernicus_glo30
    dem_resolution REAL NOT NULL DEFAULT 1.0,
    season_month INTEGER NOT NULL DEFAULT 7,
    acclimatized INTEGER NOT NULL DEFAULT 1,
    load_kg REAL DEFAULT 10,
    landcover_source TEXT,  -- worldcover_2021 | null

    -- resultats
    distance_m REAL,
    dplus_m REAL,
    dminus_m REAL,
    time_tobler_h REAL,
    glacier_pct REAL,
    cost_total REAL,
    computation_time_s REAL,

    -- geometrie stockee en GeoJSON text (pas de spatialite)
    geojson TEXT,
    -- profil altimetrique en JSON : [[dist_m, alt_m], ...]
    elevation_profile TEXT
);

-- zones utilisateur (crevasses manuelles, zones interdites, etc)
CREATE TABLE IF NOT EXISTS user_zones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    name TEXT NOT NULL,
    zone_type TEXT NOT NULL,  -- crevasse | forbidden | waypoint | ...
    -- facteur de cout multiplicatif (ex: 100 pour zone interdite)
    cost_multiplier REAL DEFAULT 100.0,
    -- geometrie en GeoJSON (polygon, point, linestring)
    geojson TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);

-- cache DEM (quelles tuiles sont deja telecharges)
CREATE TABLE IF NOT EXISTS dem_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,       -- lidar_hd | copernicus_glo30
    tile_name TEXT NOT NULL,    -- nom de la dalle ou tuile
    -- bbox L93 de la tuile
    xmin REAL, ymin REAL, xmax REAL, ymax REAL,
    resolution REAL NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER,
    downloaded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(source, tile_name, resolution)
);

-- preferences utilisateur
CREATE TABLE IF NOT EXISTS preferences (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- index pour les recherches frequentes
CREATE INDEX IF NOT EXISTS idx_routes_created ON routes(created_at);
CREATE INDEX IF NOT EXISTS idx_routes_bbox ON routes(start_lat, start_lon);
CREATE INDEX IF NOT EXISTS idx_dem_cache_source ON dem_cache(source, tile_name);
CREATE INDEX IF NOT EXISTS idx_user_zones_type ON user_zones(zone_type, active);
"""


def test_schema():
    """Cree le schema en memoire et teste des insertions."""
    print("--- Creation du schema ---")
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL)
    print("  [OK] schema cree sans erreur")

    # --- test insertion route ---
    print("\n--- Test insertion route ---")
    route_geojson = json.dumps({
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[6.8965, 45.8778, 3842], [6.9297, 45.8845, 2516]],
        },
    })

    conn.execute("""
        INSERT INTO routes (name, start_lat, start_lon, end_lat, end_lon,
                           dem_source, dem_resolution, season_month, acclimatized,
                           distance_m, dplus_m, dminus_m, time_tobler_h,
                           glacier_pct, cost_total, computation_time_s,
                           geojson, landcover_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "Requin -> Aiguille du Midi",
        45.8845, 6.9297,  # depart
        45.8778, 6.8965,  # arrivee
        "lidar_hd", 1.0, 7, 1,
        4200, 1126, 98, 3.5,
        45.2, 49017.5, 9.3,
        route_geojson,
        "worldcover_2021",
    ))
    print("  [OK] route inseree")

    # verif
    row = conn.execute("SELECT id, name, distance_m, created_at FROM routes").fetchone()
    print(f"  id={row[0]}, name='{row[1]}', dist={row[2]}m, created={row[3]}")

    # --- test zone utilisateur ---
    print("\n--- Test insertion zone utilisateur ---")
    zone_geojson = json.dumps({
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[6.89, 45.88], [6.90, 45.88], [6.90, 45.89], [6.89, 45.89], [6.89, 45.88]]],
        },
    })

    conn.execute("""
        INSERT INTO user_zones (name, zone_type, cost_multiplier, geojson)
        VALUES (?, ?, ?, ?)
    """, ("Zone crevasses Vallee Blanche", "crevasse", 50.0, zone_geojson))
    print("  [OK] zone inseree")

    # --- test cache DEM ---
    print("\n--- Test insertion cache DEM ---")
    conn.execute("""
        INSERT INTO dem_cache (source, tile_name, xmin, ymin, xmax, ymax,
                              resolution, file_path, file_size_bytes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "lidar_hd", "LHD_FXX_1002_6543_MNT_O_0M50_LAMB93_IGN69",
        1002000, 6542000, 1003000, 6543000,
        0.5, "data/dem/raw_ign/LHD_FXX_1002_6543.tif", 4200000,
    ))
    print("  [OK] cache inseree")

    # test unicite
    try:
        conn.execute("""
            INSERT INTO dem_cache (source, tile_name, xmin, ymin, xmax, ymax,
                                  resolution, file_path, file_size_bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "lidar_hd", "LHD_FXX_1002_6543_MNT_O_0M50_LAMB93_IGN69",
            1002000, 6542000, 1003000, 6543000,
            0.5, "data/dem/raw_ign/LHD_FXX_1002_6543.tif", 4200000,
        ))
        print("  [FAIL] doublon accepte (devrait etre UNIQUE)")
    except sqlite3.IntegrityError:
        print("  [OK] contrainte UNIQUE respectee (doublon rejete)")

    # --- test preferences ---
    print("\n--- Test preferences ---")
    conn.execute("INSERT INTO preferences (key, value) VALUES (?, ?)",
                 ("default_resolution", "1.0"))
    conn.execute("INSERT INTO preferences (key, value) VALUES (?, ?)",
                 ("default_season", "7"))
    conn.execute("INSERT INTO preferences (key, value) VALUES (?, ?)",
                 ("theme", "dark"))

    prefs = conn.execute("SELECT key, value FROM preferences").fetchall()
    for k, v in prefs:
        print(f"  {k} = {v}")

    # --- requetes utiles ---
    print("\n--- Requetes de test ---")

    # derniere route
    r = conn.execute("""
        SELECT name, distance_m, dplus_m, time_tobler_h, dem_source
        FROM routes ORDER BY created_at DESC LIMIT 1
    """).fetchone()
    print(f"  derniere route: {r[0]} ({r[1]}m, D+{r[2]}m, ~{r[3]}h, src={r[4]})")

    # zones actives
    zones = conn.execute("""
        SELECT name, zone_type, cost_multiplier FROM user_zones WHERE active=1
    """).fetchall()
    print(f"  zones actives: {len(zones)}")
    for z in zones:
        print(f"    {z[0]} ({z[1]}, x{z[2]})")

    # cache stats
    cache = conn.execute("""
        SELECT source, COUNT(*), SUM(file_size_bytes)/1024/1024 as mb
        FROM dem_cache GROUP BY source
    """).fetchall()
    for src, n, mb in cache:
        print(f"  cache {src}: {n} tuiles, {mb:.1f} MB")

    conn.close()
    print("\n[OK] Tous les tests passes")


def main():
    print("=" * 60)
    print("A09 - Schema SQLite")
    print("=" * 60)

    test_schema()

    # affiche le SQL pour reference
    print("\n--- Schema SQL ---")
    for line in SCHEMA_SQL.strip().split("\n"):
        print(f"  {line}")

    print("\nDone!")


if __name__ == "__main__":
    main()
