# chargement index.json, sync SQLite, hot reload

import os
import json
import logging

from alpineroute.config import GPX_DIR, GPX_INDEX_PATH, DB_PATH
from alpineroute.db.schema import get_connection, init_db
from alpineroute.alpine.routes import (
    load_gpx, compute_stats, route_to_geojson, GRADE_ORDINAL,
)

logger = logging.getLogger(__name__)


def load_index(path=None):
    """Lit index.json, retourne la liste des entries valides."""
    if path is None:
        path = GPX_INDEX_PATH
    if not os.path.isfile(path):
        logger.warning("index.json absent: %s", path)
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("erreur lecture index.json: %s", e)
        return []

    if not isinstance(data, list):
        logger.warning("index.json: attendu une liste, got %s", type(data).__name__)
        return []

    valid = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            logger.warning("index.json[%d]: pas un dict, skip", i)
            continue
        if "type" not in entry or "gpx_file" not in entry:
            logger.warning("index.json[%d]: champs type/gpx_file manquants, skip", i)
            continue
        if entry["type"] not in ("route", "segment"):
            logger.warning("index.json[%d]: type=%s invalide, skip", i, entry["type"])
            continue
        valid.append(entry)
    return valid


def sync_to_sqlite(entries, db_path=None):
    """Full replace: DELETE + INSERT pour routes et segments."""
    if db_path is None:
        db_path = DB_PATH

    conn = get_connection(db_path)
    n_routes = 0
    n_segments = 0
    n_skipped = 0

    try:
        conn.execute("DELETE FROM alpine_routes")
        conn.execute("DELETE FROM terrain_segments")

        for entry in entries:
            gpx_file = entry["gpx_file"]
            gpx_path = os.path.join(GPX_DIR, gpx_file)
            points = load_gpx(gpx_path)
            if points is None:
                n_skipped += 1
                continue

            dist_m, dplus_m = compute_stats(points)
            start = points[0]
            end = points[-1]

            if entry["type"] == "route":
                geojson = route_to_geojson(entry, points)
                grade = entry.get("grade")
                grade_ord = GRADE_ORDINAL.get(grade)

                conn.execute("""
                    INSERT INTO alpine_routes
                        (gpx_path, massif, summit, voie, grade, grade_ord,
                         distance_m, dplus_m, start_lat, start_lon, end_lat, end_lon,
                         geojson, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    gpx_file, entry.get("massif"), entry.get("summit"),
                    entry.get("voie"), grade, grade_ord,
                    round(dist_m, 1), round(dplus_m, 1),
                    start[0], start[1], end[0], end[1],
                    json.dumps(geojson), entry.get("notes"),
                ))
                n_routes += 1

            elif entry["type"] == "segment":
                geojson = route_to_geojson(entry, points)
                conn.execute("""
                    INSERT INTO terrain_segments
                        (gpx_path, start_name, end_name, segment_type,
                         trail_cost, distance_m, dplus_m,
                         start_lat, start_lon, end_lat, end_lon,
                         geojson, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    gpx_file, entry.get("start_name"), entry.get("end_name"),
                    entry.get("segment_type"), entry.get("trail_cost"),
                    round(dist_m, 1), round(dplus_m, 1),
                    start[0], start[1], end[0], end[1],
                    json.dumps(geojson),
                    entry.get("notes"),
                ))
                n_segments += 1

        conn.commit()
    finally:
        conn.close()

    summary = {"routes": n_routes, "segments": n_segments, "skipped": n_skipped}
    logger.info("sync index: %d routes, %d segments, %d skipped", n_routes, n_segments, n_skipped)
    return summary


def reload_index(db_path=None):
    """Wrapper: load_index + sync_to_sqlite."""
    entries = load_index()
    return sync_to_sqlite(entries, db_path=db_path)
