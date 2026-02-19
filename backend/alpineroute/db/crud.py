# operations CRUD pour SQLite
# routes + zones utilisateur + cache DEM

import json
import logging
from alpineroute.db.schema import get_connection

logger = logging.getLogger(__name__)


# =====================================================
#  Routes
# =====================================================

def save_route(db_path, route_data):
    """Insere une route calculee, retourne l'ID."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute("""
            INSERT INTO routes (name, start_lat, start_lon, end_lat, end_lon,
                               dem_source, dem_resolution, season_month, acclimatized,
                               distance_m, dplus_m, dminus_m, time_tobler_h,
                               glacier_pct, cost_total, computation_time_s, geojson)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            route_data.get("name"),
            route_data["start_lat"], route_data["start_lon"],
            route_data["end_lat"], route_data["end_lon"],
            route_data.get("dem_source", "lidar_hd"),
            route_data.get("resolution", 1.0),
            route_data.get("month", 7),
            1 if route_data.get("acclimatized", True) else 0,
            route_data.get("distance_m"),
            route_data.get("dplus_m"),
            route_data.get("dminus_m"),
            route_data.get("time_tobler_h"),
            route_data.get("glacier_pct"),
            route_data.get("cost_total"),
            route_data.get("computation_time_s"),
            route_data.get("geojson"),
        ))
        conn.commit()
        route_id = cur.lastrowid
        logger.info("route saved id=%d: %s", route_id, route_data.get("name"))
        return route_id
    finally:
        conn.close()


def get_route(db_path, route_id):
    """Recupere une route par ID. Retourne dict ou None."""
    conn = get_connection(db_path)
    try:
        conn.row_factory = _dict_factory
        row = conn.execute(
            "SELECT * FROM routes WHERE id = ?", (route_id,)
        ).fetchone()
        return row
    finally:
        conn.close()


def list_routes(db_path, bbox=None, date_from=None, date_to=None,
                min_distance=None, max_distance=None,
                limit=50, offset=0):
    """Liste les routes avec filtres optionnels.
    bbox = dict(lon_min, lat_min, lon_max, lat_max) filtre sur start/end.
    Ne retourne pas le geojson (trop lourd pour la liste)."""
    conn = get_connection(db_path)
    try:
        conn.row_factory = _dict_factory
        conditions = []
        params = []

        if bbox:
            # au moins un des deux points (start ou end) dans la bbox
            conditions.append("""(
                (start_lon BETWEEN ? AND ? AND start_lat BETWEEN ? AND ?)
                OR (end_lon BETWEEN ? AND ? AND end_lat BETWEEN ? AND ?)
            )""")
            params.extend([
                bbox["lon_min"], bbox["lon_max"], bbox["lat_min"], bbox["lat_max"],
                bbox["lon_min"], bbox["lon_max"], bbox["lat_min"], bbox["lat_max"],
            ])

        if date_from:
            conditions.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("created_at <= ?")
            params.append(date_to)
        if min_distance is not None:
            conditions.append("distance_m >= ?")
            params.append(min_distance)
        if max_distance is not None:
            conditions.append("distance_m <= ?")
            params.append(max_distance)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        # pas de geojson ni elevation_profile dans la liste
        sql = f"""
            SELECT id, created_at, name, start_lat, start_lon, end_lat, end_lon,
                   dem_source, dem_resolution, season_month, acclimatized,
                   distance_m, dplus_m, dminus_m, time_tobler_h,
                   glacier_pct, cost_total, computation_time_s
            FROM routes {where}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        rows = conn.execute(sql, params).fetchall()
        return rows
    finally:
        conn.close()


def delete_route(db_path, route_id):
    """Supprime une route par ID. Retourne True si supprimee."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute("DELETE FROM routes WHERE id = ?", (route_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# =====================================================
#  Zones utilisateur
# =====================================================

def save_zone(db_path, zone_data):
    """Insere une zone, retourne l'ID."""
    conn = get_connection(db_path)
    try:
        geojson_str = zone_data.get("geojson")
        if isinstance(geojson_str, dict):
            geojson_str = json.dumps(geojson_str)

        cur = conn.execute("""
            INSERT INTO user_zones (name, zone_type, cost_multiplier, geojson, active)
            VALUES (?, ?, ?, ?, ?)
        """, (
            zone_data["name"],
            zone_data["zone_type"],
            zone_data.get("cost_multiplier", 100.0),
            geojson_str,
            1 if zone_data.get("active", True) else 0,
        ))
        conn.commit()
        zone_id = cur.lastrowid
        logger.info("zone saved id=%d: %s (%s)", zone_id,
                     zone_data["name"], zone_data["zone_type"])
        return zone_id
    finally:
        conn.close()


def list_zones(db_path, zone_type=None, active_only=False):
    """Liste les zones avec filtres optionnels."""
    conn = get_connection(db_path)
    try:
        conn.row_factory = _dict_factory
        conditions = []
        params = []

        if zone_type:
            conditions.append("zone_type = ?")
            params.append(zone_type)
        if active_only:
            conditions.append("active = 1")

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        rows = conn.execute(
            f"SELECT * FROM user_zones {where} ORDER BY created_at DESC",
            params
        ).fetchall()

        # parse le geojson stocke en TEXT
        for row in rows:
            if row.get("geojson") and isinstance(row["geojson"], str):
                try:
                    row["geojson"] = json.loads(row["geojson"])
                except json.JSONDecodeError:
                    pass

        return rows
    finally:
        conn.close()


def get_zone(db_path, zone_id):
    """Recupere une zone par ID."""
    conn = get_connection(db_path)
    try:
        conn.row_factory = _dict_factory
        row = conn.execute(
            "SELECT * FROM user_zones WHERE id = ?", (zone_id,)
        ).fetchone()
        if row and isinstance(row.get("geojson"), str):
            try:
                row["geojson"] = json.loads(row["geojson"])
            except json.JSONDecodeError:
                pass
        return row
    finally:
        conn.close()


def update_zone(db_path, zone_id, updates):
    """Mise a jour partielle d'une zone (champs fournis seulement)."""
    allowed = {"name", "zone_type", "cost_multiplier", "geojson", "active"}
    fields = {k: v for k, v in updates.items() if k in allowed and v is not None}

    if not fields:
        return False

    if "geojson" in fields and isinstance(fields["geojson"], dict):
        fields["geojson"] = json.dumps(fields["geojson"])
    if "active" in fields:
        fields["active"] = 1 if fields["active"] else 0

    conn = get_connection(db_path)
    try:
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        params = list(fields.values()) + [zone_id]
        cur = conn.execute(
            f"UPDATE user_zones SET {set_clause} WHERE id = ?", params
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_zone(db_path, zone_id):
    """Supprime une zone par ID."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute("DELETE FROM user_zones WHERE id = ?", (zone_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# =====================================================
#  Cache DEM (inchange)
# =====================================================

def register_cached_tile(db_path, source, tile_name, bbox, resolution, file_path, file_size):
    """Enregistre une tuile DEM dans le cache."""
    conn = get_connection(db_path)
    try:
        conn.execute("""
            INSERT OR IGNORE INTO dem_cache
                (source, tile_name, xmin, ymin, xmax, ymax, resolution, file_path, file_size_bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (source, tile_name,
              bbox.get("xmin"), bbox.get("ymin"), bbox.get("xmax"), bbox.get("ymax"),
              resolution, file_path, file_size))
        conn.commit()
    finally:
        conn.close()


def get_cached_tiles(db_path, source, resolution):
    """Liste les tuiles en cache."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT tile_name, file_path FROM dem_cache WHERE source=? AND resolution=?",
            (source, resolution)
        ).fetchall()
        return {name: path for name, path in rows}
    finally:
        conn.close()


# =====================================================
#  Helpers
# =====================================================

def _dict_factory(cursor, row):
    """Row factory pour retourner des dicts au lieu de tuples."""
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}
