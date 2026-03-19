# segments terrain connu -- rasterisation GPX custom + merge avec OSM

import os
import logging

import numpy as np
from shapely.geometry import LineString
from rasterio.features import rasterize

from alpineroute.config import GPX_DIR, DB_PATH
from alpineroute.db.schema import get_connection
from alpineroute.alpine.routes import load_gpx
from alpineroute.utils import wgs84_to_l93, l93_to_wgs84

logger = logging.getLogger(__name__)


def load_segments_for_bbox(bbox_l93, db_path=None):
    """Recupere les segments dont au moins un endpoint tombe dans la bbox.
    Convertit bbox L93 -> WGS84, filtre sur start/end lat/lon."""
    if db_path is None:
        db_path = DB_PATH

    # bbox L93 -> WGS84
    lon_min, lat_min = l93_to_wgs84(bbox_l93["xmin"], bbox_l93["ymin"])
    lon_max, lat_max = l93_to_wgs84(bbox_l93["xmax"], bbox_l93["ymax"])

    conn = get_connection(db_path)
    try:
        conn.row_factory = _dict_factory
        rows = conn.execute("""
            SELECT id, gpx_path, segment_type, trail_cost,
                   start_lat, start_lon, end_lat, end_lon
            FROM terrain_segments
            WHERE (
                (start_lat BETWEEN ? AND ? AND start_lon BETWEEN ? AND ?)
                OR (end_lat BETWEEN ? AND ? AND end_lon BETWEEN ? AND ?)
            )
        """, (
            lat_min, lat_max, lon_min, lon_max,
            lat_min, lat_max, lon_min, lon_max,
        )).fetchall()
        return rows
    finally:
        conn.close()


def rasterize_segments(segments, transform, shape, resolution=1.0):
    """Rasterise les segments GPX custom sur la grille DEM.
    Retourne un float32 (1.0 = pas de segment, < 1.0 = segment)."""
    if not segments:
        return np.ones(shape, dtype=np.float32)

    buffered = []
    for seg in segments:
        gpx_path = os.path.join(GPX_DIR, seg["gpx_path"])
        points = load_gpx(gpx_path)
        if points is None:
            continue

        # pts WGS84 -> L93 pour la rasterisation
        coords_l93 = []
        for lat, lon, _alt in points:
            x, y = wgs84_to_l93(lon, lat)
            coords_l93.append((x, y))

        if len(coords_l93) < 2:
            continue

        line = LineString(coords_l93)
        # buffer 4m autour de la trace (avant 2m, trop etroit a 1m de resolution)
        geom = line.buffer(4.0)
        if geom.is_empty:
            continue

        cost = seg.get("trail_cost")
        if cost is None:
            cost = 0.3  # default ~sentier classique
        buffered.append((geom, float(cost)))

    if not buffered:
        return np.ones(shape, dtype=np.float32)

    # tri decroissant: le meilleur cout gagne (ecrase en dernier)
    buffered.sort(key=lambda x: x[1], reverse=True)

    result = rasterize(
        buffered,
        out_shape=shape,
        transform=transform,
        fill=1.0,
        dtype=np.float32,
        all_touched=True,
    )

    n_px = np.sum(result < 1.0)
    logger.info("segments terrain: %d segments, %d px rasterises", len(buffered), n_px)
    return result


def merge_trail_layers(osm_trail_cost, segment_trail_cost):
    """Fusionne OSM trails et segments custom: le meilleur cout gagne."""
    if osm_trail_cost is None:
        return segment_trail_cost
    if segment_trail_cost is None:
        return osm_trail_cost
    return np.minimum(osm_trail_cost, segment_trail_cost)


def _dict_factory(cursor, row):
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}
