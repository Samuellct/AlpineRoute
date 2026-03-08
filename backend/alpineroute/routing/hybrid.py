# hybrid.py -- assemblage troncons reseau + raster + ponts

import logging
import numpy as np

from alpineroute.config import (
    MAX_ROUTE_POINTS_API,
    BRIDGE_MAX_DISTANCE_M, BRIDGE_DETOUR_RATIO, BRIDGE_BBOX_MARGIN_M,
)
from alpineroute.utils import wgs84_to_l93, l93_to_wgs84, wgs84_to_pixel
from alpineroute.routing.network import haversine_km

log = logging.getLogger(__name__)


def valhalla_to_geojson_feature(valhalla_result, route_index=0):
    """Convertit le resultat Valhalla en GeoJSON Feature compatible pipeline.
    coords Valhalla = [(lat, lon), ...], on passe en [lon, lat, 0]."""
    coords_raw = valhalla_result["coords"]
    dist_km = valhalla_result["distance_km"]
    duration_s = valhalla_result["duration_s"]

    # sous-echantillonnage si trop de points
    n = len(coords_raw)
    if n > MAX_ROUTE_POINTS_API:
        step = max(1, n // MAX_ROUTE_POINTS_API)
        indices = list(range(0, n, step))
        if indices[-1] != n - 1:
            indices.append(n - 1)
        coords_raw = [coords_raw[i] for i in indices]

    # format [lon, lat, elev] -- pas d'altitude Valhalla pour l'instant
    coordinates = [
        [round(lon, 6), round(lat, 6), 0]
        for lat, lon in coords_raw
    ]

    feature = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coordinates},
        "properties": {
            "route_index": route_index,
            "is_optimal": route_index == 0,
            "distance_km": round(dist_km, 2),
            "dplus_m": 0,     # pas de DEM en mode network
            "dminus_m": 0,
            "time_tobler_h": round(duration_s / 3600, 1),
            "glacier_pct": 0,
            "cost_total": 0,
            "n_points": len(coordinates),
            "strategy": "network",
        },
    }
    return feature


def assemble_route(valhalla_coords, raster_feature, transition_point_wgs84):
    """Raccorde un troncon Valhalla et un troncon raster.
    valhalla_coords: [(lat, lon), ...]
    raster_feature: GeoJSON Feature du pathfinding raster
    transition_point_wgs84: (lat, lon) point de jonction
    Retourne un GeoJSON Feature unifie."""
    # coords valhalla -> [lon, lat, 0]
    v_coords = [[round(lon, 6), round(lat, 6), 0] for lat, lon in valhalla_coords]
    r_coords = raster_feature["geometry"]["coordinates"]

    # check distance de raccord
    if v_coords and r_coords:
        last_v = v_coords[-1]
        first_r = r_coords[0]
        gap_km = haversine_km(last_v[1], last_v[0], first_r[1], first_r[0])
        gap_m = gap_km * 1000
        if gap_m > 50:
            log.warning("raccord Valhalla-raster: %.0fm (> 50m)", gap_m)

    merged_coords = v_coords + r_coords

    # merge properties
    v_props = {
        "distance_km": 0,
        "time_tobler_h": 0,
        "dplus_m": 0,
        "dminus_m": 0,
        "glacier_pct": 0,
    }
    r_props = raster_feature.get("properties", {})

    feature = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": merged_coords},
        "properties": {
            "route_index": 0,
            "is_optimal": True,
            "distance_km": round(v_props["distance_km"] + r_props.get("distance_km", 0), 2),
            "dplus_m": round(v_props["dplus_m"] + r_props.get("dplus_m", 0)),
            "dminus_m": round(v_props["dminus_m"] + r_props.get("dminus_m", 0)),
            "time_tobler_h": round(v_props["time_tobler_h"] + r_props.get("time_tobler_h", 0), 1),
            "glacier_pct": r_props.get("glacier_pct", 0),
            "cost_total": r_props.get("cost_total", 0),
            "n_points": len(merged_coords),
            "strategy": "hybrid",
            "transition_point": [
                round(transition_point_wgs84[1], 6),
                round(transition_point_wgs84[0], 6),
            ],
        },
    }
    return feature


def reduce_bbox(exit_wgs84, dest_wgs84, margin_m=500):
    """Bbox L93 minimale pour un troncon hors-piste.
    exit_wgs84, dest_wgs84: (lat, lon)
    Retourne dict {bbox_l93, bbox_wgs84} meme format que compute_bbox()."""
    lat1, lon1 = exit_wgs84
    lat2, lon2 = dest_wgs84

    x1, y1 = wgs84_to_l93(lon1, lat1)
    x2, y2 = wgs84_to_l93(lon2, lat2)

    xmin = min(x1, x2) - margin_m
    xmax = max(x1, x2) + margin_m
    ymin = min(y1, y2) - margin_m
    ymax = max(y1, y2) + margin_m

    bbox_l93 = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}

    lon_min, lat_min = l93_to_wgs84(xmin, ymin)
    lon_max, lat_max = l93_to_wgs84(xmax, ymax)
    bbox_wgs84 = {
        "lon_min": lon_min, "lat_min": lat_min,
        "lon_max": lon_max, "lat_max": lat_max,
    }

    return {"bbox_l93": bbox_l93, "bbox_wgs84": bbox_wgs84}


# --- ponts raster (Phase 7) ---

def detect_detour_segments(valhalla_result, threshold_ratio=None):
    """Detecte les maneuvers Valhalla qui font un detour suspect.
    Retourne liste de candidats pont {start, end, maneuver_index, direct_m}."""
    if threshold_ratio is None:
        threshold_ratio = BRIDGE_DETOUR_RATIO

    maneuvers = valhalla_result.get("maneuvers", [])
    coords = valhalla_result.get("coords", [])
    if not maneuvers or len(coords) < 2:
        return []

    detours = []
    for i, m in enumerate(maneuvers):
        bi = m["begin_shape_index"]
        ei = m["end_shape_index"]
        if bi >= len(coords) or ei >= len(coords) or bi == ei:
            continue

        start_pt = coords[bi]
        end_pt = coords[ei]
        direct_km = haversine_km(start_pt[0], start_pt[1], end_pt[0], end_pt[1])
        direct_m = direct_km * 1000

        # skip segments trop courts ou trop longs
        if direct_m < 10 or direct_m > BRIDGE_MAX_DISTANCE_M:
            continue

        leg_km = m.get("length_km", 0)
        if leg_km <= 0:
            continue

        ratio = leg_km / direct_km
        if ratio > threshold_ratio:
            detours.append({
                "start": start_pt,
                "end": end_pt,
                "maneuver_index": i,
                "direct_m": round(direct_m, 1),
            })

    if detours:
        log.info("detours detectes: %d candidats pont", len(detours))
    return detours


def compute_raster_bridge(start_wgs84, end_wgs84, cost_surface,
                          transform, dem, glacier_mask, resolution):
    """Pathfinding local entre 2 points proches sur la cost surface existante.
    Crop la zone pour eviter de recalculer. Retourne dict ou None."""
    from alpineroute.routing.pathfinding import prepare_cost_grid, run_pathfinding

    try:
        s_row, s_col, _, _ = wgs84_to_pixel(
            start_wgs84[0], start_wgs84[1], transform, cost_surface.shape)
        e_row, e_col, _, _ = wgs84_to_pixel(
            end_wgs84[0], end_wgs84[1], transform, cost_surface.shape)
    except Exception:
        return None

    # fenetre locale autour du pont
    pad = int(BRIDGE_BBOX_MARGIN_M / resolution)
    r_min = max(0, min(s_row, e_row) - pad)
    r_max = min(cost_surface.shape[0], max(s_row, e_row) + pad + 1)
    c_min = max(0, min(s_col, e_col) - pad)
    c_max = min(cost_surface.shape[1], max(s_col, e_col) + pad + 1)

    crop = cost_surface[r_min:r_max, c_min:c_max].copy()
    if crop.size == 0:
        return None

    # coords locales dans le crop
    local_start = (s_row - r_min, s_col - c_min)
    local_end = (e_row - r_min, e_col - c_min)

    try:
        grid = prepare_cost_grid(crop)
        path_coords, path_cost, dt = run_pathfinding(grid, local_start, local_end)
    except Exception:
        return None

    if len(path_coords) == 0:
        return None

    # reconvertir coords locales -> globales -> WGS84
    global_rows = path_coords[:, 0] + r_min
    global_cols = path_coords[:, 1] + c_min

    from alpineroute.utils import pixel_to_l93, reproject_to_wgs84
    x_l93, y_l93 = pixel_to_l93(global_rows, global_cols, transform)
    l93_coords = np.column_stack([x_l93, y_l93])
    wgs84 = reproject_to_wgs84(l93_coords)

    coords_wgs84 = [
        (round(float(wgs84[j, 1]), 6), round(float(wgs84[j, 0]), 6))
        for j in range(len(wgs84))
    ]

    return {
        "coords_wgs84": coords_wgs84,
        "path_coords_global": np.column_stack([global_rows, global_cols]),
        "n_points": len(coords_wgs84),
        "cost": float(path_cost),
    }


def apply_bridges(valhalla_result, detours, bridge_results):
    """Remplace les troncons de detour par les ponts raster.
    Retourne un nouveau dict avec coords assemblees."""
    coords = list(valhalla_result["coords"])
    maneuvers = valhalla_result.get("maneuvers", [])

    # construire segments a remplacer, tries par index decroissant
    # pour pas decaler les indices
    replacements = []
    for det, br in zip(detours, bridge_results):
        if br is None:
            continue
        mi = det["maneuver_index"]
        if mi >= len(maneuvers):
            continue
        m = maneuvers[mi]
        replacements.append((m["begin_shape_index"], m["end_shape_index"], br))

    if not replacements:
        return None

    replacements.sort(key=lambda x: x[0], reverse=True)

    for bi, ei, br in replacements:
        bridge_coords = br["coords_wgs84"]
        coords[bi:ei + 1] = bridge_coords

    n_bridges = len(replacements)
    log.info("ponts raster appliques: %d segments remplaces", n_bridges)

    return {
        "coords": coords,
        "distance_km": valhalla_result.get("distance_km", 0),
        "duration_s": valhalla_result.get("duration_s", 0),
        "n_bridges": n_bridges,
    }
