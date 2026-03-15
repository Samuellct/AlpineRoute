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


def assemble_route(valhalla_coords, raster_feature, transition_point_wgs84,
                   valhalla_stats=None, order="valhalla_first"):
    """Raccorde un troncon Valhalla et un troncon raster.
    valhalla_coords: [(lat, lon), ...]
    raster_feature: GeoJSON Feature du pathfinding raster
    transition_point_wgs84: (lat, lon) point de jonction
    valhalla_stats: dict avec distance_km/duration_s du troncon Valhalla (optionnel)
    order: "valhalla_first" ou "raster_first"
    Retourne un GeoJSON Feature unifie."""
    # coords valhalla -> [lon, lat, 0]
    v_coords = [[round(lon, 6), round(lat, 6), 0] for lat, lon in valhalla_coords]
    r_coords = raster_feature["geometry"]["coordinates"]

    if order == "raster_first":
        # check gap raster -> valhalla
        if r_coords and v_coords:
            last_r = r_coords[-1]
            first_v = v_coords[0]
            gap_km = haversine_km(last_r[1], last_r[0], first_v[1], first_v[0])
            gap_m = gap_km * 1000
            if gap_m > 50:
                log.warning("raccord raster-Valhalla: %.0fm (> 50m)", gap_m)
        merged_coords = r_coords + v_coords
    else:
        # check distance de raccord valhalla -> raster
        if v_coords and r_coords:
            last_v = v_coords[-1]
            first_r = r_coords[0]
            gap_km = haversine_km(last_v[1], last_v[0], first_r[1], first_r[0])
            gap_m = gap_km * 1000
            if gap_m > 50:
                log.warning("raccord Valhalla-raster: %.0fm (> 50m)", gap_m)
        merged_coords = v_coords + r_coords

    # merge properties
    if valhalla_stats:
        v_props = {
            "distance_km": valhalla_stats.get("distance_km", 0),
            "time_tobler_h": round(valhalla_stats.get("duration_s", 0) / 3600, 1),
            "dplus_m": 0, "dminus_m": 0, "glacier_pct": 0,
        }
    else:
        v_props = {"distance_km": 0, "time_tobler_h": 0, "dplus_m": 0, "dminus_m": 0, "glacier_pct": 0}
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


# --- CAS B : routage hybride (approche reseau + hors-piste terminal) ---

def find_network_exit(start, end, max_snap_m=None):
    """CAS B principal : destination off-network.
    Cherche le point de sortie reseau le plus proche de la dest,
    puis calcule l'approche Valhalla de start a ce point.
    Rejette l'approche si elle ne rapproche pas de la dest.
    Retourne dict {exit_point, approach, snap_m} ou None."""
    from alpineroute.routing.network import (
        valhalla_locate, valhalla_route, parse_locate_snap, haversine_km as _hav,
    )
    from alpineroute.config import SNAP_MAX_DISTANCE_M

    if max_snap_m is None:
        max_snap_m = SNAP_MAX_DISTANCE_M

    loc = valhalla_locate(end)
    snap_pt = parse_locate_snap(loc)
    if snap_pt is None:
        return None

    # verif distance snap /locate -> dest
    snap_m = _hav(end[0], end[1], snap_pt[0], snap_pt[1]) * 1000
    if snap_m > max_snap_m:
        log.info("find_network_exit: snap trop loin (%.0fm > %.0fm)", snap_m, max_snap_m)
        return None

    try:
        vr = valhalla_route(start, snap_pt)
    except Exception as e:
        log.warning("find_network_exit: valhalla_route echec: %s", e)
        return None
    if vr is None:
        return None

    actual_exit = vr["coords"][-1] if vr["coords"] else snap_pt
    actual_snap_m = _hav(end[0], end[1], actual_exit[0], actual_exit[1]) * 1000

    # verif: l'approche doit rapprocher de la destination
    # sinon c'est un detour inutile (ex: start deja pres du reseau glaciaire,
    # l'approche monte au Montenvers puis redescend)
    direct_km = _hav(start[0], start[1], end[0], end[1])
    exit_to_end_km = _hav(actual_exit[0], actual_exit[1], end[0], end[1])
    if exit_to_end_km >= direct_km * 0.85:
        log.info("find_network_exit: approche inutile (exit %.1fkm vs direct %.1fkm), skip",
                 exit_to_end_km, direct_km)
        return None

    log.info("find_network_exit: exit=(%.5f,%.5f), locate_snap=%.0fm, "
             "actual_exit_to_dest=%.0fm, direct=%.2fkm",
             actual_exit[0], actual_exit[1], snap_m, actual_snap_m,
             direct_km)

    return {"exit_point": actual_exit, "approach": vr, "snap_m": actual_snap_m}


def find_network_entry(start, end, max_snap_m=None):
    """CAS B symetrique : depart off-network.
    Cherche le point d'entree reseau le plus proche du depart,
    puis calcule la continuation Valhalla de ce point vers end.
    Retourne dict {entry_point, continuation, snap_m} ou None."""
    from alpineroute.routing.network import (
        valhalla_locate, valhalla_route, parse_locate_snap, haversine_km as _hav,
    )
    from alpineroute.config import SNAP_MAX_DISTANCE_M

    if max_snap_m is None:
        max_snap_m = SNAP_MAX_DISTANCE_M

    loc = valhalla_locate(start)
    snap_pt = parse_locate_snap(loc)
    if snap_pt is None:
        return None

    snap_m = _hav(start[0], start[1], snap_pt[0], snap_pt[1]) * 1000
    if snap_m > max_snap_m:
        log.info("find_network_entry: snap trop loin (%.0fm > %.0fm)", snap_m, max_snap_m)
        return None

    try:
        vr = valhalla_route(snap_pt, end)
    except Exception as e:
        log.warning("find_network_entry: valhalla_route echec: %s", e)
        return None
    if vr is None:
        return None

    # premier point reel de la route (Valhalla re-snappe en interne)
    actual_entry = vr["coords"][0] if vr["coords"] else snap_pt
    actual_snap_m = _hav(start[0], start[1], actual_entry[0], actual_entry[1]) * 1000

    # verif: le dernier point doit etre proche de la destination
    # sinon la continuation ne sert a rien (dest off-network)
    actual_end = vr["coords"][-1] if vr["coords"] else snap_pt
    end_snap_m = _hav(end[0], end[1], actual_end[0], actual_end[1]) * 1000
    if end_snap_m > max_snap_m:
        log.info("find_network_entry: continuation n'atteint pas la dest "
                 "(%.0fm > %.0fm), skip", end_snap_m, max_snap_m)
        return None

    return {"entry_point": actual_entry, "continuation": vr, "snap_m": actual_snap_m}


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


# --- assemblage GPX overlay (Phase 11) ---

def assemble_gpx_route(approach_vr, gpx_result, egress_vr, route_index=0):
    """Raccorde 3 troncons : Valhalla approche + GPX milieu + Valhalla sortie.
    approach_vr, egress_vr: dict Valhalla (ou None)
    gpx_result: dict de route_via_gpx()"""
    merged_coords = []
    transition_points = []

    # troncon approche
    approach_dist_km = 0
    approach_time_s = 0
    if approach_vr is not None:
        v_coords = [
            [round(lon, 6), round(lat, 6), 0]
            for lat, lon in approach_vr["coords"]
        ]
        merged_coords.extend(v_coords)
        approach_dist_km = approach_vr.get("distance_km", 0)
        approach_time_s = approach_vr.get("duration_s", 0)

        # check gap approche -> GPX
        if v_coords and gpx_result["gpx_coords"]:
            last_v = v_coords[-1]  # [lon, lat, 0]
            first_g = gpx_result["gpx_coords"][0]  # (lat, lon, alt)
            gap_m = haversine_km(last_v[1], last_v[0], first_g[0], first_g[1]) * 1000
            if gap_m > 50:
                log.warning("raccord approche-GPX: %.0fm (> 50m)", gap_m)
            transition_points.append([round(first_g[1], 6), round(first_g[0], 6)])

    # troncon GPX
    gpx_coords = [
        [round(lon, 6), round(lat, 6), round(alt, 1)]
        for lat, lon, alt in gpx_result["gpx_coords"]
    ]
    merged_coords.extend(gpx_coords)

    # troncon sortie
    egress_dist_km = 0
    egress_time_s = 0
    if egress_vr is not None:
        e_coords = [
            [round(lon, 6), round(lat, 6), 0]
            for lat, lon in egress_vr["coords"]
        ]

        # check gap GPX -> sortie
        if gpx_result["gpx_coords"] and e_coords:
            last_g = gpx_result["gpx_coords"][-1]
            first_e = e_coords[0]
            gap_m = haversine_km(last_g[0], last_g[1], first_e[1], first_e[0]) * 1000
            if gap_m > 50:
                log.warning("raccord GPX-sortie: %.0fm (> 50m)", gap_m)
            transition_points.append([round(last_g[1], 6), round(last_g[0], 6)])

        merged_coords.extend(e_coords)
        egress_dist_km = egress_vr.get("distance_km", 0)
        egress_time_s = egress_vr.get("duration_s", 0)

    # estim temps Tobler GPX
    from alpineroute.routing.gpx_graph import _estimate_tobler_time
    gpx_time_h = _estimate_tobler_time(gpx_result["gpx_coords"])

    total_dist_km = approach_dist_km + gpx_result["distance_km"] + egress_dist_km
    total_time_h = (approach_time_s + egress_time_s) / 3600 + gpx_time_h

    feature = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": merged_coords},
        "properties": {
            "route_index": route_index,
            "is_optimal": route_index == 0,
            "distance_km": round(total_dist_km, 2),
            "dplus_m": gpx_result["dplus_m"],
            "dminus_m": gpx_result["dminus_m"],
            "time_tobler_h": round(total_time_h, 1),
            "glacier_pct": 0,
            "cost_total": 0,
            "n_points": len(merged_coords),
            "strategy": "gpx_hybrid",
            "gpx_sources": gpx_result["gpx_sources"],
            "transition_points": transition_points,
        },
    }
    return feature
