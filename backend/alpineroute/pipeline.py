# pipeline integre -- chaine bout-en-bout de la requete au GeoJSON
# remplace le chargement au startup

import json
import math
import logging
import numpy as np

from alpineroute.config import (
    MAX_GRID_PIXELS, MAX_GRID_PIXELS_ANISO,
    MAX_ROUTE_POINTS_API, NODATA_VALUE, DB_PATH,
)
from alpineroute.utils import (
    compute_bbox, load_dem, wgs84_to_pixel, pixel_to_l93,
    compute_path_stats, reproject_to_wgs84,
    PointOutOfBoundsError, DownloadError,
)
from alpineroute.dem.download import get_dem
from alpineroute.dem.terrain import compute_slope_aspect, compute_roughness
from alpineroute.cost.landcover import get_landcover_cost
from alpineroute.cost.glacier import get_glacier_mask
from alpineroute.cost.surface import build_cost_surface, build_base_cost
from alpineroute.cost.zones import rasterize_user_zones
from alpineroute.routing.pathfinding import (
    prepare_cost_grid, run_pathfinding, run_pathfinding_alternatives,
    dijkstra_anisotropic, run_aniso_alternatives,
)
from alpineroute.db.crud import list_zones, save_route

logger = logging.getLogger(__name__)

# cache de la derniere surface de cout (pour le calque front)
_last_cost_surface: dict = {}


# poids SSE par etape (total = 100%)
_STEP_WEIGHTS = {
    "bbox": 0,
    "dem": 40,
    "terrain": 15,
    "worldcover": 5,
    "glacier": 5,
    "cost": 5,
    "zones": 2,
    "pathfinding": 23,
    "result": 5,
}


def _progress(callback, step, pct_within_step=1.0):
    """Envoie le progress SSE. pct_within_step = 0..1 dans l'etape courante."""
    if callback is None:
        return

    steps = list(_STEP_WEIGHTS.keys())
    idx = steps.index(step)
    base = sum(_STEP_WEIGHTS[s] for s in steps[:idx])
    current = base + _STEP_WEIGHTS[step] * pct_within_step
    callback(int(current), step)


def _wrap_dem_progress(callback):
    """Convertit le callback get_dem(step, i, n) en progress global."""
    if callback is None:
        return None

    def _dem_cb(step, current, total):
        if total > 0:
            pct = current / total
        else:
            pct = 0
        _progress(callback, "dem", pct)

    return _dem_cb


def _build_route_feature(path_coords, dem, transform, glacier_mask,
                         resolution, route_index=0, path_cost=0):
    """Construit un GeoJSON Feature + stats pour un trajet.
    Retourne (feature_dict, stats_dict)."""
    stats, arrays = compute_path_stats(
        path_coords, dem, transform, glacier_mask)

    # reprojection WGS84
    l93_coords = np.column_stack([arrays["x_l93"], arrays["y_l93"]])
    wgs84 = reproject_to_wgs84(l93_coords)

    lons, lats = wgs84[:, 0], wgs84[:, 1]
    elevations = arrays["elevations"]

    # sous-echantillonnage si trop de points
    n = len(lons)
    if n > MAX_ROUTE_POINTS_API:
        step = max(1, n // MAX_ROUTE_POINTS_API)
        idx = np.arange(0, n, step)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)
        lons, lats, elevations = lons[idx], lats[idx], elevations[idx]

    coordinates = [
        [round(float(lo), 6), round(float(la), 6), round(float(el), 1)]
        for lo, la, el in zip(lons, lats, elevations)
    ]

    feature = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coordinates},
        "properties": {
            "route_index": route_index,
            "is_optimal": route_index == 0,
            "distance_km": round(stats["dist_2d_m"] / 1000, 2),
            "dplus_m": round(stats["dplus"]),
            "dminus_m": round(stats["dminus"]),
            "time_tobler_h": round(stats["time_tobler_h"], 1),
            "glacier_pct": round(stats["glacier_pct"], 1),
            "cost_total": round(path_cost, 2),
            "n_points": len(coordinates),
            "resolution_m": resolution,
        },
    }

    return feature, stats


def run_pipeline(req, progress_callback=None):
    """Pipeline complet: coords -> GeoJSON result.
    req: RouteRequest (pydantic model)
    progress_callback(pct: int, step: str): pour SSE
    """
    # -- 1. bbox
    _progress(progress_callback, "bbox")
    start = (req.start_lat, req.start_lon)
    end = (req.end_lat, req.end_lon)
    bboxes = compute_bbox(start, end)
    bbox_l93 = bboxes["bbox_l93"]
    bbox_wgs84 = bboxes["bbox_wgs84"]

    logger.info("bbox L93: %s", bbox_l93)

    # estim rapide de la taille grille avant de telecharger
    width_m = bbox_l93["xmax"] - bbox_l93["xmin"]
    height_m = bbox_l93["ymax"] - bbox_l93["ymin"]
    est_pixels = int((width_m / req.resolution) * (height_m / req.resolution))

    if est_pixels > MAX_GRID_PIXELS:
        bbox_area = width_m * height_m
        rec_res = math.ceil(math.sqrt(bbox_area / MAX_GRID_PIXELS))
        raise ValueError(
            f"grille estimee trop grande: ~{est_pixels/1e6:.0f}M px "
            f"(max {MAX_GRID_PIXELS/1e6:.0f}M). "
            f"Reduire la zone ou augmenter la resolution (recommande: {rec_res}m+)."
        )

    use_aniso = getattr(req, 'anisotropic', False)
    if use_aniso and est_pixels > MAX_GRID_PIXELS_ANISO:
        bbox_area = width_m * height_m
        rec_res = math.ceil(math.sqrt(bbox_area / MAX_GRID_PIXELS_ANISO))
        raise ValueError(
            f"grille estimee trop grande pour le mode precis: ~{est_pixels/1e6:.0f}M px "
            f"(max {MAX_GRID_PIXELS_ANISO/1e6:.0f}M). "
            f"Resolution recommandee: {rec_res}m ou plus."
        )

    # -- 2. DEM download/mosaic
    _progress(progress_callback, "dem", 0)
    dem_cb = _wrap_dem_progress(progress_callback)
    dem_path = get_dem(bbox_l93, resolution=req.resolution,
                       progress_callback=dem_cb, bbox_wgs84=bbox_wgs84)
    _progress(progress_callback, "dem", 1.0)

    # charger le DEM en memoire
    dem, profile, transform = load_dem(dem_path)
    logger.info("DEM charge: %s (%.1f MP)", dem.shape,
                dem.size / 1e6)

    # garde-fou post-load (au cas ou l'estim pre-vol etait optimiste)
    if dem.size > MAX_GRID_PIXELS:
        rec_res = math.ceil(math.sqrt(dem.size * req.resolution**2 / MAX_GRID_PIXELS))
        raise ValueError(
            f"grille trop grande: {dem.size/1e6:.0f}M px (max {MAX_GRID_PIXELS/1e6:.0f}M). "
            f"Reduire la zone ou passer a {rec_res}m+."
        )

    if use_aniso and dem.size > MAX_GRID_PIXELS_ANISO:
        rec_res = math.ceil(math.sqrt(dem.size * req.resolution**2 / MAX_GRID_PIXELS_ANISO))
        raise ValueError(
            f"grille trop grande pour le mode precis: {dem.size/1e6:.0f}M px "
            f"(max {MAX_GRID_PIXELS_ANISO/1e6:.0f}M). "
            f"Resolution recommandee: {rec_res}m ou plus."
        )

    # -- 3. terrain analysis
    _progress(progress_callback, "terrain", 0)
    slope, aspect = compute_slope_aspect(dem, req.resolution)
    _progress(progress_callback, "terrain", 0.7)
    roughness = compute_roughness(dem)
    _progress(progress_callback, "terrain", 1.0)

    # -- 4. worldcover (optionnel)
    _progress(progress_callback, "worldcover", 0)
    landcover = get_landcover_cost(bbox_wgs84, bbox_l93, dem.shape)
    _progress(progress_callback, "worldcover", 1.0)

    # -- 5. glacier (optionnel)
    _progress(progress_callback, "glacier", 0)
    glacier_mask = get_glacier_mask(bbox_l93, transform, dem.shape)
    _progress(progress_callback, "glacier", 1.0)

    # -- 6. cost surface
    _progress(progress_callback, "cost", 0)

    if use_aniso:
        # mode anisotrope: base cost sans Tobler (calcule per-edge)
        base_cost, nodata_mask = build_base_cost(
            dem, slope, aspect, roughness, glacier_mask,
            month=req.month, acclimatized=req.acclimatized,
            landcover_cost=landcover,
        )
        # on garde aussi la cost surface complete pour le calque
        cost_for_display, factors, _ = build_cost_surface(
            dem, slope, aspect, roughness, glacier_mask,
            month=req.month, acclimatized=req.acclimatized,
            landcover_cost=landcover,
        )
        del factors
    else:
        # mode isotrope: surface complete
        cost_for_display, factors, nodata_mask = build_cost_surface(
            dem, slope, aspect, roughness, glacier_mask,
            month=req.month, acclimatized=req.acclimatized,
            landcover_cost=landcover,
        )
        del factors
        base_cost = None

    del slope, aspect, roughness, landcover
    _progress(progress_callback, "cost", 1.0)

    # -- 6b. zones utilisateur
    _progress(progress_callback, "zones", 0)
    try:
        user_zones = list_zones(DB_PATH, active_only=True)
    except Exception:
        user_zones = []

    if user_zones:
        zone_grid = rasterize_user_zones(user_zones, transform, cost_for_display.shape)
        if zone_grid is not None:
            valid = cost_for_display != NODATA_VALUE
            cost_for_display[valid] *= zone_grid[valid]
            if base_cost is not None:
                valid_base = np.isfinite(base_cost)
                base_cost[valid_base] *= zone_grid[valid_base]
            logger.info("zones appliquees: %d zones actives", len(user_zones))
            del zone_grid
    _progress(progress_callback, "zones", 1.0)

    # cache la surface de cout pour l'endpoint /cost-surface
    global _last_cost_surface
    _last_cost_surface = {
        "cost": cost_for_display.copy(),
        "bbox_wgs84": bbox_wgs84,
        "transform": transform,
        "shape": cost_for_display.shape,
    }

    # -- 7. pathfinding
    _progress(progress_callback, "pathfinding", 0)

    start_row, start_col, _, _ = wgs84_to_pixel(
        req.start_lat, req.start_lon, transform, cost_for_display.shape)
    end_row, end_col, _, _ = wgs84_to_pixel(
        req.end_lat, req.end_lon, transform, cost_for_display.shape)

    start_rc = (start_row, start_col)
    end_rc = (end_row, end_col)

    if use_aniso:
        # pathfinding anisotrope
        if req.n_alternatives > 0:
            all_results = run_aniso_alternatives(
                dem, base_cost, start_rc, end_rc,
                req.resolution, n_alt=req.n_alternatives)
        else:
            pc, pc_cost, dt = dijkstra_anisotropic(
                dem, base_cost, start_rc, end_rc, req.resolution)
            all_results = [(pc, pc_cost, dt)]
        del base_cost
    else:
        # pathfinding isotrope classique
        cost_grid = prepare_cost_grid(cost_for_display)
        if req.n_alternatives > 0:
            all_results = run_pathfinding_alternatives(
                cost_grid, start_rc, end_rc, n_alt=req.n_alternatives)
        else:
            pc, pc_cost, dt = run_pathfinding(cost_grid, start_rc, end_rc)
            all_results = [(pc, pc_cost, dt)]
        del cost_grid

    del cost_for_display

    if len(all_results) == 0 or len(all_results[0][0]) == 0:
        raise RuntimeError("pathfinding a retourne un chemin vide")

    _progress(progress_callback, "pathfinding", 1.0)

    # -- 8. construction des features
    _progress(progress_callback, "result", 0)

    routes = []
    total_dt = 0
    for i, (path_coords, path_cost, dt) in enumerate(all_results):
        if len(path_coords) == 0:
            continue
        feature, stats = _build_route_feature(
            path_coords, dem, transform, glacier_mask,
            req.resolution, route_index=i, path_cost=path_cost)
        routes.append(feature)
        total_dt += dt

    _progress(progress_callback, "result", 0.5)

    # -- 9. auto-save route optimale
    saved_route_id = None
    if req.save and routes:
        try:
            optimal = routes[0]
            props = optimal["properties"]
            route_data = {
                "name": req.name,
                "start_lat": req.start_lat,
                "start_lon": req.start_lon,
                "end_lat": req.end_lat,
                "end_lon": req.end_lon,
                "resolution": req.resolution,
                "month": req.month,
                "acclimatized": req.acclimatized,
                "distance_m": props["distance_km"] * 1000,
                "dplus_m": props["dplus_m"],
                "dminus_m": props["dminus_m"],
                "time_tobler_h": props["time_tobler_h"],
                "glacier_pct": props["glacier_pct"],
                "cost_total": props.get("cost_total"),
                "computation_time_s": round(total_dt, 1),
                "geojson": json.dumps(optimal),
            }
            saved_route_id = save_route(DB_PATH, route_data)
        except Exception as e:
            # save fail = pas bloquant, on log et on continue
            logger.warning("auto-save failed: %s", e)

    _progress(progress_callback, "result", 1.0)

    result = {
        "status": "ok",
        "route": routes[0] if routes else None,
        "computation_time_s": round(total_dt, 1),
    }

    if len(routes) > 1:
        result["routes"] = routes
        result["n_routes"] = len(routes)

    if saved_route_id is not None:
        result["saved_route_id"] = saved_route_id

    return result
