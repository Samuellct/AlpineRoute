# pipeline integre -- chaine bout-en-bout de la requete au GeoJSON
# remplace le chargement au startup

import json
import math
import logging
import numpy as np

from alpineroute.config import (
    MAX_GRID_PIXELS, MAX_GRID_PIXELS_ANISO,
    MAX_ROUTE_POINTS_API, NODATA_VALUE, DB_PATH,
    ISOTROPIC_WARNING_DPLUS_M, STREAM_CROSSING_PENALTY,
    SNAP_MAX_DISTANCE_M, GHOST_ROUTE_MIN_DISTANCE_KM, HYBRID_BBOX_MARGIN_M,
    VALHALLA_COVERAGE_BBOX,
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
from alpineroute.cost.trails import get_trail_cost
from alpineroute.cost.barriers import get_barrier_masks
from alpineroute.cost.zones import rasterize_user_zones
from alpineroute.routing.pathfinding import (
    prepare_cost_grid, run_pathfinding, run_pathfinding_alternatives,
    dijkstra_anisotropic, run_aniso_alternatives,
)
from alpineroute.routing.network import (
    valhalla_available, valhalla_route, is_detour_excessive, haversine_km,
)
from alpineroute.routing.hybrid import (
    valhalla_to_geojson_feature, detect_detour_segments,
    compute_raster_bridge, apply_bridges,
    assemble_route, reduce_bbox,
    find_network_exit, find_network_entry,
    assemble_gpx_route,
)
from alpineroute.routing.gpx_graph import route_via_gpx, gpx_to_geojson_feature
from alpineroute.alpine.segments import (
    load_segments_for_bbox, rasterize_segments, merge_trail_layers,
)
from alpineroute.db.crud import list_zones, save_route
from alpineroute.utils import ValhallaError

logger = logging.getLogger(__name__)

# cache de la derniere surface de cout (pour le calque front)
_last_cost_surface: dict = {}


# poids SSE par etape (total = 100%)
_STEP_WEIGHTS = {
    "network": 5,
    "gpx_graph": 2,
    "bbox": 0,
    "dem": 33,
    "terrain": 14,
    "worldcover": 5,
    "glacier": 5,
    "osm": 5,
    "cost": 5,
    "zones": 2,
    "pathfinding": 19,
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


def _in_coverage(lat, lon):
    """Check si un point WGS84 est dans la bbox du PBF Valhalla."""
    w, s, e, n = VALHALLA_COVERAGE_BBOX
    return s <= lat <= n and w <= lon <= e


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
    start = (req.start_lat, req.start_lon)
    end = (req.end_lat, req.end_lon)
    warnings = []

    # -- 0. tentative reseau (Valhalla)
    _progress(progress_callback, "network", 0)
    strategy = "raster"
    valhalla_result = None
    hybrid_info = None
    valhalla_up = False

    try:
        valhalla_up = valhalla_available()
    except Exception:
        pass

    # verif couverture PBF avant d'appeler Valhalla
    if valhalla_up:
        start_ok = _in_coverage(start[0], start[1])
        end_ok = _in_coverage(end[0], end[1])
        if not start_ok or not end_ok:
            logger.warning("hors couverture PBF: start_ok=%s end_ok=%s", start_ok, end_ok)
            warnings.append("Point(s) hors couverture Valhalla, routage raster.")
            valhalla_up = False

    if valhalla_up:
        try:
            vr = valhalla_route(start, end)

            if vr is not None:
                logger.info("valhalla: dist=%.2fkm snap_start=%.0fm snap_end=%.0fm",
                            vr["distance_km"], vr["snap_start_m"], vr["snap_end_m"])

            # validation snap + ghost route
            if vr is not None:
                direct_km = haversine_km(start[0], start[1], end[0], end[1])
                # ghost check: route quasi-nulle pour des points distants
                if vr["distance_km"] < 0.01 and direct_km > GHOST_ROUTE_MIN_DISTANCE_KM:
                    logger.warning("ghost route rejetee: %.3f km pour %.1f km vol d'oiseau",
                                   vr["distance_km"], direct_km)
                    vr = None
                # snap trop loin -> pas de route fiable
                elif vr["snap_end_m"] > SNAP_MAX_DISTANCE_M:
                    logger.warning("snap end trop loin: %.0fm > %.0fm",
                                   vr["snap_end_m"], SNAP_MAX_DISTANCE_M)
                    vr = None
                elif vr["snap_start_m"] > SNAP_MAX_DISTANCE_M:
                    logger.warning("snap start trop loin: %.0fm > %.0fm",
                                   vr["snap_start_m"], SNAP_MAX_DISTANCE_M)
                    vr = None

            # ghost: boucle (premier == dernier point a 10m pres)
            if vr is not None and len(vr["coords"]) >= 2:
                c0, cN = vr["coords"][0], vr["coords"][-1]
                if haversine_km(c0[0], c0[1], cN[0], cN[1]) * 1000 < 10:
                    logger.warning("ghost route: boucle (first~=last)")
                    vr = None

            # ghost: route avec < 3 points
            if vr is not None and len(vr["coords"]) < 3:
                logger.warning("ghost route: %d points seulement", len(vr["coords"]))
                vr = None

            if vr is not None:
                if is_detour_excessive(vr["distance_km"], start, end):
                    # cas C: detour -> raster, mais on garde vr pour tenter des ponts
                    strategy = "raster"
                    valhalla_result = vr
                else:
                    # cas A: route Valhalla OK
                    strategy = "network"
                    valhalla_result = vr
            else:
                # CAS B: NoRoute / ghost / snap loin -> tenter hybride
                hybrid_info = find_network_exit(start, end)
                if hybrid_info is None:
                    hybrid_info = find_network_entry(start, end)
                if hybrid_info is not None:
                    strategy = "hybrid"
                else:
                    strategy = "raster"
        except ValhallaError:
            strategy = "raster"

    _progress(progress_callback, "network", 1.0)
    logger.info("strategy=%s valhalla_up=%s hybrid_info=%s",
                strategy, valhalla_up, "exit" if hybrid_info and "exit_point" in hybrid_info
                else "entry" if hybrid_info else None)

    # -- 0b. tentative graphe GPX
    gpx_result = None
    if strategy != "network":
        _progress(progress_callback, "gpx_graph", 0)
        try:
            gpx_result = route_via_gpx(start, end)
            if gpx_result is not None:
                logger.info("gpx graph: coverage=%s, %.2fkm via %s",
                            gpx_result["coverage"], gpx_result["distance_km"],
                            gpx_result["gpx_sources"])
        except Exception as e:
            logger.warning("gpx graph echec: %s", e)
        _progress(progress_callback, "gpx_graph", 1.0)

    # GPX full coverage -> assembler Valhalla approche + GPX + Valhalla sortie
    if gpx_result is not None and gpx_result["coverage"] == "full" and valhalla_up:
        entry_p = gpx_result["entry_portal"]
        exit_p = gpx_result["exit_portal"]
        try:
            approach_vr = valhalla_route(start, entry_p["osm_coords"])
        except Exception:
            approach_vr = None
        try:
            egress_vr = valhalla_route(exit_p["osm_coords"], end)
        except Exception:
            egress_vr = None

        # valider que l'egress ne contourne pas tout le massif
        if egress_vr is not None and is_detour_excessive(
                egress_vr["distance_km"], exit_p["osm_coords"], end):
            logger.warning("gpx full: egress detour excessif (%.1fkm), drop",
                           egress_vr["distance_km"])
            egress_vr = None

        if approach_vr is not None and is_detour_excessive(
                approach_vr["distance_km"], start, entry_p["osm_coords"]):
            logger.warning("gpx full: approach detour excessif (%.1fkm), drop",
                           approach_vr["distance_km"])
            approach_vr = None

        # sans egress viable: verifier que le GPX arrive assez pres de la dest
        # sinon degrader en partial pour que le raster comble le trou
        if egress_vr is None:
            exit_c = exit_p["gpx_coords"]
            exit_to_end_m = haversine_km(
                exit_c[0], exit_c[1], end[0], end[1]) * 1000
            if exit_to_end_m > SNAP_MAX_DISTANCE_M:
                logger.warning("gpx full -> partial: exit a %.0fm de dest, "
                               "raster prendra le relais", exit_to_end_m)
                gpx_result = {**gpx_result, "coverage": "partial",
                              "gpx_exit_wgs84": exit_c}

        if gpx_result["coverage"] == "full" and (
                approach_vr is not None or egress_vr is not None):
            gpx_feature = assemble_gpx_route(approach_vr, gpx_result, egress_vr)
            _progress(progress_callback, "result", 0)

            saved_route_id = None
            if req.save:
                try:
                    props = gpx_feature["properties"]
                    route_data = {
                        "name": req.name,
                        "start_lat": req.start_lat, "start_lon": req.start_lon,
                        "end_lat": req.end_lat, "end_lon": req.end_lon,
                        "resolution": req.resolution, "month": req.month,
                        "acclimatized": req.acclimatized,
                        "distance_m": props["distance_km"] * 1000,
                        "dplus_m": props["dplus_m"], "dminus_m": props["dminus_m"],
                        "time_tobler_h": props["time_tobler_h"],
                        "glacier_pct": 0, "cost_total": 0,
                        "computation_time_s": 0,
                        "geojson": json.dumps(gpx_feature),
                    }
                    saved_route_id = save_route(DB_PATH, route_data)
                except Exception as e:
                    logger.warning("auto-save gpx_hybrid failed: %s", e)

            _progress(progress_callback, "result", 1.0)
            result = {
                "status": "ok",
                "route": gpx_feature,
                "computation_time_s": 0,
                "strategy": "gpx_hybrid",
                "valhalla_available": True,
                "layers_used": ["gpx_graph", "valhalla"],
                "coverage": "full",
            }
            if warnings:
                result["warnings"] = warnings
            if saved_route_id is not None:
                result["saved_route_id"] = saved_route_id
            return result

    # GPX partial coverage -> Valhalla approche + GPX + raster pour le reste
    # on injecte les points dans le pipeline hybrid existant
    if gpx_result is not None and gpx_result["coverage"] == "partial" and valhalla_up:
        entry_p = gpx_result["entry_portal"]
        gpx_exit = gpx_result.get("gpx_exit_wgs84")
        if entry_p and gpx_exit:
            try:
                approach_vr = valhalla_route(start, entry_p["osm_coords"])
            except Exception:
                approach_vr = None

            # verif detour approach (ex: si entry portal est de l'autre cote du massif)
            if approach_vr is not None and is_detour_excessive(
                    approach_vr["distance_km"], start, entry_p["osm_coords"]):
                logger.warning("gpx partial: approach detour excessif (%.1fkm), skip GPX",
                               approach_vr["distance_km"])
                approach_vr = None

            if approach_vr is not None:
                # remplacer la strategie: on fait hybrid avec le GPX au milieu
                # le raster partira du bout du GPX vers la destination
                strategy = "hybrid"
                hybrid_info = {
                    "exit_point": gpx_exit,
                    "approach": approach_vr,
                    "snap_m": entry_p["snap_m"],
                    "_gpx_result": gpx_result,
                }
                valhalla_result = approach_vr
                logger.info("gpx partial: approach Valhalla OK, raster depuis "
                            "(%.5f,%.5f) -> dest", gpx_exit[0], gpx_exit[1])

    # -- cas network: pas besoin du pipeline raster
    if strategy == "network":
        _progress(progress_callback, "result", 0)
        feature = valhalla_to_geojson_feature(valhalla_result)

        # auto-save si demande
        saved_route_id = None
        if req.save:
            try:
                props = feature["properties"]
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
                    "computation_time_s": 0,
                    "geojson": json.dumps(feature),
                }
                saved_route_id = save_route(DB_PATH, route_data)
            except Exception as e:
                logger.warning("auto-save failed: %s", e)

        _progress(progress_callback, "result", 1.0)
        result = {
            "status": "ok",
            "route": feature,
            "computation_time_s": 0,
            "strategy": "network",
            "valhalla_available": True,
            "layers_used": ["valhalla"],
            "coverage": "full",
            "snap_start_m": round(valhalla_result.get("snap_start_m", 0), 1),
            "snap_end_m": round(valhalla_result.get("snap_end_m", 0), 1),
        }
        if warnings:
            result["warnings"] = warnings
        if saved_route_id is not None:
            result["saved_route_id"] = saved_route_id
        return result

    # -- pipeline raster (strategy = "raster" ou "hybrid" avec raster terminal)

    # -- 1. bbox
    _progress(progress_callback, "bbox")

    raster_start = start
    raster_end = end

    if strategy == "hybrid" and hybrid_info is not None:
        if "exit_point" in hybrid_info:
            raster_start = hybrid_info["exit_point"]
            raster_end = end
        else:
            raster_start = start
            raster_end = hybrid_info["entry_point"]
        bboxes = reduce_bbox(raster_start, raster_end, margin_m=HYBRID_BBOX_MARGIN_M)
    else:
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
    landcover = get_landcover_cost(bbox_wgs84, bbox_l93, dem.shape,
                                    dst_transform=transform)
    _progress(progress_callback, "worldcover", 1.0)

    # -- 5. glacier (optionnel)
    _progress(progress_callback, "glacier", 0)
    glacier_mask = get_glacier_mask(bbox_l93, transform, dem.shape)
    _progress(progress_callback, "glacier", 1.0)

    # -- 5b. OSM trails + barriers (toujours charge)
    _progress(progress_callback, "osm", 0)
    trail_cost = get_trail_cost(bbox_l93, transform, dem.shape, resolution=req.resolution)
    if trail_cost is None:
        logger.warning("ATTENTION: trail_cost=None, aucun sentier OSM ne sera utilise pour le pathfinding")
    _progress(progress_callback, "osm", 0.5)
    barrier_masks = get_barrier_masks(bbox_l93, transform, dem.shape)
    _progress(progress_callback, "osm", 0.8)

    # -- 5c. segments terrain (traces GPX custom)
    segments_loaded = False
    try:
        segments = load_segments_for_bbox(bbox_l93)
        if segments:
            seg_cost = rasterize_segments(segments, transform, dem.shape, resolution=req.resolution)
            trail_cost = merge_trail_layers(trail_cost, seg_cost)
            del seg_cost
            segments_loaded = True
            logger.info("segments terrain: %d segments merged", len(segments))
    except Exception as e:
        logger.warning("segments terrain echec (non bloquant): %s", e)

    _progress(progress_callback, "osm", 1.0)

    _layers = []
    if trail_cost is not None: _layers.append("trails")
    if barrier_masks is not None: _layers.append("barriers")
    if glacier_mask is not None: _layers.append("glacier")
    logger.info("layers loaded: %s", _layers or "aucun")

    # -- 6. cost surface
    _progress(progress_callback, "cost", 0)

    if use_aniso:
        # mode anisotrope: base cost sans Tobler (calcule per-edge)
        base_cost, nodata_mask = build_base_cost(
            dem, slope, aspect, roughness, glacier_mask,
            month=req.month, acclimatized=req.acclimatized,
            landcover_cost=landcover, trail_cost=trail_cost,
        )
        # on garde aussi la cost surface complete pour le calque
        cost_for_display, factors, _ = build_cost_surface(
            dem, slope, aspect, roughness, glacier_mask,
            month=req.month, acclimatized=req.acclimatized,
            landcover_cost=landcover, trail_cost=trail_cost,
        )
        del factors
    else:
        # mode isotrope: surface complete
        cost_for_display, factors, nodata_mask = build_cost_surface(
            dem, slope, aspect, roughness, glacier_mask,
            month=req.month, acclimatized=req.acclimatized,
            landcover_cost=landcover, trail_cost=trail_cost,
        )
        del factors
        base_cost = None

    del slope, aspect, roughness, landcover

    # -- appliquer barrieres OSM
    if barrier_masks is not None:
        bmask = barrier_masks["barrier_mask"]
        smask = barrier_masks["stream_mask"]
        # barrieres -> infranchissable
        cost_for_display[bmask] = NODATA_VALUE
        if base_cost is not None:
            base_cost[bmask] = np.inf
        # ruisseaux -> penalite
        valid_stream = smask & (cost_for_display != NODATA_VALUE)
        cost_for_display[valid_stream] *= STREAM_CROSSING_PENALTY
        if base_cost is not None:
            valid_stream_base = smask & np.isfinite(base_cost)
            base_cost[valid_stream_base] *= STREAM_CROSSING_PENALTY
        logger.info("barriers: %d blocked, %d stream px", bmask.sum(), smask.sum())

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

    # -- 6c. ponts raster (si Valhalla avait un detour)
    bridge_feature = None
    if valhalla_result is not None and strategy == "raster":
        try:
            detours = detect_detour_segments(valhalla_result)
            if detours:
                bridges = []
                for d in detours:
                    br = compute_raster_bridge(
                        d["start"], d["end"],
                        cost_for_display, transform, dem, glacier_mask, req.resolution)
                    bridges.append(br)
                if any(br is not None for br in bridges):
                    hybrid_coords = apply_bridges(valhalla_result, detours, bridges)
                    if hybrid_coords is not None:
                        # construire le feature GeoJSON depuis coords assemblees
                        bridge_feature = valhalla_to_geojson_feature(hybrid_coords)
                        bridge_feature["properties"]["strategy"] = "hybrid_bridge"
                        bridge_feature["properties"]["n_bridges"] = hybrid_coords.get("n_bridges", 0)
                        strategy = "hybrid_bridge"
        except Exception as e:
            logger.warning("ponts raster echec (fallback raster): %s", e)

    # -- 7. pathfinding
    _progress(progress_callback, "pathfinding", 0)

    # si ponts raster ont reussi, pas besoin du pathfinding complet
    if bridge_feature is not None:
        _progress(progress_callback, "pathfinding", 1.0)
        _progress(progress_callback, "result", 0)

        layers_used = ["dem", "terrain", "valhalla"]
        if trail_cost is not None:
            layers_used.append("osm_trails")
        if segments_loaded:
            layers_used.append("segments")
        if glacier_mask is not None:
            layers_used.append("glacier")

        saved_route_id = None
        if req.save:
            try:
                props = bridge_feature["properties"]
                route_data = {
                    "name": req.name,
                    "start_lat": req.start_lat, "start_lon": req.start_lon,
                    "end_lat": req.end_lat, "end_lon": req.end_lon,
                    "resolution": req.resolution, "month": req.month,
                    "acclimatized": req.acclimatized,
                    "distance_m": props["distance_km"] * 1000,
                    "dplus_m": props["dplus_m"], "dminus_m": props["dminus_m"],
                    "time_tobler_h": props["time_tobler_h"],
                    "glacier_pct": props["glacier_pct"],
                    "cost_total": props.get("cost_total"),
                    "computation_time_s": 0,
                    "geojson": json.dumps(bridge_feature),
                }
                saved_route_id = save_route(DB_PATH, route_data)
            except Exception as e:
                logger.warning("auto-save bridge failed: %s", e)

        _progress(progress_callback, "result", 1.0)
        result = {
            "status": "ok",
            "route": bridge_feature,
            "computation_time_s": 0,
            "strategy": "hybrid_bridge",
            "valhalla_available": True,
            "layers_used": layers_used,
        }
        if saved_route_id is not None:
            result["saved_route_id"] = saved_route_id
        return result

    start_row, start_col, _, _ = wgs84_to_pixel(
        raster_start[0], raster_start[1], transform, cost_for_display.shape)
    end_row, end_col, _, _ = wgs84_to_pixel(
        raster_end[0], raster_end[1], transform, cost_for_display.shape)

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
                cost_grid, start_rc, end_rc, n_alt=req.n_alternatives,
                resolution=req.resolution)
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

    # assemblage hybrid si CAS B
    if strategy == "hybrid" and hybrid_info is not None and routes:
        raster_feature = routes[0]
        gpx_partial = hybrid_info.get("_gpx_result")

        if gpx_partial is not None:
            # GPX partial: Valhalla approche + GPX milieu + raster fin
            assembled = assemble_gpx_route(
                hybrid_info["approach"], gpx_partial, None)
            # concatener le raster apres le GPX
            gpx_geojson_coords = assembled["geometry"]["coordinates"]
            raster_coords = raster_feature["geometry"]["coordinates"]
            merged = gpx_geojson_coords + raster_coords
            assembled["geometry"]["coordinates"] = merged
            r_props = raster_feature["properties"]
            a_props = assembled["properties"]
            a_props["distance_km"] = round(
                a_props["distance_km"] + r_props.get("distance_km", 0), 2)
            a_props["dplus_m"] = round(
                a_props["dplus_m"] + r_props.get("dplus_m", 0))
            a_props["dminus_m"] = round(
                a_props["dminus_m"] + r_props.get("dminus_m", 0))
            a_props["time_tobler_h"] = round(
                a_props["time_tobler_h"] + r_props.get("time_tobler_h", 0), 1)
            a_props["glacier_pct"] = r_props.get("glacier_pct", 0)
            a_props["n_points"] = len(merged)
            a_props["strategy"] = "gpx_hybrid"
            routes[0] = assembled
            logger.info("gpx partial assembled: Valhalla + GPX + raster")
        elif "exit_point" in hybrid_info:
            assembled = assemble_route(
                hybrid_info["approach"]["coords"],
                raster_feature,
                hybrid_info["exit_point"],
                valhalla_stats=hybrid_info["approach"],
                order="valhalla_first")
            routes[0] = assembled
        else:
            assembled = assemble_route(
                hybrid_info["continuation"]["coords"],
                raster_feature,
                hybrid_info["entry_point"],
                valhalla_stats=hybrid_info["continuation"],
                order="raster_first")
            routes[0] = assembled

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

    # warning isotrope
    if not use_aniso and routes:
        dplus = routes[0]["properties"]["dplus_m"]
        dminus = routes[0]["properties"]["dminus_m"]
        if max(dplus, dminus) > ISOTROPIC_WARNING_DPLUS_M:
            warnings.append(
                f"Mode isotrope avec D+={dplus:.0f}m / D-={dminus:.0f}m : "
                "le mode precis (anisotrope) donnerait un meilleur resultat "
                "car il distingue montee et descente."
            )

    layers_used = ["dem", "terrain"]
    if trail_cost is not None:
        layers_used.append("osm_trails")
    if barrier_masks is not None:
        layers_used.append("osm_barriers")
    if segments_loaded:
        layers_used.append("segments")
    if glacier_mask is not None:
        layers_used.append("glacier")
    if strategy == "hybrid":
        layers_used.append("valhalla")
    if gpx_result is not None:
        layers_used.append("gpx_graph")

    # coverage status
    if not valhalla_up:
        coverage = "none"
    elif strategy in ("hybrid", "network"):
        coverage = "full"
    else:
        coverage = "partial"

    # si GPX partial a ete utilise, overrider la strategy
    effective_strategy = strategy
    if (gpx_result is not None and gpx_result["coverage"] == "partial"
            and hybrid_info is not None and "_gpx_result" in hybrid_info):
        effective_strategy = "gpx_hybrid"

    result = {
        "status": "ok",
        "route": routes[0] if routes else None,
        "computation_time_s": round(total_dt, 1),
        "strategy": effective_strategy,
        "valhalla_available": valhalla_up,
        "layers_used": layers_used,
        "coverage": coverage,
    }

    # snap si Valhalla a ete consulte
    if valhalla_result is not None:
        result["snap_start_m"] = round(valhalla_result.get("snap_start_m", 0), 1)
        result["snap_end_m"] = round(valhalla_result.get("snap_end_m", 0), 1)

    if warnings:
        result["warnings"] = warnings

    if len(routes) > 1:
        result["routes"] = routes
        result["n_routes"] = len(routes)

    if saved_route_id is not None:
        result["saved_route_id"] = saved_route_id

    return result
