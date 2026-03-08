# hybrid.py -- assemblage troncons reseau + raster
# pose la structure pour le vrai hybrid (Phase 7)

import logging
import numpy as np

from alpineroute.config import MAX_ROUTE_POINTS_API
from alpineroute.utils import wgs84_to_l93, l93_to_wgs84
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
