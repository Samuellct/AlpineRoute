# parsing GPX, stats, conversion GeoJSON pour les traces alpine

import os
import json
import math
import logging

import gpxpy

logger = logging.getLogger(__name__)

# cotations alpinisme -> ordinal pour tri
GRADE_ORDINAL = {
    "F": 1, "F+": 2,
    "PD-": 3, "PD": 4, "PD+": 5,
    "AD-": 6, "AD": 7, "AD+": 8,
    "D-": 9, "D": 10, "D+": 11,
    "TD-": 12, "TD": 13, "TD+": 14,
    "ED-": 15, "ED": 16, "ED+": 17,
    "ABO-": 18, "ABO": 19, "ABO+": 20,
}


def load_gpx(gpx_path):
    """Parse un fichier GPX, retourne [(lat, lon, alt), ...] ou None."""
    if not os.path.isfile(gpx_path):
        logger.warning("gpx manquant: %s", gpx_path)
        return None
    try:
        with open(gpx_path, "r", encoding="utf-8") as f:
            gpx = gpxpy.parse(f)
        points = []
        for track in gpx.tracks:
            for seg in track.segments:
                for pt in seg.points:
                    alt = pt.elevation if pt.elevation is not None else 0.0
                    points.append((pt.latitude, pt.longitude, alt))
        if not points:
            logger.warning("gpx vide: %s", gpx_path)
            return None
        return points
    except Exception as e:
        logger.warning("erreur parsing gpx %s: %s", gpx_path, e)
        return None


def _haversine(lat1, lon1, lat2, lon2):
    """Distance haversine en metres."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_stats(points):
    """Retourne (distance_m, dplus_m) depuis une liste de (lat, lon, alt)."""
    dist = 0.0
    dplus = 0.0
    for i in range(1, len(points)):
        lat1, lon1, alt1 = points[i - 1]
        lat2, lon2, alt2 = points[i]
        dist += _haversine(lat1, lon1, lat2, lon2)
        dz = alt2 - alt1
        if dz > 0:
            dplus += dz
    return dist, dplus


def route_to_geojson(entry, points):
    """Convertit une entry index + points en GeoJSON Feature."""
    coords = [[lon, lat, alt] for lat, lon, alt in points]
    dist_m, dplus_m = compute_stats(points)
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coords,
        },
        "properties": {
            "massif": entry.get("massif"),
            "summit": entry.get("summit"),
            "voie": entry.get("voie"),
            "grade": entry.get("grade"),
            "distance_m": round(dist_m, 1),
            "dplus_m": round(dplus_m, 1),
        },
    }
