# export GPX + GeoJSON WGS84

import os
import json
import logging
import numpy as np
import gpxpy
import gpxpy.gpx
from shapely.geometry import LineString

from alpineroute.config import (
    NODATA_VALUE, CRS_L93, CRS_WGS84,
    SIMPLIFY_TOLERANCE_M, DEM_RESOLUTION,
)
from alpineroute.utils import (
    reproject_to_wgs84, load_dem, compute_distance_2d,
)

logger = logging.getLogger(__name__)


# ====================================
#  Elevation sampling avec context manager
# =========================================

def sample_elevations(coords_l93, dem_path):
    """Echantillonne l'altitude du MNT pour chaque point L93."""
    data, _, transform = load_dem(dem_path)

    elevations = np.zeros(len(coords_l93), dtype=np.float64)
    for i, (x, y) in enumerate(coords_l93):
        col, row = ~transform * (x, y)
        row, col = int(round(row)), int(round(col))
        row = max(0, min(row, data.shape[0] - 1))
        col = max(0, min(col, data.shape[1] - 1))
        val = data[row, col]
        elevations[i] = val if val != NODATA_VALUE else 0.0

    return elevations


# ====================================
#  Simplification Douglas-Peucker
# ====================================

def simplify_route(coords_l93, tolerance_m=None):
    """Simplification en L93 (metres)."""
    if tolerance_m is None:
        tolerance_m = SIMPLIFY_TOLERANCE_M

    line = LineString(coords_l93)
    simplified = line.simplify(tolerance_m, preserve_topology=True)
    simp_coords = np.array(simplified.coords)
    logger.info("simplify: %d -> %d pts (tol=%.0fm)",
                len(coords_l93), len(simp_coords), tolerance_m)
    return simp_coords


# ============
#  Export GPX
# ============

def export_gpx(coords_wgs84, elevations, stats, output_path, route_name="Route"):
    """Ecrit un fichier GPX 3D."""
    gpx = gpxpy.gpx.GPX()
    gpx.name = route_name
    dist_km = stats.get("dist_2d_m", 0) / 1000
    gpx.description = (
        f"Route {dist_km:.1f}km D+{stats.get('dplus', 0):.0f}m "
        f"~{stats.get('time_tobler_h', 0):.1f}h"
    )
    gpx.creator = "AlpineRoute Optimizer"

    track = gpxpy.gpx.GPXTrack()
    track.name = route_name
    gpx.tracks.append(track)

    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)

    for (lon, lat), elev in zip(coords_wgs84, elevations):
        pt = gpxpy.gpx.GPXTrackPoint(
            latitude=float(lat), longitude=float(lon), elevation=float(elev),
        )
        segment.points.append(pt)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(gpx.to_xml())

    logger.info("GPX: %s (%d pts)", output_path, len(coords_wgs84))
    return output_path


# ====================================
#  Export GeoJSON WGS84
# ====================================

def export_geojson(coords_wgs84, elevations, stats, output_path,
                   route_name="Route", extra_props=None):
    """Ecrit un GeoJSON 3D (lon, lat, elev)."""
    coords_3d = [
        [round(float(lon), 7), round(float(lat), 7), round(float(elev), 1)]
        for (lon, lat), elev in zip(coords_wgs84, elevations)
    ]

    properties = {
        "name": route_name,
        "distance_km": round(stats.get("dist_2d_m", 0) / 1000, 2),
        "dplus_m": round(stats.get("dplus", 0)),
        "dminus_m": round(stats.get("dminus", 0)),
        "time_tobler_h": round(stats.get("time_tobler_h", 0), 1),
        "glacier_pct": round(stats.get("glacier_pct", 0), 1),
        "n_points": len(coords_3d),
        "crs": CRS_WGS84,
    }
    if extra_props:
        properties.update(extra_props)

    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords_3d},
            "properties": properties,
        }],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

    logger.info("GeoJSON: %s (%d pts)", output_path, len(coords_3d))
    return output_path
