# worldCover dans la surface de cout

import logging
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.windows import from_bounds

from alpineroute.config import (
    CRS_L93, CRS_WGS84, WORLDCOVER_MULTIPLIERS, WORLDCOVER_URL_PATTERN,
)
from alpineroute.utils import DownloadError

logger = logging.getLogger(__name__)


def get_worldcover_tile_name(lat, lon):
    """Determine le nom de la tuile WorldCover pour des coords."""
    lat_floor = int(np.floor(lat))
    lon_floor = int(np.floor(lon))
    lat_str = f"N{lat_floor:02d}" if lat_floor >= 0 else f"S{abs(lat_floor):02d}"
    lon_str = f"E{lon_floor:03d}" if lon_floor >= 0 else f"W{abs(lon_floor):03d}"
    return f"{lat_str}{lon_str}"


def load_worldcover_windowed(bbox_wgs84):
    """Lecture fenetree d'une tuile WorldCover via /vsicurl/."""
    # determine la tuile
    lat_center = (bbox_wgs84["lat_min"] + bbox_wgs84["lat_max"]) / 2
    lon_center = (bbox_wgs84["lon_min"] + bbox_wgs84["lon_max"]) / 2
    tile_name = get_worldcover_tile_name(lat_center, lon_center)
    url = WORLDCOVER_URL_PATTERN.format(tile=tile_name)

    vsicurl = f"/vsicurl/{url}"
    logger.info("WorldCover tile %s via vsicurl", tile_name)

    try:
        with rasterio.open(vsicurl) as ds:
            window = from_bounds(
                bbox_wgs84["lon_min"], bbox_wgs84["lat_min"],
                bbox_wgs84["lon_max"], bbox_wgs84["lat_max"],
                ds.transform,
            )
            data = ds.read(1, window=window)
            win_transform = ds.window_transform(window)
            return data, win_transform, ds.crs
    except Exception as e:
        raise DownloadError(f"WorldCover: {e}")


def reproject_worldcover_l93(data, src_transform, src_crs, bbox_l93, bbox_wgs84):
    """Reprojette WorldCover en L93 a 10m (nearest pour categorique)."""
    dst_res = 10
    width = int((bbox_l93["xmax"] - bbox_l93["xmin"]) / dst_res)
    height = int((bbox_l93["ymax"] - bbox_l93["ymin"]) / dst_res)

    dst_transform, _, _ = calculate_default_transform(
        src_crs, CRS_L93,
        data.shape[1], data.shape[0],
        left=bbox_wgs84["lon_min"], bottom=bbox_wgs84["lat_min"],
        right=bbox_wgs84["lon_max"], top=bbox_wgs84["lat_max"],
        dst_width=width, dst_height=height,
    )

    dst = np.zeros((height, width), dtype=np.uint8)
    reproject(
        source=data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=CRS_L93,
        resampling=Resampling.nearest,
    )

    logger.info("WorldCover L93: %dx%d @ %dm", width, height, dst_res)
    return dst, dst_transform


def build_landcover_cost(lc_data):
    """Construit le multiplicateur de cout depuis les classes WorldCover."""
    cost = np.ones_like(lc_data, dtype=np.float32)
    for code, mult in WORLDCOVER_MULTIPLIERS.items():
        cost[lc_data == code] = mult
    return cost


# -- fonctions ajoutees pour le pipeline integre --

def resample_to_grid(lc_cost, target_shape, src_transform=None,
                     dst_transform=None):
    """Resample la grille landcover vers la grille DEM avec alignement spatial.
    Utilise rasterio.warp.reproject pour un calage geospatial exact."""
    if lc_cost.shape == target_shape:
        return lc_cost

    dst = np.ones(target_shape, dtype=lc_cost.dtype)
    reproject(
        source=lc_cost,
        destination=dst,
        src_transform=src_transform,
        src_crs=CRS_L93,
        dst_transform=dst_transform,
        dst_crs=CRS_L93,
        resampling=Resampling.nearest,
    )
    return dst


def get_landcover_cost(bbox_wgs84, bbox_l93, target_shape, dst_transform=None):
    """Pipeline complet WorldCover: load -> reproject -> cost -> resample.
    dst_transform: transform de la grille DEM cible (pour alignement spatial).
    Retourne None si echec (non bloquant)."""
    try:
        data, src_transform, src_crs = load_worldcover_windowed(bbox_wgs84)
        lc_l93, lc_transform = reproject_worldcover_l93(
            data, src_transform, src_crs, bbox_l93, bbox_wgs84)
        lc_cost = build_landcover_cost(lc_l93)
        resampled = resample_to_grid(lc_cost, target_shape,
                                     src_transform=lc_transform,
                                     dst_transform=dst_transform)
        logger.info("landcover cost ready: %s", resampled.shape)
        return resampled

    except Exception as e:
        logger.warning("landcover echec (non bloquant): %s", e)
        return None
