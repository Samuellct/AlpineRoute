# utils -- fonctions partagees entre tous les modules

import os
import logging
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.ndimage import binary_dilation
from matplotlib.colors import LightSource

from alpineroute.config import (
    CRS_L93, CRS_WGS84, NODATA_VALUE,
    HILLSHADE_AZIMUTH, HILLSHADE_ALTITUDE, HILLSHADE_VERT_EXAG,
    BBOX_MARGIN_M, BBOX_MAX_SIZE_M, BBOX_ALIGN_M,
)


# --- exceptions ---

class DataNotFoundError(FileNotFoundError):
    pass

class PointOutOfBoundsError(ValueError):
    pass

class DownloadError(RuntimeError):
    pass

class ValhallaError(RuntimeError):
    pass


# --- logging ---

def setup_logging(name=None, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(name)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


#--- raster I/O ---

def load_dem(path):
    # charge le tif, retourne (array, profile, transform)
    if not os.path.exists(path):
        raise DataNotFoundError(f"MNT introuvable: {path}")

    with rasterio.open(path) as ds:
        data = ds.read(1).astype(np.float32)
        profile = ds.profile.copy()
        transform = ds.transform
    return data, profile, transform


def load_raster(path, dtype=np.float32):
    if not os.path.exists(path):
        raise DataNotFoundError(f"Raster introuvable: {path}")

    with rasterio.open(path) as ds:
        data = ds.read(1).astype(dtype)
        profile = ds.profile.copy()
    return data, profile


def save_raster(data, path, profile, dtype="float32", nodata=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    out_profile = profile.copy()
    out_profile.update({
        "driver": "GTiff",
        "dtype": dtype,
        "count": 1,
        "nodata": nodata if nodata is not None else NODATA_VALUE,
        "compress": "deflate",
        "tiled": True,
    })
    # predictor=2 ok pour float, pas pour uint8
    if dtype in ("float32", "float64"):
        out_profile["predictor"] = 2
    else:
        out_profile.pop("predictor", None)

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data.astype(dtype), 1)

    return path


def make_nodata_mask(dem, dilate=True):
    # masque nodata + dilate 1px (bords)
    mask = (dem == NODATA_VALUE) | np.isnan(dem)
    if dilate:
        struct = np.ones((3, 3), dtype=bool)
        mask = binary_dilation(mask, structure=struct)
    return mask


# ---- coordonnees ----

# cache des transformers pour eviter de les recreer a chaque appel
_transformer_cache = {}

def _cached_transformer(src_crs, dst_crs):
    key = (src_crs, dst_crs)
    if key not in _transformer_cache:
        _transformer_cache[key] = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return _transformer_cache[key]


def wgs84_to_l93(lon, lat):
    proj = _cached_transformer(CRS_WGS84, CRS_L93)
    return proj.transform(lon, lat)


def l93_to_wgs84(x, y):
    proj = _cached_transformer(CRS_L93, CRS_WGS84)
    return proj.transform(x, y)


def wgs84_to_pixel(lat, lon, transform, shape):
    """WGS84 lat/lon -> (row, col) pixel sur grille L93.
    Raise PointOutOfBoundsError si hors grille."""
    x_l93, y_l93 = wgs84_to_l93(lon, lat)

    col, row = ~transform * (x_l93, y_l93)
    row, col = int(round(row)), int(round(col))

    h, w = shape
    if not (0 <= row < h and 0 <= col < w):
        raise PointOutOfBoundsError(
            f"Point ({lat}, {lon}) -> pixel ({row}, {col}) hors grille {h}x{w}")
    return row, col, x_l93, y_l93


def pixel_to_l93(rows, cols, transform):
    coords = np.array([transform * (c, r) for r, c in zip(rows, cols)])
    return coords[:, 0], coords[:, 1]


def reproject_to_wgs84(coords_l93):
    """Coords L93 [x,y] -> WGS84 [lon,lat]."""
    proj = _cached_transformer(CRS_L93, CRS_WGS84)
    xs = coords_l93[:, 0]
    ys = coords_l93[:, 1]
    lons, lats = proj.transform(xs, ys)
    return np.column_stack([lons, lats])


def compute_bbox(start_wgs84, end_wgs84, margin_m=None, max_size_m=None):
    """Calcule la bbox englobant deux points WGS84 avec marge.
    start_wgs84, end_wgs84 : tuples (lat, lon)
    Retourne dict avec bbox_l93 et bbox_wgs84."""
    if margin_m is None:
        margin_m = BBOX_MARGIN_M
    if max_size_m is None:
        max_size_m = BBOX_MAX_SIZE_M

    lat1, lon1 = start_wgs84
    lat2, lon2 = end_wgs84

    # conversion en L93
    x1, y1 = wgs84_to_l93(lon1, lat1)
    x2, y2 = wgs84_to_l93(lon2, lat2)

    # rectangle englobant + marge
    xmin = min(x1, x2) - margin_m
    xmax = max(x1, x2) + margin_m
    ymin = min(y1, y2) - margin_m
    ymax = max(y1, y2) + margin_m

    # arrondi au km pour aligner avec les dalles IGN
    align = BBOX_ALIGN_M
    xmin = int(xmin // align) * align
    xmax = (int(xmax // align) + 1) * align
    ymin = int(ymin // align) * align
    ymax = (int(ymax // align) + 1) * align

    # contrainte taille max
    dx = xmax - xmin
    dy = ymax - ymin
    if dx > max_size_m or dy > max_size_m:
        # centre + clip
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        half = max_size_m / 2
        xmin = int((cx - half) // align) * align
        xmax = int((cx + half) // align + 1) * align
        ymin = int((cy - half) // align) * align
        ymax = int((cy + half) // align + 1) * align

    bbox_l93 = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}

    # bbox WGS84 correspondante (coins)
    lon_min, lat_min = l93_to_wgs84(xmin, ymin)
    lon_max, lat_max = l93_to_wgs84(xmax, ymax)
    bbox_wgs84 = {
        "lon_min": lon_min, "lat_min": lat_min,
        "lon_max": lon_max, "lat_max": lat_max,
    }

    return {"bbox_l93": bbox_l93, "bbox_wgs84": bbox_wgs84}


# hillshade pour le fond de carte
def make_hillshade(dem, resolution):
    dem_display = np.where(dem == NODATA_VALUE, np.nan, dem)
    dem_filled = np.where(np.isnan(dem_display), 0, dem_display)
    ls = LightSource(azdeg=HILLSHADE_AZIMUTH, altdeg=HILLSHADE_ALTITUDE)
    return ls.hillshade(dem_filled, vert_exag=HILLSHADE_VERT_EXAG,
                        dx=resolution, dy=resolution)



def compute_distance_2d(coords):
    dx = np.diff(coords[:, 0])
    dy = np.diff(coords[:, 1])
    return float(np.sum(np.sqrt(dx**2 + dy**2)))


def compute_path_stats(path_coords, dem, transform, glacier_mask=None,
                       tobler_speed=None, tobler_gradient=None):
    """Stats completes d'un trajet pixel.
    Retourne (stats_dict, arrays_dict) avec cum_dist, elevations etc."""
    from alpineroute.config import (
        TOBLER_BASE_SPEED_KMH, TOBLER_OPTIMAL_GRADIENT,
    )
    if tobler_speed is None:
        tobler_speed = TOBLER_BASE_SPEED_KMH
    if tobler_gradient is None:
        tobler_gradient = TOBLER_OPTIMAL_GRADIENT

    rows = path_coords[:, 0]
    cols = path_coords[:, 1]

    elevations = dem[rows, cols]
    x_l93, y_l93 = pixel_to_l93(rows, cols, transform)

    dx = np.diff(x_l93)
    dy = np.diff(y_l93)
    dz = np.diff(elevations)

    seg_dist_2d = np.sqrt(dx**2 + dy**2)
    seg_dist_3d = np.sqrt(dx**2 + dy**2 + dz**2)

    dist_2d = float(np.sum(seg_dist_2d))
    dist_3d = float(np.sum(seg_dist_3d))
    cum_dist = np.concatenate([[0], np.cumsum(seg_dist_2d)])

    dplus = float(np.sum(dz[dz > 0]))
    dminus = float(abs(np.sum(dz[dz < 0])))

    # pentes locales
    slopes = np.degrees(np.arctan2(np.abs(dz), seg_dist_2d))
    slopes = np.where(np.isnan(slopes), 0, slopes)

    # temps Tobler
    gradient = np.where(seg_dist_2d > 0, dz / seg_dist_2d, 0)
    v = tobler_speed * np.exp(-3.5 * np.abs(gradient + tobler_gradient))
    v = np.maximum(v, 0.01)
    seg_time_h = (seg_dist_2d / 1000.0) / v
    total_time_h = float(np.sum(seg_time_h))

    # % glacier
    glacier_pct = 0.0
    if glacier_mask is not None:
        on_glacier = glacier_mask[rows, cols]
        glacier_pct = float(on_glacier.sum() / len(rows) * 100)

    stats = {
        "n_pixels": len(path_coords),
        "dist_2d_m": dist_2d,
        "dist_3d_m": dist_3d,
        "dplus": dplus,
        "dminus": dminus,
        "elev_start": float(elevations[0]),
        "elev_end": float(elevations[-1]),
        "elev_min": float(elevations.min()),
        "elev_max": float(elevations.max()),
        "glacier_pct": glacier_pct,
        "time_tobler_h": total_time_h,
    }

    arrays = {
        "cum_dist": cum_dist,
        "elevations": elevations,
        "slopes": slopes,
        "x_l93": x_l93,
        "y_l93": y_l93,
    }

    return stats, arrays


def save_figure(fig, path, dpi=150, uhd_dpi=None):
    if uhd_dpi is None:
        from alpineroute.config import UHD_DPI
        uhd_dpi = UHD_DPI

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')

    # version UHD dans un sous-dossier
    parent = os.path.dirname(path)
    fname = os.path.basename(path)
    uhd_dir = os.path.join(parent, "uhd")
    os.makedirs(uhd_dir, exist_ok=True)
    fig.savefig(os.path.join(uhd_dir, fname), dpi=uhd_dpi,
                bbox_inches='tight', facecolor='white')
