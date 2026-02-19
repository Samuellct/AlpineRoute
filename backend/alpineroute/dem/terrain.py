# analyse terrain : pente, aspect, rugosite

import numpy as np
from scipy.ndimage import convolve
import logging

from alpineroute.config import NODATA_VALUE
from alpineroute.utils import make_nodata_mask

logger = logging.getLogger(__name__)


# -- pente/aspect (Horn) --

def compute_slope_aspect(dem, resolution):
    """Calcule pente (deg) et aspect (deg, 0=N clockwise) via Horn kernels."""
    res = resolution

    work = dem.astype(np.float64)
    nodata_base = (dem == NODATA_VALUE) | np.isnan(dem)
    work[nodata_base] = np.nan

    padded = np.pad(work, 1, mode='reflect')

    # Horn kernels
    gx_kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float64) / (8.0 * res)

    gy_kernel = np.array([[-1, -2, -1],
                          [ 0,  0,  0],
                          [ 1,  2,  1]], dtype=np.float64) / (8.0 * res)

    # nan -> 0 pour la convolution, on masque apres
    padded_clean = np.where(np.isnan(padded), 0.0, padded)

    gx_full = convolve(padded_clean, gx_kernel, mode='constant', cval=0.0)
    gy_full = convolve(padded_clean, gy_kernel, mode='constant', cval=0.0)

    gx = gx_full[1:-1, 1:-1]
    gy = gy_full[1:-1, 1:-1]

    # pente en degres
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # aspect : 0=Nord, 90=Est, 180=Sud, 270=Ouest
    aspect_rad = np.arctan2(-gy, gx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = 90.0 - aspect_deg
    aspect_deg = np.mod(aspect_deg, 360.0).astype(np.float32)

    # terrain plat -> aspect indefini
    flat_mask = (gx == 0) & (gy == 0)
    aspect_deg[flat_mask] = NODATA_VALUE

    # masque nodata dilate
    mask = make_nodata_mask(dem, dilate=True)
    slope_deg[mask] = NODATA_VALUE
    aspect_deg[mask] = NODATA_VALUE

    logger.info("slope/aspect: %d px valides, %d plats",
                np.sum(~mask), np.sum(flat_mask & ~mask))
    return slope_deg, aspect_deg


# -- rugosite TRI --

def compute_roughness(dem):
    """TRI via ecart-type 3x3, NaN-aware."""
    nodata_base = (dem == NODATA_VALUE) | np.isnan(dem)
    valid_px = (~nodata_base).astype(np.float64)
    work = np.where(nodata_base, 0.0, dem.astype(np.float64))

    kernel = np.ones((3, 3), dtype=np.float64)

    count = convolve(valid_px, kernel, mode='reflect')
    count = np.maximum(count, 1.0)

    sum_z = convolve(work, kernel, mode='reflect')
    sum_z2 = convolve(work ** 2, kernel, mode='reflect')

    mean_z = sum_z / count
    variance = np.maximum(sum_z2 / count - mean_z ** 2, 0.0)
    roughness = np.sqrt(variance).astype(np.float32)

    mask = make_nodata_mask(dem, dilate=True)
    roughness[mask] = NODATA_VALUE

    logger.info("roughness TRI 3x3 done")
    return roughness
