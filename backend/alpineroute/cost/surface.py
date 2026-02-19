# surface de cout multi-criteres

import numpy as np
import logging

from alpineroute.config import (
    NODATA_VALUE,
    TOBLER_BASE_SPEED_KMH, TOBLER_OPTIMAL_GRADIENT, OFF_TRAIL_FACTOR,
    STEEP_SLOPE_THRESHOLD_DEG, STEEP_SLOPE_MULTIPLIER,
    CRITICAL_SLOPE_DEG, CRITICAL_SLOPE_MULTIPLIER,
    HYPOXIA_ALTITUDE_THRESHOLD, HYPOXIA_RATE_ACCLIMATIZED,
    HYPOXIA_RATE_NOT_ACCLIMATIZED, HYPOXIA_MIN_CAPACITY,
    ASPECT_SUMMER_MONTHS, ASPECT_SOUTH_PENALTY_MAX,
    ASPECT_SOUTH_SLOPE_THRESHOLD, ASPECT_SOUTH_ALTITUDE_THRESHOLD,
    ASPECT_NORTH_PENALTY_MAX, ASPECT_NORTH_SLOPE_THRESHOLD,
    GLACIER_COST_FLAT, GLACIER_COST_MODERATE,
    GLACIER_COST_STEEP, GLACIER_COST_VERY_STEEP,
    ROUGHNESS_CLAMP, ROUGHNESS_SCALE,
)

logger = logging.getLogger(__name__)


# -- facteur pente (Tobler)

def compute_slope_cost(slope_deg):
    """Tobler hiking function hors-sentier."""
    slope_rad = np.radians(np.clip(slope_deg, 0, 89.9))
    gradient = np.tan(slope_rad)

    v = TOBLER_BASE_SPEED_KMH * np.exp(
        -3.5 * np.abs(gradient + TOBLER_OPTIMAL_GRADIENT)
    ) * OFF_TRAIL_FACTOR
    v = np.maximum(v, 0.01)

    # normalise: cout=1.0 sur terrain plat
    v_flat = TOBLER_BASE_SPEED_KMH * np.exp(
        -3.5 * TOBLER_OPTIMAL_GRADIENT
    ) * OFF_TRAIL_FACTOR
    cost = v_flat / v

    steep = slope_deg > STEEP_SLOPE_THRESHOLD_DEG
    cost = np.where(steep, cost * STEEP_SLOPE_MULTIPLIER, cost)

    very_steep = slope_deg > CRITICAL_SLOPE_DEG
    cost = np.where(very_steep, cost * CRITICAL_SLOPE_MULTIPLIER, cost)

    return cost.astype(np.float32)


# -- facteur altitude / hypoxie

def compute_altitude_cost(elevation, acclimatized=True):
    """Penalite hypoxie au-dessus de 1500m."""
    rate = HYPOXIA_RATE_ACCLIMATIZED if acclimatized else HYPOXIA_RATE_NOT_ACCLIMATIZED
    reduction = np.maximum(0, (elevation - HYPOXIA_ALTITUDE_THRESHOLD) * rate / 1000.0)
    capacity = np.maximum(1.0 - reduction, HYPOXIA_MIN_CAPACITY)
    return (1.0 / capacity).astype(np.float32)


# -- facteur aspect / saison

def compute_aspect_cost(aspect_deg, slope_deg, elevation, month=7):
    """Penalite orientation/saison. Faces sud en ete, nord en hiver."""
    cost = np.ones_like(slope_deg, dtype=np.float32)
    valid = (aspect_deg != NODATA_VALUE) & ~np.isnan(aspect_deg)

    if month in ASPECT_SUMMER_MONTHS:
        south_exp = np.cos(np.radians(aspect_deg - 180.0))
        penalty = 1.0 + ASPECT_SOUTH_PENALTY_MAX * np.maximum(south_exp, 0)
        mask = (valid
                & (slope_deg > ASPECT_SOUTH_SLOPE_THRESHOLD)
                & (elevation > ASPECT_SOUTH_ALTITUDE_THRESHOLD))
        cost = np.where(mask, penalty, cost)
    else:
        north_exp = np.cos(np.radians(aspect_deg))
        penalty = 1.0 + ASPECT_NORTH_PENALTY_MAX * np.maximum(north_exp, 0)
        mask = valid & (slope_deg > ASPECT_NORTH_SLOPE_THRESHOLD)
        cost = np.where(mask, penalty, cost)

    return cost


# -- facteur glacier

def compute_glacier_cost(glacier_mask, slope_deg):
    """Surcout glacier en 4 niveaux selon la pente."""
    if glacier_mask is None:
        return np.ones_like(slope_deg, dtype=np.float32)

    cost = np.ones_like(slope_deg, dtype=np.float32)
    cost = np.where(glacier_mask & (slope_deg < 10), GLACIER_COST_FLAT, cost)
    cost = np.where(glacier_mask & (slope_deg >= 10) & (slope_deg < 20),
                    GLACIER_COST_MODERATE, cost)
    cost = np.where(glacier_mask & (slope_deg >= 20) & (slope_deg < 30),
                    GLACIER_COST_STEEP, cost)
    cost = np.where(glacier_mask & (slope_deg >= 30), GLACIER_COST_VERY_STEEP, cost)
    return cost


# -- facteur rugosite

def compute_roughness_cost(roughness):
    """Cout rugosite: 1 + scale*TRI, clamp a 5m."""
    r = np.minimum(roughness, ROUGHNESS_CLAMP)
    return (1.0 + ROUGHNESS_SCALE * r).astype(np.float32)


# =====================================================
#  Assemblage
# =====================================================

def build_cost_surface(dem, slope, aspect, roughness, glacier_mask,
                       month=7, acclimatized=True, landcover_cost=None):
    """Construit la surface de cout multi-criteres. Retourne (cost, factors, nodata_mask)."""
    nodata_mask = (slope == NODATA_VALUE) | np.isnan(slope)

    slope_clean = np.where(nodata_mask, 0, slope)
    aspect_clean = np.where(nodata_mask, 0, aspect)
    rough_clean = np.where(nodata_mask, 0, roughness)
    dem_clean = np.where(nodata_mask, 0, dem)

    logger.info("f_slope (Tobler hors-sentier)")
    f_slope = compute_slope_cost(slope_clean)

    logger.info("f_altitude (hypoxie)")
    f_alt = compute_altitude_cost(dem_clean, acclimatized)

    logger.info("f_aspect (orientation/saison, month=%d)", month)
    f_aspect = compute_aspect_cost(aspect_clean, slope_clean, dem_clean, month)

    logger.info("f_glacier")
    f_glacier = compute_glacier_cost(glacier_mask, slope_clean)

    logger.info("f_roughness")
    f_rough = compute_roughness_cost(rough_clean)

    cost = f_slope * f_alt * f_aspect * f_glacier * f_rough

    # landcover (WorldCover) si dispo
    if landcover_cost is not None:
        logger.info("f_landcover (WorldCover)")
        cost *= landcover_cost

    cost[nodata_mask] = NODATA_VALUE

    factors = {
        "slope": f_slope, "altitude": f_alt, "aspect": f_aspect,
        "glacier": f_glacier, "roughness": f_rough,
    }
    if landcover_cost is not None:
        factors["landcover"] = landcover_cost
    for f in factors.values():
        f[nodata_mask] = NODATA_VALUE

    logger.info("cost surface: %s, valid=%d px", cost.shape, np.sum(~nodata_mask))
    return cost, factors, nodata_mask


def build_base_cost(dem, slope, aspect, roughness, glacier_mask,
                    month=7, acclimatized=True, landcover_cost=None):
    """Surface de cout sans le facteur pente Tobler.
    Utilisee par le Dijkstra anisotrope qui calcule Tobler per-edge."""
    nodata_mask = (slope == NODATA_VALUE) | np.isnan(slope)

    slope_clean = np.where(nodata_mask, 0, slope)
    aspect_clean = np.where(nodata_mask, 0, aspect)
    rough_clean = np.where(nodata_mask, 0, roughness)
    dem_clean = np.where(nodata_mask, 0, dem)

    f_alt = compute_altitude_cost(dem_clean, acclimatized)
    f_aspect = compute_aspect_cost(aspect_clean, slope_clean, dem_clean, month)
    f_glacier = compute_glacier_cost(glacier_mask, slope_clean)
    f_rough = compute_roughness_cost(rough_clean)

    # tout sauf Tobler
    base = f_alt * f_aspect * f_glacier * f_rough

    if landcover_cost is not None:
        base *= landcover_cost

    base[nodata_mask] = np.inf

    logger.info("base cost (sans Tobler): %s, valid=%d px",
                base.shape, np.sum(~nodata_mask))
    return base, nodata_mask
