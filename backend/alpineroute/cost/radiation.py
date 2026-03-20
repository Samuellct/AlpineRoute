# radiation solaire avec ombres portees
# remplace le f_aspect cosinus par un modele physique:
# position solaire NOAA -> horizons -> ombres -> irradiance directe -> cout

import os
import hashlib
import logging
import math

import numpy as np
from scipy.ndimage import zoom

from alpineroute.config import (
    RADIATION_CACHE_DIR,
    RADIATION_N_AZIMUTHS,
    RADIATION_DEM_RESOLUTION,
    RADIATION_TIME_STEP_H,
    RADIATION_HORIZON_MAX_DIST_M,
    RADIATION_SUMMER_PENALTY,
    RADIATION_WINTER_PENALTY,
    RADIATION_SUMMER_MONTHS,
    RADIATION_SLOPE_THRESHOLD,
    RADIATION_ALTITUDE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ============================================================
# 1. position solaire (algo NOAA simplifie)
# ============================================================

def solar_position(day_of_year, hour_utc, lat_deg, lon_deg):
    """Position du soleil. Retourne (elevation_deg, azimuth_deg).
    Azimut: 0=Nord, 90=Est, 180=Sud, 270=Ouest (compass)."""
    # fraction d'annee en radians
    gamma = 2 * math.pi / 365.0 * (day_of_year - 1 + (hour_utc - 12) / 24.0)

    # declinaison solaire (Spencer 1971)
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))

    # equation du temps (minutes)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
                        - 0.014615 * math.cos(2 * gamma) - 0.04089 * math.sin(2 * gamma))

    # angle horaire
    time_offset = eqtime + 4.0 * lon_deg  # minutes
    true_solar_time = hour_utc * 60.0 + time_offset
    ha = math.radians((true_solar_time / 4.0) - 180.0)  # deg -> rad

    lat_r = math.radians(lat_deg)

    # elevation
    sin_elev = (math.sin(lat_r) * math.sin(decl)
                + math.cos(lat_r) * math.cos(decl) * math.cos(ha))
    sin_elev = max(-1.0, min(1.0, sin_elev))
    elev = math.degrees(math.asin(sin_elev))

    # azimut
    cos_elev = math.cos(math.radians(elev))
    if cos_elev > 1e-6:
        cos_az = (math.sin(decl) - math.sin(lat_r) * sin_elev) / (math.cos(lat_r) * cos_elev)
        cos_az = max(-1.0, min(1.0, cos_az))
        az = math.degrees(math.acos(cos_az))
        if ha > 0:
            az = 360.0 - az
    else:
        az = 180.0  # soleil au zenith, azimut indetermine

    return elev, az


# ============================================================
# 2. angles d'horizon (balayage radial vectorise)
# ============================================================

def compute_horizon_angles(dem, resolution, n_azimuths=36):
    """Angle d'horizon par pixel dans N directions.
    Retourne array (n_azimuths, H, W) en degres."""
    H, W = dem.shape
    horizons = np.zeros((n_azimuths, H, W), dtype=np.float32)
    azimuths = np.linspace(0, 360, n_azimuths, endpoint=False)

    # distances en pixels (espacement logarithmique)
    max_px = int(RADIATION_HORIZON_MAX_DIST_M / resolution)
    distances = [1, 2, 3, 5, 8, 13, 20, 32, 50, 80, 130, 200, 320, 500, 800]
    distances = [d for d in distances if d <= max_px]
    if not distances:
        distances = [1]

    for ai, az in enumerate(azimuths):
        az_rad = math.radians(az)
        # direction: dx=sin(az), dy=-cos(az) car y pointe vers le sud en raster
        dx = math.sin(az_rad)
        dy = -math.cos(az_rad)

        max_angle = np.full((H, W), -90.0, dtype=np.float32)

        for d in distances:
            # shift en pixels entiers (arrondi au plus proche)
            shift_x = int(round(dx * d))
            shift_y = int(round(dy * d))
            if shift_x == 0 and shift_y == 0:
                continue

            dist_m = math.sqrt(shift_x**2 + shift_y**2) * resolution

            # decaler le DEM: shifted[r,c] = dem[r+shift_y, c+shift_x]
            # (on veut l'altitude du point distant dans la direction az)
            shifted = np.full_like(dem, np.nan)
            src_y0 = max(0, shift_y)
            src_y1 = H - max(0, -shift_y)
            src_x0 = max(0, shift_x)
            src_x1 = W - max(0, -shift_x)
            dst_y0 = max(0, -shift_y)
            dst_y1 = H - max(0, shift_y)
            dst_x0 = max(0, -shift_x)
            dst_x1 = W - max(0, shift_x)

            if src_y1 <= src_y0 or src_x1 <= src_x0:
                continue

            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = dem[src_y0:src_y1, src_x0:src_x1]

            # angle d'elevation vers le point distant
            dz = shifted - dem
            angle = np.degrees(np.arctan2(dz, dist_m))
            valid = ~np.isnan(shifted)
            max_angle = np.where(valid & (angle > max_angle), angle, max_angle)

        horizons[ai] = np.maximum(max_angle, 0.0)

    return horizons


# ============================================================
# 3. ombre portee
# ============================================================

def is_shadowed(horizon_angles, sun_elevation, sun_azimuth, azimuths=None):
    """True si pixel a l'ombre. Interpole l'horizon entre 2 azimuths."""
    n_az = horizon_angles.shape[0]
    if azimuths is None:
        azimuths = np.linspace(0, 360, n_az, endpoint=False)

    step = 360.0 / n_az
    # index bas
    idx = ((sun_azimuth % 360.0) / step)
    i0 = int(idx) % n_az
    i1 = (i0 + 1) % n_az
    frac = idx - int(idx)

    # interpolation lineaire
    horizon_at_sun = horizon_angles[i0] * (1 - frac) + horizon_angles[i1] * frac
    return sun_elevation < horizon_at_sun


# ============================================================
# 4. irradiance directe sur surface inclinee
# ============================================================

def direct_irradiance(sun_elev, sun_az, slope_deg, aspect_deg, shadow_mask=None):
    """Irradiance directe normalisee. 0 si a l'ombre ou soleil sous l'horizon."""
    if sun_elev <= 0:
        return np.zeros_like(slope_deg)

    sun_zen = 90.0 - sun_elev
    sun_zen_r = np.radians(sun_zen)
    slope_r = np.radians(slope_deg)
    # diff azimut soleil - aspect terrain
    daz_r = np.radians(sun_az - aspect_deg)

    cos_inc = (np.cos(sun_zen_r) * np.cos(slope_r)
               + np.sin(sun_zen_r) * np.sin(slope_r) * np.cos(daz_r))
    irr = np.maximum(cos_inc, 0.0)

    if shadow_mask is not None:
        irr[shadow_mask] = 0.0

    return irr.astype(np.float32)


# ============================================================
# 5. integration journaliere
# ============================================================

def daily_radiation(dem, slope_deg, aspect_deg, resolution, lat, lon,
                    day_of_year, horizon_angles=None):
    """Cumul irradiance sur une journee (pas 30 min, 4h-22h UTC).
    Retourne array (H, W) en unites arbitraires normalisees."""
    dt = RADIATION_TIME_STEP_H
    total = np.zeros(dem.shape, dtype=np.float64)
    n_steps = 0

    for h_idx in range(int(4.0 / dt), int(22.0 / dt) + 1):
        hour = h_idx * dt
        elev, az = solar_position(day_of_year, hour, lat, lon)
        if elev <= 0:
            continue

        shadow = None
        if horizon_angles is not None:
            shadow = is_shadowed(horizon_angles, elev, az)

        irr = direct_irradiance(elev, az, slope_deg, aspect_deg, shadow)
        total += irr * dt
        n_steps += 1

    if n_steps == 0:
        return np.ones(dem.shape, dtype=np.float32) * 0.5

    return total.astype(np.float32)


# ============================================================
# 6. radiation -> facteur de cout
# ============================================================

def compute_radiation_cost(daily_rad, slope_deg, elevation, month):
    """Transforme la radiation journaliere en facteur de cout.
    Ete: forte radiation = penalite (neige molle, faces chaudes)
    Hiver: faible radiation = penalite (verglas, neige dure)"""
    cost = np.ones_like(daily_rad, dtype=np.float32)

    # normaliser 0-1 (robuste aux outliers)
    p5 = np.percentile(daily_rad[daily_rad > 0], 5) if np.any(daily_rad > 0) else 0
    p95 = np.percentile(daily_rad[daily_rad > 0], 95) if np.any(daily_rad > 0) else 1
    spread = max(p95 - p5, 0.01)
    norm = np.clip((daily_rad - p5) / spread, 0.0, 1.0)

    # zone ou la radiation compte (haute montagne, pentes significatives)
    active = (slope_deg > RADIATION_SLOPE_THRESHOLD) & (elevation > RADIATION_ALTITUDE_THRESHOLD)

    if month in RADIATION_SUMMER_MONTHS:
        # ete: penaliser les faces tres exposees
        penalty = 1.0 + RADIATION_SUMMER_PENALTY * norm
        cost = np.where(active, penalty, cost)
    else:
        # hiver: penaliser les faces ombrees (faible radiation)
        penalty = 1.0 + RADIATION_WINTER_PENALTY * (1.0 - norm)
        cost = np.where(active, penalty, cost)

    return cost.astype(np.float32)


# ============================================================
# 7. cache horizons + radiation mensuelle
# ============================================================

def _cache_key(bbox_l93, resolution, suffix=""):
    """Hash pour le cache radiation."""
    raw = (f"{bbox_l93['xmin']:.0f}_{bbox_l93['ymin']:.0f}_"
           f"{bbox_l93['xmax']:.0f}_{bbox_l93['ymax']:.0f}_"
           f"{resolution}{suffix}")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_npz(path, key):
    """Charge un array depuis un npz. None si absent."""
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path)
        return data[key]
    except Exception:
        return None


def _save_npz(path, key, arr):
    """Sauvegarde un array en npz compresse."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **{key: arr})


def load_horizon_cache(bbox_l93, resolution):
    """Charge horizons du cache. None si absent."""
    key = _cache_key(bbox_l93, resolution, "_horizon")
    path = os.path.join(RADIATION_CACHE_DIR, f"horizon_{key}.npz")
    return _load_npz(path, "horizons")


def save_horizon_cache(bbox_l93, resolution, horizons):
    """Sauvegarde horizons (permanent, pas de TTL)."""
    key = _cache_key(bbox_l93, resolution, "_horizon")
    path = os.path.join(RADIATION_CACHE_DIR, f"horizon_{key}.npz")
    _save_npz(path, "horizons", horizons)
    logger.info("horizon cache saved: %s", path)


def load_radiation_cache(bbox_l93, resolution, month):
    """Charge radiation mensuelle. None si absent."""
    key = _cache_key(bbox_l93, resolution, f"_rad_m{month:02d}")
    path = os.path.join(RADIATION_CACHE_DIR, f"rad_{key}.npz")
    return _load_npz(path, "daily_rad")


def save_radiation_cache(bbox_l93, resolution, month, daily_rad):
    """Sauvegarde radiation mensuelle."""
    key = _cache_key(bbox_l93, resolution, f"_rad_m{month:02d}")
    path = os.path.join(RADIATION_CACHE_DIR, f"rad_{key}.npz")
    _save_npz(path, "daily_rad", daily_rad)
    logger.info("radiation cache saved: %s (month=%d)", path, month)


# ============================================================
# 8. wrapper pipeline
# ============================================================

def get_radiation_cost(dem, slope_deg, aspect_deg, resolution, bbox_l93, bbox_wgs84, month):
    """Pipeline complet radiation. Retourne facteur de cout, ou None si echec."""
    try:
        # centre bbox en WGS84 pour la position solaire
        lat = (bbox_wgs84["lat_min"] + bbox_wgs84["lat_max"]) / 2.0
        lon = (bbox_wgs84["lon_min"] + bbox_wgs84["lon_max"]) / 2.0

        # resolution de travail pour les horizons
        work_res = max(resolution, RADIATION_DEM_RESOLUTION)
        scale = resolution / work_res

        if scale < 1.0:
            # downsampler le DEM pour les horizons
            dem_low = zoom(dem, scale, order=1)
            slope_low = zoom(slope_deg, scale, order=1)
            aspect_low = zoom(aspect_deg, scale, order=0)  # nearest pour aspect
        else:
            dem_low = dem
            slope_low = slope_deg
            aspect_low = aspect_deg
            work_res = resolution

        # horizons (cache permanent)
        horizons = load_horizon_cache(bbox_l93, work_res)
        if horizons is not None:
            # verif taille compatible
            if horizons.shape[1:] != dem_low.shape:
                logger.info("horizon cache shape mismatch, recalcul")
                horizons = None

        if horizons is None:
            logger.info("computing horizon angles (%dx%d, res=%.1fm, %d azimuths)",
                        dem_low.shape[0], dem_low.shape[1], work_res, RADIATION_N_AZIMUTHS)
            horizons = compute_horizon_angles(dem_low, work_res, RADIATION_N_AZIMUTHS)
            save_horizon_cache(bbox_l93, work_res, horizons)
        else:
            logger.info("horizon cache hit (%dx%d)", dem_low.shape[0], dem_low.shape[1])

        # radiation journaliere (cache mensuel)
        daily_rad = load_radiation_cache(bbox_l93, work_res, month)
        if daily_rad is not None and daily_rad.shape != dem_low.shape:
            daily_rad = None

        if daily_rad is None:
            # jour representatif = 15 du mois
            doy = _month_to_doy(month)
            logger.info("computing daily radiation (doy=%d, lat=%.2f, lon=%.2f)", doy, lat, lon)
            daily_rad = daily_radiation(dem_low, slope_low, aspect_low, work_res,
                                        lat, lon, doy, horizon_angles=horizons)
            save_radiation_cache(bbox_l93, work_res, month, daily_rad)
        else:
            logger.info("radiation cache hit (month=%d)", month)

        # upsampler si necessaire
        if daily_rad.shape != dem.shape:
            up_scale = (dem.shape[0] / daily_rad.shape[0],
                        dem.shape[1] / daily_rad.shape[1])
            daily_rad = zoom(daily_rad, up_scale, order=1)

        # masque nodata sur slope original
        from alpineroute.config import NODATA_VALUE
        nodata = (slope_deg == NODATA_VALUE) | np.isnan(slope_deg)
        slope_clean = np.where(nodata, 0, slope_deg)
        dem_clean = np.where(nodata, 0, dem)

        cost = compute_radiation_cost(daily_rad, slope_clean, dem_clean, month)
        logger.info("radiation cost computed: range [%.2f, %.2f]", cost.min(), cost.max())
        return cost

    except Exception as e:
        logger.warning("radiation fallback to aspect: %s", e)
        return None


def _month_to_doy(month):
    """Jour representatif du mois (le 15)."""
    # cumul jours par mois (non-bissextile)
    days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return sum(days[:month]) + 15
