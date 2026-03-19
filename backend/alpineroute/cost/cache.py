# cache surface de cout pre-calculee
# evite de recalculer DEM + terrain + worldcover + glacier a chaque requete
# cle = hash(bbox_arrondi + resolution + mois)

import os
import json
import time
import hashlib
import logging

import numpy as np
from rasterio.transform import Affine

from alpineroute.config import COST_CACHE_DIR, COST_CACHE_MAX_AGE_DAYS

logger = logging.getLogger(__name__)


def cost_cache_key(bbox_l93, resolution, month):
    """Hash sha256 de la bbox arrondie + resolution + mois."""
    raw = (f"{bbox_l93['xmin']:.0f}_{bbox_l93['ymin']:.0f}_"
           f"{bbox_l93['xmax']:.0f}_{bbox_l93['ymax']:.0f}_"
           f"{resolution}_{month}")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_cached_cost(cache_key):
    """Charge le cache si present et pas expire. Retourne dict ou None."""
    npz_path = os.path.join(COST_CACHE_DIR, f"{cache_key}.npz")
    meta_path = os.path.join(COST_CACHE_DIR, f"{cache_key}.json")

    if not os.path.exists(npz_path) or not os.path.exists(meta_path):
        return None

    # verif TTL sur le npz
    age_days = (time.time() - os.path.getmtime(npz_path)) / 86400
    if age_days > COST_CACHE_MAX_AGE_DAYS:
        logger.info("cost cache expire: %s (%.0f jours)", cache_key, age_days)
        return None

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)

        data = np.load(npz_path)
        # reconstruire le transform depuis les 6 floats
        t = meta["transform"]
        transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])

        result = {
            "cached_base": data["cached_base"],
            "slope_deg": data["slope_deg"],
            "dem": data["dem"],
            "glacier_mask": data["glacier_mask"].astype(bool),
            "nodata_mask": data["nodata_mask"].astype(bool),
            "transform": transform,
            "metadata": meta,
        }
        logger.info("cost cache hit: %s (%.0f jours)", cache_key, age_days)
        return result
    except Exception as e:
        logger.warning("cost cache load fail: %s", e)
        return None


def save_cost_cache(cache_key, cached_base, slope_deg, dem, glacier_mask,
                    nodata_mask, transform, bbox_l93, resolution, month):
    """Sauvegarde la base pre-calculee + metadata."""
    os.makedirs(COST_CACHE_DIR, exist_ok=True)
    npz_path = os.path.join(COST_CACHE_DIR, f"{cache_key}.npz")
    meta_path = os.path.join(COST_CACHE_DIR, f"{cache_key}.json")

    try:
        # glacier_mask peut etre None
        gl = glacier_mask if glacier_mask is not None else np.zeros(dem.shape, dtype=bool)

        np.savez_compressed(
            npz_path,
            cached_base=cached_base.astype(np.float32),
            slope_deg=slope_deg.astype(np.float32),
            dem=dem.astype(np.float32),
            glacier_mask=gl.astype(np.uint8),
            nodata_mask=nodata_mask.astype(np.uint8),
        )

        # Affine -> tuple 6 floats
        t = transform
        meta = {
            "bbox_l93": bbox_l93,
            "resolution": resolution,
            "month": month,
            "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
            "shape": list(dem.shape),
            "created": time.time(),
            "version": "2.0.0-beta.3",
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        size_mb = os.path.getsize(npz_path) / 1e6
        logger.info("cost cache saved: %s (%.1f MB)", cache_key, size_mb)
    except Exception as e:
        logger.warning("cost cache save fail: %s", e)


def invalidate_cache(bbox_l93=None):
    """Supprime les entrees du cache. Si bbox fourni, seulement celles qui intersectent."""
    if not os.path.isdir(COST_CACHE_DIR):
        return 0

    count = 0
    for fname in os.listdir(COST_CACHE_DIR):
        if not fname.endswith(".json"):
            continue

        meta_path = os.path.join(COST_CACHE_DIR, fname)
        key = fname[:-5]  # sans .json
        npz_path = os.path.join(COST_CACHE_DIR, f"{key}.npz")

        if bbox_l93 is not None:
            # verif intersection
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                cb = meta["bbox_l93"]
                # pas d'intersection si separes
                if (cb["xmax"] < bbox_l93["xmin"] or cb["xmin"] > bbox_l93["xmax"]
                        or cb["ymax"] < bbox_l93["ymin"] or cb["ymin"] > bbox_l93["ymax"]):
                    continue
            except Exception:
                continue

        # supprimer npz + json
        for p in [npz_path, meta_path]:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
        count += 1

    logger.info("cost cache invalidated: %d entries", count)
    return count


def cache_stats():
    """Stats du cache: nb entries, taille totale, age min/max."""
    if not os.path.isdir(COST_CACHE_DIR):
        return {"entries": 0, "total_size_mb": 0, "age_min_days": 0, "age_max_days": 0}

    entries = 0
    total_size = 0
    ages = []
    now = time.time()

    for fname in os.listdir(COST_CACHE_DIR):
        if not fname.endswith(".npz"):
            continue
        fpath = os.path.join(COST_CACHE_DIR, fname)
        total_size += os.path.getsize(fpath)
        ages.append((now - os.path.getmtime(fpath)) / 86400)
        entries += 1

    return {
        "entries": entries,
        "total_size_mb": round(total_size / 1e6, 1),
        "age_min_days": round(min(ages), 1) if ages else 0,
        "age_max_days": round(max(ages), 1) if ages else 0,
    }
