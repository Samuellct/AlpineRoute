# download + classification + rasterisation des sentiers OSM

import os
import time
import hashlib
import logging

import numpy as np
import geopandas as gpd
import httpx
from shapely.geometry import LineString
from rasterio.features import rasterize

from alpineroute.config import (
    CRS_L93, OVERPASS_URL, OVERPASS_TIMEOUT,
    OSM_CACHE_DIR, OSM_CACHE_TTL_DAYS,
    TRAIL_COST_MULTIPLIERS, TRAIL_BUFFER_M,
)
from alpineroute.utils import l93_to_wgs84

logger = logging.getLogger(__name__)


# --- cache helpers ---

def _bbox_cache_key(bbox_l93):
    raw = f"{bbox_l93['xmin']:.0f}_{bbox_l93['ymin']:.0f}_{bbox_l93['xmax']:.0f}_{bbox_l93['ymax']:.0f}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_valid(path):
    if not os.path.exists(path):
        return False
    age_days = (time.time() - os.path.getmtime(path)) / 86400
    return age_days < OSM_CACHE_TTL_DAYS


# --- overpass parsing ---

def _parse_overpass_ways(data, tag_keys):
    """Parse les ways de la reponse Overpass (out geom;) en GeoDataFrame."""
    rows = []
    for elem in data.get("elements", []):
        if elem.get("type") != "way":
            continue
        geom_pts = elem.get("geometry", [])
        if len(geom_pts) < 2:
            continue
        coords = [(p["lon"], p["lat"]) for p in geom_pts]
        tags = elem.get("tags", {})
        row = {"geometry": LineString(coords)}
        for k in tag_keys:
            row[k] = tags.get(k, "")
        rows.append(row)

    if not rows:
        return gpd.GeoDataFrame(columns=["geometry"] + tag_keys, crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


# --- download ---

def download_trails(bbox_l93):
    """Telecharge les highways depuis Overpass, cache en .gpkg."""
    os.makedirs(OSM_CACHE_DIR, exist_ok=True)
    cache_key = _bbox_cache_key(bbox_l93)
    cache_path = os.path.join(OSM_CACHE_DIR, f"trails_{cache_key}.gpkg")

    if _cache_valid(cache_path):
        logger.info("trails cache hit: %s", cache_path)
        gdf = gpd.read_file(cache_path)
        if gdf.crs and gdf.crs.to_epsg() != 2154:
            gdf = gdf.to_crs(CRS_L93)
        return gdf

    # bbox WGS84
    lon_min, lat_min = l93_to_wgs84(bbox_l93["xmin"], bbox_l93["ymin"])
    lon_max, lat_max = l93_to_wgs84(bbox_l93["xmax"], bbox_l93["ymax"])

    query = f"""[out:json][timeout:{OVERPASS_TIMEOUT}];
way["highway"]({lat_min},{lon_min},{lat_max},{lon_max});
out geom;"""

    logger.info("overpass trails query %.4f,%.4f -> %.4f,%.4f", lat_min, lon_min, lat_max, lon_max)
    resp = httpx.post(OVERPASS_URL, data={"data": query}, timeout=OVERPASS_TIMEOUT + 10)
    resp.raise_for_status()

    tag_keys = ["highway", "surface", "tracktype", "sac_scale", "foot", "access"]
    gdf = _parse_overpass_ways(resp.json(), tag_keys)

    if len(gdf) == 0:
        logger.warning("aucun highway dans la bbox")
        return gdf

    # reprojection WGS84 -> L93
    gdf = gdf.to_crs(CRS_L93)

    gdf.to_file(cache_path, driver="GPKG")
    logger.info("trails cache saved: %d ways -> %s", len(gdf), cache_path)
    return gdf


# --- classification ---

def classify_trails(gdf):
    """Classification en 12 niveaux. Retourne GDF filtre (sans EXCLUDE/BARRIER/unclassified)."""
    if len(gdf) == 0:
        gdf = gdf.copy()
        gdf["trail_class"] = []
        gdf["trail_cost"] = []
        return gdf

    gdf = gdf.copy()
    gdf["trail_class"] = "unclassified"
    gdf["trail_cost"] = 1.0

    hw = gdf["highway"].fillna("")
    surface = gdf["surface"].fillna("")
    tracktype = gdf["tracktype"].fillna("")
    sac = gdf["sac_scale"].fillna("")
    foot = gdf["foot"].fillna("")
    access = gdf["access"].fillna("")

    # regles par priorite croissante (les dernieres ecrasent)
    # R12 - routes
    mask = hw.isin(["residential", "tertiary", "secondary", "service"])
    gdf.loc[mask, "trail_class"] = "road"
    gdf.loc[mask, "trail_cost"] = TRAIL_COST_MULTIPLIERS["road"]

    # R11 - sentiers generiques
    mask = hw.isin(["path", "footway"])
    gdf.loc[mask, "trail_class"] = "trail_default"
    gdf.loc[mask, "trail_cost"] = TRAIL_COST_MULTIPLIERS["trail_default"]

    # R10 - pistes degradees
    mask = (hw == "track") & tracktype.isin(["grade4", "grade5"])
    gdf.loc[mask, "trail_class"] = "track_soft"
    gdf.loc[mask, "trail_cost"] = TRAIL_COST_MULTIPLIERS["track_soft"]

    # R9 - T6
    mask = sac == "difficult_alpine_hiking"
    gdf.loc[mask, "trail_class"] = "trail_t6"
    gdf.loc[mask, "trail_cost"] = TRAIL_COST_MULTIPLIERS["trail_t6"]

    # R8 - T5
    mask = sac == "demanding_alpine_hiking"
    gdf.loc[mask, "trail_class"] = "trail_t5"
    gdf.loc[mask, "trail_cost"] = TRAIL_COST_MULTIPLIERS["trail_t5"]

    # R7 - T4
    mask = sac == "alpine_hiking"
    gdf.loc[mask, "trail_class"] = "trail_t4"
    gdf.loc[mask, "trail_cost"] = TRAIL_COST_MULTIPLIERS["trail_t4"]

    # R6 - T3
    mask = sac == "demanding_mountain_hiking"
    gdf.loc[mask, "trail_class"] = "trail_t3"
    gdf.loc[mask, "trail_cost"] = TRAIL_COST_MULTIPLIERS["trail_t3"]

    # R5 - T1/T2
    mask = sac.isin(["hiking", "mountain_hiking"])
    gdf.loc[mask, "trail_class"] = "trail_t1t2"
    gdf.loc[mask, "trail_cost"] = TRAIL_COST_MULTIPLIERS["trail_t1t2"]

    # R4 - pistes dures
    mask = (hw == "track") & tracktype.isin(["grade1", "grade2"])
    gdf.loc[mask, "trail_class"] = "gravel"
    gdf.loc[mask, "trail_cost"] = TRAIL_COST_MULTIPLIERS["gravel"]

    # R3 - surface revetu
    mask = surface.isin(["asphalt", "paved", "concrete"])
    gdf.loc[mask, "trail_class"] = "paved"
    gdf.loc[mask, "trail_cost"] = TRAIL_COST_MULTIPLIERS["paved"]

    # R2 - autoroutes -> BARRIER
    mask = hw.isin(["motorway", "motorway_link", "trunk", "trunk_link"])
    gdf.loc[mask, "trail_class"] = "BARRIER"
    gdf.loc[mask, "trail_cost"] = np.inf

    # R1 - interdit
    mask = (foot == "no") | (access == "private")
    gdf.loc[mask, "trail_class"] = "EXCLUDE"
    gdf.loc[mask, "trail_cost"] = np.nan

    # filtre: virer EXCLUDE, BARRIER, unclassified
    gdf = gdf[~gdf["trail_class"].isin(["EXCLUDE", "BARRIER", "unclassified"])]
    gdf = gdf.reset_index(drop=True)

    logger.info("classify: %d trails gardes", len(gdf))
    return gdf


# --- rasterisation ---

def _get_buffer_width(trail_class):
    if trail_class in ("road", "paved", "gravel"):
        return TRAIL_BUFFER_M["road"]
    elif trail_class in ("trail_t4", "trail_t5", "trail_t6"):
        return TRAIL_BUFFER_M["alpine"]
    else:
        return TRAIL_BUFFER_M["trail"]


def rasterize_trail_cost(gdf, transform, shape):
    """Rasterise les sentiers classes en raster float32 (1.0 = hors sentier)."""
    if len(gdf) == 0:
        return np.ones(shape, dtype=np.float32)

    # buffer + tri par cout decroissant (le meilleur gagne via MergeAlg.replace)
    buffered = []
    for _, row in gdf.iterrows():
        w = _get_buffer_width(row["trail_class"])
        geom = row.geometry.buffer(w)
        if not geom.is_empty:
            buffered.append((geom, row["trail_cost"]))

    if not buffered:
        return np.ones(shape, dtype=np.float32)

    # tri decroissant: 1.0 rasterise en premier, 0.5 en dernier -> 0.5 ecrase
    buffered.sort(key=lambda x: x[1], reverse=True)

    result = rasterize(
        buffered,
        out_shape=shape,
        transform=transform,
        fill=1.0,
        dtype=np.float32,
        all_touched=True,
    )

    n_trail = np.sum(result < 1.0)
    pct = n_trail / result.size * 100
    logger.info("trail raster: %d sentiers, %d px couverts (%.1f%%)", len(gdf), n_trail, pct)
    return result


# --- wrapper pipeline ---

def get_trail_cost(bbox_l93, transform, shape):
    """Pipeline complet trails: download -> classify -> rasterize.
    Retourne None si echec (non bloquant)."""
    try:
        gdf = download_trails(bbox_l93)
        gdf = classify_trails(gdf)
        raster = rasterize_trail_cost(gdf, transform, shape)
        logger.info("trail cost ready: %s", raster.shape)
        return raster
    except Exception as e:
        logger.warning("trails echec (non bloquant): %s", e)
        return None
