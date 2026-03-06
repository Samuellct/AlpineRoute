# barrieres OSM : rivieres, canaux, autoroutes, ponts, gues

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
    RIVER_BUFFER_M, CANAL_BUFFER_M, STREAM_BUFFER_M,
    BRIDGE_BUFFER_M, MOTORWAY_BUFFER_M,
)
from alpineroute.utils import l93_to_wgs84

logger = logging.getLogger(__name__)


def _bbox_cache_key(bbox_l93):
    raw = f"{bbox_l93['xmin']:.0f}_{bbox_l93['ymin']:.0f}_{bbox_l93['xmax']:.0f}_{bbox_l93['ymax']:.0f}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_valid(path):
    if not os.path.exists(path):
        return False
    age_days = (time.time() - os.path.getmtime(path)) / 86400
    return age_days < OSM_CACHE_TTL_DAYS


def _parse_overpass_ways(data, tag_keys):
    """Parse les ways Overpass (out geom;) en GeoDataFrame."""
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

def download_barriers(bbox_l93):
    """Telecharge waterways + motorways + bridges + fords depuis Overpass."""
    os.makedirs(OSM_CACHE_DIR, exist_ok=True)
    cache_key = _bbox_cache_key(bbox_l93)
    cache_path = os.path.join(OSM_CACHE_DIR, f"barriers_{cache_key}.gpkg")

    if _cache_valid(cache_path):
        logger.info("barriers cache hit: %s", cache_path)
        gdf = gpd.read_file(cache_path)
        if gdf.crs and gdf.crs.to_epsg() != 2154:
            gdf = gdf.to_crs(CRS_L93)
        return gdf

    lon_min, lat_min = l93_to_wgs84(bbox_l93["xmin"], bbox_l93["ymin"])
    lon_max, lat_max = l93_to_wgs84(bbox_l93["xmax"], bbox_l93["ymax"])
    bbox_str = f"{lat_min},{lon_min},{lat_max},{lon_max}"

    query = f"""[out:json][timeout:{OVERPASS_TIMEOUT}];
(
  way["waterway"~"^(river|stream|canal)$"]({bbox_str});
  way["highway"~"^(motorway|motorway_link|trunk|trunk_link)$"]({bbox_str});
  way["bridge"="yes"]({bbox_str});
  way["ford"="yes"]({bbox_str});
);
out geom;"""

    logger.info("overpass barriers query %s", bbox_str)
    resp = httpx.post(OVERPASS_URL, data={"data": query}, timeout=OVERPASS_TIMEOUT + 10)
    resp.raise_for_status()

    tag_keys = ["waterway", "highway", "bridge", "ford"]
    gdf = _parse_overpass_ways(resp.json(), tag_keys)

    if len(gdf) == 0:
        logger.warning("aucune barriere dans la bbox")
        return gdf

    gdf = gdf.to_crs(CRS_L93)
    gdf.to_file(cache_path, driver="GPKG")
    logger.info("barriers cache saved: %d elements -> %s", len(gdf), cache_path)
    return gdf


# --- masks ---

def _rasterize_bool(geometries, buffer_m, transform, shape):
    """Buffer + rasterize une liste de geometries en masque bool."""
    if not geometries:
        return np.zeros(shape, dtype=bool)
    shapes = []
    for geom in geometries:
        b = geom.buffer(buffer_m)
        if not b.is_empty:
            shapes.append((b, 1))
    if not shapes:
        return np.zeros(shape, dtype=bool)
    arr = rasterize(shapes, out_shape=shape, transform=transform,
                    fill=0, dtype=np.uint8, all_touched=True)
    return arr.astype(bool)


def build_barrier_masks(gdf, transform, shape):
    """Construit barrier_mask et stream_mask a partir du GDF barriers.
    barrier_mask = rivieres + canaux + autoroutes, troue par ponts/gues.
    stream_mask = ruisseaux (penalite, pas blocage)."""
    if len(gdf) == 0:
        return {
            "barrier_mask": np.zeros(shape, dtype=bool),
            "stream_mask": np.zeros(shape, dtype=bool),
        }

    # separer par type
    ww = gdf["waterway"].fillna("")
    hw = gdf["highway"].fillna("")
    br = gdf["bridge"].fillna("")
    fo = gdf["ford"].fillna("")

    rivers = gdf[ww == "river"].geometry.tolist()
    canals = gdf[ww == "canal"].geometry.tolist()
    streams = gdf[ww == "stream"].geometry.tolist()
    motorways = gdf[hw.isin(["motorway", "motorway_link", "trunk", "trunk_link"])].geometry.tolist()
    bridges = gdf[br == "yes"].geometry.tolist()
    fords = gdf[fo == "yes"].geometry.tolist()

    # barrieres
    barrier_mask = _rasterize_bool(rivers, RIVER_BUFFER_M, transform, shape)
    barrier_mask |= _rasterize_bool(canals, CANAL_BUFFER_M, transform, shape)
    barrier_mask |= _rasterize_bool(motorways, MOTORWAY_BUFFER_M, transform, shape)

    # passages (ponts + gues)
    passage_mask = _rasterize_bool(bridges, BRIDGE_BUFFER_M, transform, shape)
    passage_mask |= _rasterize_bool(fords, BRIDGE_BUFFER_M, transform, shape)
    barrier_mask &= ~passage_mask

    # ruisseaux
    stream_mask = _rasterize_bool(streams, STREAM_BUFFER_M, transform, shape)

    n_barrier = np.sum(barrier_mask)
    n_stream = np.sum(stream_mask)
    logger.info("barriers: %d px blocked, %d px stream, %d rivers, %d streams, %d bridges",
                n_barrier, n_stream, len(rivers), len(streams), len(bridges))
    return {"barrier_mask": barrier_mask, "stream_mask": stream_mask}


# --- wrapper pipeline ---

def get_barrier_masks(bbox_l93, transform, shape):
    """Pipeline complet barriers: download -> build masks.
    Retourne None si echec (non bloquant)."""
    try:
        gdf = download_barriers(bbox_l93)
        masks = build_barrier_masks(gdf, transform, shape)
        logger.info("barrier masks ready: %s", shape)
        return masks
    except Exception as e:
        logger.warning("barriers echec (non bloquant): %s", e)
        return None
