# telechargement DEM IGN Lidar HD + fallback Copernicus GLO-30
# source: T01_dem_download.py, adapte pour bbox dynamique + cache

import os
import math
import time
import logging
from urllib.parse import urlparse, parse_qs, urlencode

import httpx
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds

from alpineroute.config import (
    WFS_URL, WFS_TYPENAME, HTTP_TIMEOUT, MAX_RETRIES, RETRY_DELAY,
    CRS_L93, CRS_WGS84, NODATA_VALUE, DEM_CACHE_DIR,
    COPERNICUS_S3_BASE, COPERNICUS_DL_TIMEOUT,
)
from alpineroute.utils import DownloadError, DataNotFoundError

logger = logging.getLogger(__name__)


# --- WFS discovery ---

def discover_tiles(bbox_l93, resolution=1.0):
    """Interroge le WFS IGN pour trouver les dalles qui intersectent la bbox.
    Retourne la liste de features GeoJSON."""
    bbox_str = "{xmin},{ymin},{xmax},{ymax}".format(**bbox_l93)

    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": WFS_TYPENAME,
        "OUTPUTFORMAT": "application/json",
        "BBOX": f"{bbox_str},EPSG:2154",
        "COUNT": "500",
    }

    logger.info("WFS query bbox L93: %s", bbox_str)
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        resp = client.get(WFS_URL, params=params)
        resp.raise_for_status()

    data = resp.json()
    features = data.get("features", [])
    logger.info("%d dalles trouvees", len(features))
    return features


# download d'une dalle

def _build_tile_url(tile_feature, resolution):
    """Construit l'URL WMS-R avec la bonne taille de pixel."""
    props = tile_feature.get("properties", {})
    base_url = props.get("url", "")
    if not base_url:
        return None

    parsed = urlparse(base_url)
    params = parse_qs(parsed.query, keep_blank_values=True)

    bbox_str = props.get("bbox", "")
    if bbox_str:
        parts = [float(x) for x in bbox_str.split(",")]
        width_m = parts[2] - parts[0]
        height_m = parts[3] - parts[1]
    else:
        width_m, height_m = 1000, 1000

    width_px = int(round(width_m / resolution))
    height_px = int(round(height_m / resolution))

    flat_params = {k: v[0] for k, v in params.items()}
    flat_params["WIDTH"] = str(width_px)
    flat_params["HEIGHT"] = str(height_px)

    new_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{urlencode(flat_params)}"
    return new_url, width_px, height_px


def download_tile(tile_feature, cache_dir, resolution=1.0):
    """Telecharge une dalle, retourne le path du fichier. Verifie le cache."""
    props = tile_feature.get("properties", {})
    tile_name = props.get("name", "unknown")
    out_path = os.path.join(cache_dir, f"{tile_name}.tif")

    # check cache
    if os.path.exists(out_path):
        try:
            with rasterio.open(out_path) as ds:
                _ = ds.shape
            logger.debug("cache hit: %s", tile_name)
            return out_path
        except Exception:
            os.remove(out_path)

    result = _build_tile_url(tile_feature, resolution)
    if result is None:
        logger.warning("pas d'url pour %s, skip", tile_name)
        return None

    url, w, h = result
    logger.info("download %s (%dx%d)...", tile_name, w, h)

    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                resp = client.get(url)
                resp.raise_for_status()

            ct = resp.headers.get("content-type", "")
            if "tiff" not in ct and "image" not in ct:
                logger.warning("reponse inattendue (%s): %s", ct, resp.text[:200])
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return None

            with open(out_path, "wb") as f:
                f.write(resp.content)

            # verif rapide
            with rasterio.open(out_path) as ds:
                _ = ds.shape
            return out_path

        except Exception as e:
            logger.warning("retry %d/%d: %s", attempt + 1, MAX_RETRIES, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    return None


# ==== Mosaic + crop ====

def build_mosaic(tile_paths, bbox_l93, resolution):
    """Merge les dalles, reproject/crop sur la bbox, retourne le path."""
    if not tile_paths:
        raise DataNotFoundError("Aucune dalle a traiter")

    output_dir = os.path.join(DEM_CACHE_DIR, "mosaic")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"dem_{bbox_l93['xmin']}_{bbox_l93['ymin']}_{resolution}m.tif"
    )

    # check cache mosaic
    if os.path.exists(output_path):
        try:
            with rasterio.open(output_path) as ds:
                data = ds.read(1)
                valid = data[data != NODATA_VALUE]
                if len(valid) > 0:
                    logger.info("mosaic en cache: %s", output_path)
                    return output_path
        except Exception:
            pass

    logger.info("merge %d dalles...", len(tile_paths))
    datasets = [rasterio.open(p) for p in tile_paths]

    if len(datasets) == 1:
        mosaic = datasets[0].read(1)
        mosaic_transform = datasets[0].transform
        mosaic_crs = datasets[0].crs
        src_nodata = datasets[0].nodata
    else:
        mosaic, mosaic_transform = merge(datasets, nodata=NODATA_VALUE)
        mosaic = mosaic[0]
        mosaic_crs = datasets[0].crs
        src_nodata = NODATA_VALUE

    for ds in datasets:
        ds.close()

    # target grid
    width = int(round((bbox_l93["xmax"] - bbox_l93["xmin"]) / resolution))
    height = int(round((bbox_l93["ymax"] - bbox_l93["ymin"]) / resolution))

    dst_transform = from_bounds(
        bbox_l93["xmin"], bbox_l93["ymin"],
        bbox_l93["xmax"], bbox_l93["ymax"],
        width, height,
    )
    dst_array = np.full((height, width), NODATA_VALUE, dtype=np.float32)

    logger.info("reproject -> L93, %dx%d, res=%sm", width, height, resolution)
    reproject(
        source=mosaic.astype(np.float32),
        destination=dst_array,
        src_transform=mosaic_transform,
        src_crs=mosaic_crs,
        dst_transform=dst_transform,
        dst_crs=CRS_L93,
        resampling=Resampling.bilinear,
        src_nodata=src_nodata if src_nodata is not None else NODATA_VALUE,
        dst_nodata=NODATA_VALUE,
    )

    # nettoyage artefacts
    bad = dst_array < -100
    if bad.any():
        logger.info("fix %d pixels aberrants -> nodata", bad.sum())
        dst_array[bad] = NODATA_VALUE

    # ecriture
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": width,
        "height": height,
        "count": 1,
        "crs": CRS_L93,
        "transform": dst_transform,
        "nodata": NODATA_VALUE,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(dst_array, 1)

    logger.info("mosaic ecrite: %s", output_path)
    return output_path


# --- Fallback Copernicus GLO-30 ---

def _copernicus_tile_names(bbox_wgs84):
    """Calcule les noms de tuiles Copernicus qui couvrent la bbox.
    Chaque tuile = 1 deg x 1 deg."""
    lat_min = int(math.floor(bbox_wgs84["lat_min"]))
    lat_max = int(math.floor(bbox_wgs84["lat_max"]))
    lon_min = int(math.floor(bbox_wgs84["lon_min"]))
    lon_max = int(math.floor(bbox_wgs84["lon_max"]))

    tiles = []
    for lat in range(lat_min, lat_max + 1):
        for lon in range(lon_min, lon_max + 1):
            lat_str = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
            lon_str = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
            name = f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"
            tiles.append(name)
    return tiles


def download_copernicus_tile(tile_name, cache_dir):
    """Telecharge une tuile Copernicus depuis S3. Retourne le path ou None."""
    out_path = os.path.join(cache_dir, f"{tile_name}.tif")

    if os.path.exists(out_path):
        try:
            with rasterio.open(out_path) as ds:
                _ = ds.shape
            logger.debug("copernicus cache hit: %s", tile_name)
            return out_path
        except Exception:
            os.remove(out_path)

    url = f"{COPERNICUS_S3_BASE}/{tile_name}/{tile_name}.tif"
    logger.info("copernicus download %s...", tile_name)

    try:
        with httpx.Client(timeout=COPERNICUS_DL_TIMEOUT, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()

        with open(out_path, "wb") as f:
            f.write(resp.content)

        # verif rapide
        with rasterio.open(out_path) as ds:
            _ = ds.shape
        logger.info("copernicus ok (%s, %.1f MB)", tile_name, len(resp.content) / 1024 / 1024)
        return out_path

    except Exception as e:
        logger.warning("copernicus echec %s: %s", tile_name, e)
        if os.path.exists(out_path):
            os.remove(out_path)
        return None


def get_dem_copernicus(bbox_wgs84, bbox_l93, resolution, cache_dir=None,
                       progress_callback=None):
    """Fallback: telecharge les tuiles Copernicus GLO-30 et construit la mosaic.
    Resolution native ~30m, on resample sur la grille L93 demandee."""
    if cache_dir is None:
        cache_dir = os.path.join(DEM_CACHE_DIR, "copernicus")
    os.makedirs(cache_dir, exist_ok=True)

    tile_names = _copernicus_tile_names(bbox_wgs84)
    logger.info("copernicus: %d tuiles a verifier", len(tile_names))

    tile_paths = []
    for i, name in enumerate(tile_names):
        if progress_callback:
            progress_callback("download_copernicus", i, len(tile_names))
        path = download_copernicus_tile(name, cache_dir)
        if path:
            tile_paths.append(path)

    if not tile_paths:
        raise DownloadError("Aucune tuile Copernicus telechargee")

    if progress_callback:
        progress_callback("download_copernicus", len(tile_names), len(tile_names))

    # reuse build_mosaic - les tuiles sont en WGS84 mais build_mosaic
    # fait le reproject vers L93
    if progress_callback:
        progress_callback("mosaic", 0, 1)
    dem_path = build_mosaic(tile_paths, bbox_l93, resolution)
    if progress_callback:
        progress_callback("mosaic", 1, 1)

    return dem_path


# pipeline complet

def get_dem(bbox_l93, resolution=1.0, cache_dir=None, progress_callback=None,
            bbox_wgs84=None):
    """Pipeline complet: discover > download > mosaic.
    Si aucune dalle IGN disponible (hors France), fallback Copernicus GLO-30.
    Retourne le chemin du DEM final.
    progress_callback(step, current, total) pour SSE."""
    if cache_dir is None:
        cache_dir = os.path.join(DEM_CACHE_DIR, "lidar_hd")
    os.makedirs(cache_dir, exist_ok=True)

    # 1. discover dalles IGN
    if progress_callback:
        progress_callback("discover", 0, 1)
    features = discover_tiles(bbox_l93, resolution)

    # pas de dalle IGN -> fallback Copernicus (zone hors France)
    if not features:
        if bbox_wgs84 is None:
            raise DownloadError("Aucune dalle IGN, et pas de bbox WGS84 pour le fallback Copernicus")
        logger.info("aucune dalle IGN, fallback Copernicus GLO-30")
        return get_dem_copernicus(bbox_wgs84, bbox_l93, resolution,
                                  progress_callback=progress_callback)

    # 2. download
    tile_paths = []
    for i, feat in enumerate(features):
        if progress_callback:
            progress_callback("download", i, len(features))

        path = download_tile(feat, cache_dir, resolution)
        if path:
            tile_paths.append(path)
            time.sleep(0.2)  # courtoisie API

    if not tile_paths:
        raise DownloadError("Aucune dalle telechargee avec succes")

    if progress_callback:
        progress_callback("download", len(features), len(features))

    # 3. mosaic
    if progress_callback:
        progress_callback("mosaic", 0, 1)
    dem_path = build_mosaic(tile_paths, bbox_l93, resolution)

    if progress_callback:
        progress_callback("mosaic", 1, 1)

    return dem_path
