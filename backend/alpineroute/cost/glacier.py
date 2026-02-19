# masque glaciaire RGI 7.0

import os
import logging
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize

from alpineroute.config import RGI_DIR, CRS_L93

logger = logging.getLogger(__name__)


def find_rgi_shapefiles(rgi_dir=None):
    """Glob recursif pour trouver les .shp dans le dossier RGI."""
    if rgi_dir is None:
        rgi_dir = RGI_DIR
    shp_paths = []
    for root, dirs, files in os.walk(rgi_dir):
        for f in files:
            if f.endswith(".shp"):
                shp_paths.append(os.path.join(root, f))
    return shp_paths


def load_and_clip_rgi(shp_path, bbox_l93):
    """Charge un shapefile RGI, reprojette L93 et clip sur la bbox."""
    gdf = gpd.read_file(shp_path)
    if gdf.crs and gdf.crs.to_epsg() != 2154:
        gdf = gdf.to_crs(CRS_L93)
    # clip spatial
    gdf = gdf.cx[bbox_l93["xmin"]:bbox_l93["xmax"],
                  bbox_l93["ymin"]:bbox_l93["ymax"]]
    logger.info("RGI %s: %d glaciers dans la bbox", os.path.basename(shp_path), len(gdf))
    return gdf


def rasterize_glaciers(gdf, transform, shape):
    """Rasterise les polygones glacier -> masque bool."""
    if len(gdf) == 0:
        return np.zeros(shape, dtype=bool)

    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    mask = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
    )
    n_px = np.sum(mask == 1)
    logger.info("glacier mask: %d px (%.1f%%)", n_px, n_px / mask.size * 100)
    return mask.astype(bool)


def get_glacier_mask(bbox_l93, transform, shape):
    """Pipeline complet: find shp -> load/clip -> rasterize.
    Retourne None si pas de data RGI dispo."""
    try:
        shp_files = find_rgi_shapefiles()
        if not shp_files:
            logger.warning("pas de shapefiles RGI dans %s", RGI_DIR)
            return None

        # prend le premier shp (region 11 normalement)
        gdf = load_and_clip_rgi(shp_files[0], bbox_l93)
        if len(gdf) == 0:
            logger.info("aucun glacier dans la bbox")
            return None

        return rasterize_glaciers(gdf, transform, shape)

    except Exception as e:
        logger.warning("glacier mask echec (non bloquant): %s", e)
        return None
