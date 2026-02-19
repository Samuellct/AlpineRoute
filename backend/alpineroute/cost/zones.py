# rasterisation des zones utilisateur sur la grille de cout
# chaque zone = polygone GeoJSON + multiplicateur

import logging
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import shape
from pyproj import Transformer

from alpineroute.config import CRS_WGS84, CRS_L93

logger = logging.getLogger(__name__)

# transformer WGS84 -> L93 (cache module-level)
_wgs_to_l93 = Transformer.from_crs(CRS_WGS84, CRS_L93, always_xy=True)


def _reproject_geojson(geojson_dict):
    """Reprojette un GeoJSON WGS84 -> L93 via shapely + pyproj."""
    geom = shape(geojson_dict)
    from shapely.ops import transform as shapely_transform
    return shapely_transform(_wgs_to_l93.transform, geom)


def rasterize_user_zones(zones, transform, grid_shape):
    """Rasterise les zones utilisateur en grille de multiplicateurs.
    zones: liste de dicts avec 'geojson' (WGS84) et 'cost_multiplier'.
    Retourne array float32 (1.0 = neutre) ou None si pas de zones."""
    if not zones:
        return None

    # on part de 1.0 partout (neutre)
    result = np.ones(grid_shape, dtype=np.float32)

    for z in zones:
        try:
            geom_l93 = _reproject_geojson(z["geojson"])
            mult = float(z.get("cost_multiplier", 100.0))

            # rasterise: 1 dans la zone, 0 dehors
            mask = rasterize(
                [(geom_l93, 1)],
                out_shape=grid_shape,
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )
            # les zones se multiplient entre elles
            result[mask == 1] *= mult
            logger.info("zone '%s' rasterisee (mult=%.0f, %d px)",
                        z.get("name", "?"), mult, int(mask.sum()))
        except Exception as e:
            # zone mal formee -> on skip, pas de crash
            logger.warning("skip zone '%s': %s", z.get("name", "?"), e)

    return result
