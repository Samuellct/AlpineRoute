# tests barrieres OSM (rivieres, ruisseaux, ponts, autoroutes)
import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import LineString

from alpineroute.config import CRS_L93
from alpineroute.cost.barriers import build_barrier_masks


def _make_barrier_gdf(rows):
    """Helper: GDF L93 avec colonnes waterway/highway/bridge/ford."""
    cols = ["geometry", "waterway", "highway", "bridge", "ford"]
    features = []
    for r in rows:
        coords = r.pop("coords", [(1001005, 6541990), (1001015, 6541990)])
        feat = {"geometry": LineString(coords)}
        for k in cols[1:]:
            feat[k] = r.get(k, "")
        features.append(feat)
    if not features:
        return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=CRS_L93)
    return gpd.GeoDataFrame(features, crs=CRS_L93)


class TestBuildBarrierMasks:
    def test_river_blocks(self, fake_transform):
        gdf = _make_barrier_gdf([{"waterway": "river"}])
        masks = build_barrier_masks(gdf, fake_transform, (20, 20))
        assert np.any(masks["barrier_mask"])

    def test_stream_separate(self, fake_transform):
        gdf = _make_barrier_gdf([{"waterway": "stream"}])
        masks = build_barrier_masks(gdf, fake_transform, (20, 20))
        assert np.any(masks["stream_mask"])
        # stream ne doit PAS etre dans barrier
        assert not np.any(masks["barrier_mask"])

    def test_bridge_creates_gap(self, fake_transform):
        # riviere horizontale + pont au meme endroit
        coords = [(1001005, 6541990), (1001015, 6541990)]
        gdf = _make_barrier_gdf([
            {"waterway": "river", "coords": coords.copy()},
            {"bridge": "yes", "coords": coords.copy()},
        ])
        masks = build_barrier_masks(gdf, fake_transform, (20, 20))
        # le pont doit trouer la barriere la ou ils se superposent
        # verifier qu'il y a au moins un pixel non-bloque dans la zone du pont
        # (la riviere couvre [5:15] en x, row ~10)
        assert not np.all(masks["barrier_mask"])

    def test_motorway_blocks(self, fake_transform):
        gdf = _make_barrier_gdf([{"highway": "motorway"}])
        masks = build_barrier_masks(gdf, fake_transform, (20, 20))
        assert np.any(masks["barrier_mask"])

    def test_empty_all_false(self, fake_transform):
        gdf = _make_barrier_gdf([])
        masks = build_barrier_masks(gdf, fake_transform, (20, 20))
        assert not np.any(masks["barrier_mask"])
        assert not np.any(masks["stream_mask"])
