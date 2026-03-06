# tests classification + rasterisation trails OSM
import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import LineString
from rasterio.transform import from_origin

from alpineroute.config import CRS_L93, TRAIL_COST_MULTIPLIERS
from alpineroute.cost.trails import classify_trails, rasterize_trail_cost


def _make_trail_gdf(rows):
    """Helper: construit un GDF L93 a partir d'une liste de dicts (tags + coords)."""
    cols = ["geometry", "highway", "surface", "tracktype", "sac_scale", "foot", "access"]
    features = []
    for r in rows:
        coords = r.pop("coords", [(1001010, 6541990), (1001050, 6541990)])
        feat = {"geometry": LineString(coords)}
        for k in cols[1:]:
            feat[k] = r.get(k, "")
        features.append(feat)
    if not features:
        return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=CRS_L93)
    return gpd.GeoDataFrame(features, crs=CRS_L93)


# ---- classification ----

class TestClassifyTrails:
    def test_paved_surface(self):
        gdf = _make_trail_gdf([{"highway": "path", "surface": "asphalt"}])
        result = classify_trails(gdf)
        assert len(result) == 1
        assert result.iloc[0]["trail_class"] == "paved"
        assert result.iloc[0]["trail_cost"] == pytest.approx(0.50)

    def test_sac_scale_t3(self):
        gdf = _make_trail_gdf([{"highway": "path", "sac_scale": "demanding_mountain_hiking"}])
        result = classify_trails(gdf)
        assert result.iloc[0]["trail_class"] == "trail_t3"
        assert result.iloc[0]["trail_cost"] == pytest.approx(0.70)

    def test_foot_no_excluded(self):
        gdf = _make_trail_gdf([{"highway": "path", "foot": "no"}])
        result = classify_trails(gdf)
        assert len(result) == 0

    def test_motorway_filtered(self):
        gdf = _make_trail_gdf([{"highway": "motorway"}])
        result = classify_trails(gdf)
        assert len(result) == 0

    def test_priority_sac_over_highway(self):
        # sac_scale=hiking doit prendre le dessus sur highway=path
        gdf = _make_trail_gdf([{"highway": "path", "sac_scale": "hiking"}])
        result = classify_trails(gdf)
        assert result.iloc[0]["trail_class"] == "trail_t1t2"
        assert result.iloc[0]["trail_cost"] == pytest.approx(0.60)

    def test_track_grades(self):
        gdf = _make_trail_gdf([
            {"highway": "track", "tracktype": "grade1"},
            {"highway": "track", "tracktype": "grade5"},
        ])
        result = classify_trails(gdf)
        classes = result["trail_class"].tolist()
        assert "gravel" in classes
        assert "track_soft" in classes

    def test_default_path(self):
        gdf = _make_trail_gdf([{"highway": "path"}])
        result = classify_trails(gdf)
        assert result.iloc[0]["trail_class"] == "trail_default"
        assert result.iloc[0]["trail_cost"] == pytest.approx(0.60)

    def test_empty_input(self):
        gdf = _make_trail_gdf([])
        result = classify_trails(gdf)
        assert len(result) == 0


# ---- rasterisation ----

class TestRasterizeTrailCost:
    def test_trail_pixels_have_cost(self, fake_transform):
        # sentier horizontal au milieu de la grille
        line = LineString([(1001002, 6541990), (1001018, 6541990)])
        gdf = gpd.GeoDataFrame(
            [{"geometry": line, "trail_class": "trail_default", "trail_cost": 0.60}],
            crs=CRS_L93,
        )
        result = rasterize_trail_cost(gdf, fake_transform, (20, 20))
        # qqpart sur la ligne y a du < 1.0
        assert np.any(result < 1.0)
        assert np.min(result[result < 1.0]) == pytest.approx(0.60)

    def test_off_trail_neutral(self, fake_transform):
        # sentier tout en bas, pixels du haut doivent etre 1.0
        line = LineString([(1001002, 6541982), (1001018, 6541982)])
        gdf = gpd.GeoDataFrame(
            [{"geometry": line, "trail_class": "paved", "trail_cost": 0.50}],
            crs=CRS_L93,
        )
        result = rasterize_trail_cost(gdf, fake_transform, (20, 20))
        # premiere ligne (y=6542000 -> row 0) doit etre 1.0
        assert np.all(result[0, :] == 1.0)

    def test_empty_gdf_all_ones(self, fake_transform):
        gdf = gpd.GeoDataFrame(
            columns=["geometry", "trail_class", "trail_cost"], crs=CRS_L93
        )
        result = rasterize_trail_cost(gdf, fake_transform, (20, 20))
        assert np.all(result == 1.0)
