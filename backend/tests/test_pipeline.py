# tests d'integration pipeline
# mini-DEM synthetique, pas de reseau

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from rasterio.transform import from_origin


def _make_mini_dem(size=50, alt_range=(2800, 3200)):
    """DEM synthetique avec un gradient N->S."""
    rng = np.random.default_rng(42)
    gradient = np.linspace(alt_range[0], alt_range[1], size).reshape(-1, 1)
    dem = np.broadcast_to(gradient, (size, size)).copy().astype(np.float32)
    dem += rng.normal(0, 5, dem.shape).astype(np.float32)
    return dem


def _make_steep_dem(size=50):
    """DEM avec forte denivelee pour trigger le warning isotrope."""
    gradient = np.linspace(2000, 3500, size).reshape(-1, 1)
    dem = np.broadcast_to(gradient, (size, size)).copy().astype(np.float32)
    return dem


# transform: ~1m resolution pres de Chamonix
_TRANSFORM = from_origin(1001000.0, 6542000.0, 1.0, 1.0)

# bbox coherente avec le transform et la grille 50x50
_BBOX_L93 = {
    "xmin": 1001000.0, "ymin": 6541950.0,
    "xmax": 1001050.0, "ymax": 6542000.0,
}
_BBOX_WGS84 = {
    "lon_min": 6.86, "lat_min": 45.86,
    "lon_max": 6.87, "lat_max": 45.87,
}


def _fake_req(**kwargs):
    """Construit un faux RouteRequest."""
    from alpineroute.api.models import RouteRequest
    defaults = {
        "start_lat": 45.865, "start_lon": 6.865,
        "end_lat": 45.868, "end_lon": 6.868,
        "resolution": 1.0, "month": 7,
        "acclimatized": True, "n_alternatives": 0,
        "anisotropic": False, "save": False,
    }
    defaults.update(kwargs)
    return RouteRequest(**defaults)


def _mock_pipeline_deps(dem):
    """Ecrit le DEM dans un tif temporaire."""
    import tempfile, rasterio
    from rasterio.crs import CRS

    tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": dem.shape[1], "height": dem.shape[0],
        "count": 1, "crs": CRS.from_epsg(2154),
        "transform": _TRANSFORM,
    }
    with rasterio.open(tmp.name, "w", **profile) as dst:
        dst.write(dem, 1)
    return tmp.name


def _fake_wgs84_to_pixel_start_end(start_rc, end_rc):
    """Renvoie un mock de wgs84_to_pixel qui mappe les premiers/derniers appels."""
    call_count = [0]
    def _mock(lat, lon, transform, shape):
        r, c = [start_rc, end_rc][min(call_count[0], 1)]
        call_count[0] += 1
        return r, c, 1001000.0 + c, 6542000.0 - r
    return _mock


class TestPipelineIntegration:
    """Tests pipeline complet avec mocks (pas de DL reseau)."""

    @pytest.fixture
    def mini_dem_path(self):
        dem = _make_mini_dem()
        return _mock_pipeline_deps(dem)

    @pytest.fixture
    def steep_dem_path(self):
        dem = _make_steep_dem()
        return _mock_pipeline_deps(dem)

    @patch("alpineroute.pipeline.get_glacier_mask", return_value=None)
    @patch("alpineroute.pipeline.get_landcover_cost", return_value=None)
    @patch("alpineroute.pipeline.get_dem")
    @patch("alpineroute.pipeline.list_zones", return_value=[])
    @patch("alpineroute.pipeline.compute_bbox")
    @patch("alpineroute.pipeline.wgs84_to_pixel")
    def test_pipeline_basic(self, mock_w2p, mock_bbox, mock_zones,
                            mock_get_dem, mock_lc, mock_gl, mini_dem_path):
        """Le pipeline retourne un resultat valide."""
        mock_get_dem.return_value = mini_dem_path
        mock_bbox.return_value = {"bbox_l93": _BBOX_L93, "bbox_wgs84": _BBOX_WGS84}
        mock_w2p.side_effect = _fake_wgs84_to_pixel_start_end((5, 5), (45, 45))

        from alpineroute.pipeline import run_pipeline
        req = _fake_req()
        result = run_pipeline(req)

        assert result["status"] == "ok"
        assert result["route"] is not None
        props = result["route"]["properties"]
        assert props["distance_km"] > 0
        assert props["dplus_m"] >= 0
        assert "computation_time_s" in result

    @patch("alpineroute.pipeline.get_glacier_mask", return_value=None)
    @patch("alpineroute.pipeline.get_landcover_cost", return_value=None)
    @patch("alpineroute.pipeline.get_dem")
    @patch("alpineroute.pipeline.list_zones", return_value=[])
    @patch("alpineroute.pipeline.compute_bbox")
    @patch("alpineroute.pipeline.wgs84_to_pixel")
    def test_isotropic_warning_high_dplus(self, mock_w2p, mock_bbox,
                                          mock_zones, mock_get_dem,
                                          mock_lc, mock_gl, steep_dem_path):
        """Forte denivelee en mode isotrope -> warning present."""
        mock_get_dem.return_value = steep_dem_path
        mock_bbox.return_value = {"bbox_l93": _BBOX_L93, "bbox_wgs84": _BBOX_WGS84}
        mock_w2p.side_effect = _fake_wgs84_to_pixel_start_end((5, 5), (45, 45))

        from alpineroute.pipeline import run_pipeline
        req = _fake_req(anisotropic=False)
        result = run_pipeline(req)

        assert result["status"] == "ok"
        route = result["route"]
        dplus = route["properties"]["dplus_m"]
        dminus = route["properties"]["dminus_m"]
        # 2000 -> 3500m sur 50px, D+ > 500 garanti
        if max(dplus, dminus) > 500:
            assert "warnings" in result
            assert len(result["warnings"]) > 0
            assert "isotrope" in result["warnings"][0].lower()

    @patch("alpineroute.pipeline.get_glacier_mask", return_value=None)
    @patch("alpineroute.pipeline.get_landcover_cost", return_value=None)
    @patch("alpineroute.pipeline.get_dem")
    @patch("alpineroute.pipeline.list_zones", return_value=[])
    @patch("alpineroute.pipeline.compute_bbox")
    @patch("alpineroute.pipeline.wgs84_to_pixel")
    def test_anisotropic_mode(self, mock_w2p, mock_bbox, mock_zones,
                               mock_get_dem, mock_lc, mock_gl, mini_dem_path):
        """Mode anisotrope fonctionne aussi."""
        mock_get_dem.return_value = mini_dem_path
        mock_bbox.return_value = {"bbox_l93": _BBOX_L93, "bbox_wgs84": _BBOX_WGS84}
        mock_w2p.side_effect = _fake_wgs84_to_pixel_start_end((5, 5), (45, 45))

        from alpineroute.pipeline import run_pipeline
        req = _fake_req(anisotropic=True)
        result = run_pipeline(req)

        assert result["status"] == "ok"
        assert result["route"] is not None
        # pas de warning en mode aniso
        assert "warnings" not in result or len(result.get("warnings", [])) == 0
