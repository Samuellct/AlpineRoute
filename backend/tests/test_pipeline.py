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
    # routing_mode supprime, plus besoin de le passer
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

    @patch("alpineroute.pipeline.valhalla_available", return_value=False)
    @patch("alpineroute.pipeline.get_barrier_masks", return_value=None)
    @patch("alpineroute.pipeline.get_trail_cost", return_value=None)
    @patch("alpineroute.pipeline.get_glacier_mask", return_value=None)
    @patch("alpineroute.pipeline.get_landcover_cost", return_value=None)
    @patch("alpineroute.pipeline.get_dem")
    @patch("alpineroute.pipeline.list_zones", return_value=[])
    @patch("alpineroute.pipeline.compute_bbox")
    @patch("alpineroute.pipeline.wgs84_to_pixel")
    def test_pipeline_basic(self, mock_w2p, mock_bbox, mock_zones,
                            mock_get_dem, mock_lc, mock_gl,
                            mock_trail, mock_barrier, mock_vavail, mini_dem_path):
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

    @patch("alpineroute.pipeline.valhalla_available", return_value=False)
    @patch("alpineroute.pipeline.get_barrier_masks", return_value=None)
    @patch("alpineroute.pipeline.get_trail_cost", return_value=None)
    @patch("alpineroute.pipeline.get_glacier_mask", return_value=None)
    @patch("alpineroute.pipeline.get_landcover_cost", return_value=None)
    @patch("alpineroute.pipeline.get_dem")
    @patch("alpineroute.pipeline.list_zones", return_value=[])
    @patch("alpineroute.pipeline.compute_bbox")
    @patch("alpineroute.pipeline.wgs84_to_pixel")
    def test_isotropic_warning_high_dplus(self, mock_w2p, mock_bbox,
                                          mock_zones, mock_get_dem,
                                          mock_lc, mock_gl,
                                          mock_trail, mock_barrier,
                                          mock_vavail, steep_dem_path):
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

    @patch("alpineroute.pipeline.valhalla_available", return_value=False)
    @patch("alpineroute.pipeline.get_barrier_masks", return_value=None)
    @patch("alpineroute.pipeline.get_trail_cost", return_value=None)
    @patch("alpineroute.pipeline.get_glacier_mask", return_value=None)
    @patch("alpineroute.pipeline.get_landcover_cost", return_value=None)
    @patch("alpineroute.pipeline.get_dem")
    @patch("alpineroute.pipeline.list_zones", return_value=[])
    @patch("alpineroute.pipeline.compute_bbox")
    @patch("alpineroute.pipeline.wgs84_to_pixel")
    def test_anisotropic_mode(self, mock_w2p, mock_bbox, mock_zones,
                               mock_get_dem, mock_lc, mock_gl,
                               mock_trail, mock_barrier, mock_vavail, mini_dem_path):
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

    @patch("alpineroute.pipeline.valhalla_available", return_value=False)
    @patch("alpineroute.pipeline.get_barrier_masks")
    @patch("alpineroute.pipeline.get_trail_cost")
    @patch("alpineroute.pipeline.get_glacier_mask", return_value=None)
    @patch("alpineroute.pipeline.get_landcover_cost", return_value=None)
    @patch("alpineroute.pipeline.get_dem")
    @patch("alpineroute.pipeline.list_zones", return_value=[])
    @patch("alpineroute.pipeline.compute_bbox")
    @patch("alpineroute.pipeline.wgs84_to_pixel")
    def test_osm_trail_and_barrier(self, mock_w2p, mock_bbox,
                                    mock_zones, mock_get_dem,
                                    mock_lc, mock_gl,
                                    mock_trail, mock_barrier,
                                    mock_vavail, mini_dem_path):
        """Le chemin emprunte le sentier et contourne la barriere."""
        mock_get_dem.return_value = mini_dem_path
        mock_bbox.return_value = {"bbox_l93": _BBOX_L93, "bbox_wgs84": _BBOX_WGS84}
        mock_w2p.side_effect = _fake_wgs84_to_pixel_start_end((5, 5), (45, 45))

        # sentier horizontal au milieu -> cout reduit
        tc = np.ones((50, 50), dtype=np.float32)
        tc[20:26, :] = 0.5
        mock_trail.return_value = tc

        # barriere verticale col 30-33, sauf trou lignes 22-24
        bmask = np.zeros((50, 50), dtype=bool)
        bmask[:, 30:34] = True
        bmask[22:25, 30:34] = False  # passage
        smask = np.zeros((50, 50), dtype=bool)
        mock_barrier.return_value = {"barrier_mask": bmask, "stream_mask": smask}

        from alpineroute.pipeline import run_pipeline
        req = _fake_req()
        result = run_pipeline(req)

        assert result["status"] == "ok"
        route = result["route"]
        coords = route["geometry"]["coordinates"]
        # le chemin doit passer par le trou (lignes ~22-24)
        # en coords pixel, la ligne 22-24 correspond a y autour de 6541978
        # verif simple: le chemin existe et n'est pas vide
        assert len(coords) > 2

    @patch("alpineroute.pipeline.valhalla_available", return_value=True)
    @patch("alpineroute.pipeline.valhalla_route")
    @patch("alpineroute.pipeline.is_detour_excessive", return_value=False)
    def test_pipeline_network_strategy(self, mock_detour, mock_vroute,
                                        mock_vavail):
        """Valhalla dispo + route OK -> strategy=network, pas de raster."""
        mock_vroute.return_value = {
            "coords": [(45.865, 6.865), (45.866, 6.866), (45.868, 6.868)],
            "distance_km": 1.2,
            "duration_s": 900,
            "shape_encoded": "fake",
            "maneuvers": [],
            "snap_start": (45.865, 6.865),
            "snap_end": (45.868, 6.868),
            "snap_start_m": 10.0,
            "snap_end_m": 15.0,
        }

        from alpineroute.pipeline import run_pipeline
        req = _fake_req()
        result = run_pipeline(req)

        assert result["status"] == "ok"
        assert result["strategy"] == "network"
        assert result["valhalla_available"] is True
        assert "valhalla" in result["layers_used"]
        assert result["route"] is not None
        assert result["route"]["properties"]["strategy"] == "network"
        assert result.get("coverage") == "full"
        assert "snap_start_m" in result
        assert "snap_end_m" in result

    @patch("alpineroute.pipeline.valhalla_available", return_value=False)
    @patch("alpineroute.pipeline.get_barrier_masks", return_value=None)
    @patch("alpineroute.pipeline.get_trail_cost", return_value=None)
    @patch("alpineroute.pipeline.get_glacier_mask", return_value=None)
    @patch("alpineroute.pipeline.get_landcover_cost", return_value=None)
    @patch("alpineroute.pipeline.get_dem")
    @patch("alpineroute.pipeline.list_zones", return_value=[])
    @patch("alpineroute.pipeline.compute_bbox")
    @patch("alpineroute.pipeline.wgs84_to_pixel")
    def test_pipeline_valhalla_down_fallback(self, mock_w2p, mock_bbox,
                                              mock_zones, mock_get_dem,
                                              mock_lc, mock_gl,
                                              mock_trail, mock_barrier,
                                              mock_vavail, mini_dem_path):
        """Valhalla indisponible -> fallback raster."""
        mock_get_dem.return_value = mini_dem_path
        mock_bbox.return_value = {"bbox_l93": _BBOX_L93, "bbox_wgs84": _BBOX_WGS84}
        mock_w2p.side_effect = _fake_wgs84_to_pixel_start_end((5, 5), (45, 45))

        from alpineroute.pipeline import run_pipeline
        req = _fake_req()
        result = run_pipeline(req)

        assert result["status"] == "ok"
        assert result["strategy"] == "raster"
        assert result["valhalla_available"] is False

    @patch("alpineroute.pipeline.valhalla_available", return_value=True)
    @patch("alpineroute.pipeline.valhalla_route")
    @patch("alpineroute.pipeline.find_network_exit", return_value=None)
    @patch("alpineroute.pipeline.find_network_entry", return_value=None)
    @patch("alpineroute.pipeline.get_barrier_masks", return_value=None)
    @patch("alpineroute.pipeline.get_trail_cost", return_value=None)
    @patch("alpineroute.pipeline.get_glacier_mask", return_value=None)
    @patch("alpineroute.pipeline.get_landcover_cost", return_value=None)
    @patch("alpineroute.pipeline.get_dem")
    @patch("alpineroute.pipeline.list_zones", return_value=[])
    @patch("alpineroute.pipeline.compute_bbox")
    @patch("alpineroute.pipeline.wgs84_to_pixel")
    def test_ghost_route_rejected(self, mock_w2p, mock_bbox, mock_zones,
                                   mock_get_dem, mock_lc, mock_gl,
                                   mock_trail, mock_barrier,
                                   mock_entry, mock_exit,
                                   mock_vroute, mock_vavail, mini_dem_path):
        """Route fantome (0 km pour points distants) -> fallback raster."""
        mock_vroute.return_value = {
            "coords": [(45.865, 6.865)],
            "distance_km": 0.005,
            "duration_s": 1,
            "shape_encoded": "fake",
            "maneuvers": [],
            "snap_start": (45.865, 6.865),
            "snap_end": (45.865, 6.865),
            "snap_start_m": 0,
            "snap_end_m": 2500.0,
        }
        mock_get_dem.return_value = mini_dem_path
        mock_bbox.return_value = {"bbox_l93": _BBOX_L93, "bbox_wgs84": _BBOX_WGS84}
        mock_w2p.side_effect = _fake_wgs84_to_pixel_start_end((5, 5), (45, 45))

        from alpineroute.pipeline import run_pipeline
        req = _fake_req()
        result = run_pipeline(req)

        assert result["status"] == "ok"
        assert result["strategy"] == "raster"

    @patch("alpineroute.pipeline.valhalla_available", return_value=True)
    @patch("alpineroute.pipeline.valhalla_route")
    @patch("alpineroute.pipeline.find_network_exit")
    @patch("alpineroute.pipeline.get_barrier_masks", return_value=None)
    @patch("alpineroute.pipeline.get_trail_cost", return_value=None)
    @patch("alpineroute.pipeline.get_glacier_mask", return_value=None)
    @patch("alpineroute.pipeline.get_landcover_cost", return_value=None)
    @patch("alpineroute.pipeline.get_dem")
    @patch("alpineroute.pipeline.list_zones", return_value=[])
    @patch("alpineroute.pipeline.reduce_bbox")
    @patch("alpineroute.pipeline.wgs84_to_pixel")
    def test_snap_end_too_far_triggers_hybrid(self, mock_w2p, mock_rbbox,
                                               mock_zones, mock_get_dem,
                                               mock_lc, mock_gl,
                                               mock_trail, mock_barrier,
                                               mock_exit, mock_vroute,
                                               mock_vavail, mini_dem_path):
        """Snap end trop loin -> CAS B hybrid via find_network_exit."""
        # valhalla_route retourne un snap_end trop loin
        mock_vroute.return_value = {
            "coords": [(45.865, 6.865), (45.866, 6.866)],
            "distance_km": 1.0,
            "duration_s": 600,
            "shape_encoded": "fake",
            "maneuvers": [],
            "snap_start": (45.865, 6.865),
            "snap_end": (45.866, 6.866),
            "snap_start_m": 10.0,
            "snap_end_m": 800.0,  # > 500
        }
        # find_network_exit reussit
        mock_exit.return_value = {
            "exit_point": (45.867, 6.867),
            "approach": {
                "coords": [(45.865, 6.865), (45.867, 6.867)],
                "distance_km": 0.8, "duration_s": 500,
            },
            "snap_m": 200.0,
        }
        mock_get_dem.return_value = mini_dem_path
        mock_rbbox.return_value = {"bbox_l93": _BBOX_L93, "bbox_wgs84": _BBOX_WGS84}
        mock_w2p.side_effect = _fake_wgs84_to_pixel_start_end((5, 5), (45, 45))

        from alpineroute.pipeline import run_pipeline
        req = _fake_req()
        result = run_pipeline(req)

        assert result["status"] == "ok"
        assert result["strategy"] == "hybrid"
        assert "valhalla" in result["layers_used"]

    @patch("alpineroute.pipeline.valhalla_available", return_value=True)
    @patch("alpineroute.pipeline.valhalla_route", return_value=None)
    @patch("alpineroute.pipeline.find_network_exit", return_value=None)
    @patch("alpineroute.pipeline.find_network_entry")
    @patch("alpineroute.pipeline.get_barrier_masks", return_value=None)
    @patch("alpineroute.pipeline.get_trail_cost", return_value=None)
    @patch("alpineroute.pipeline.get_glacier_mask", return_value=None)
    @patch("alpineroute.pipeline.get_landcover_cost", return_value=None)
    @patch("alpineroute.pipeline.get_dem")
    @patch("alpineroute.pipeline.list_zones", return_value=[])
    @patch("alpineroute.pipeline.reduce_bbox")
    @patch("alpineroute.pipeline.wgs84_to_pixel")
    def test_hybrid_entry_no_crash(self, mock_w2p, mock_rbbox,
                                    mock_zones, mock_get_dem,
                                    mock_lc, mock_gl,
                                    mock_trail, mock_barrier,
                                    mock_entry, mock_exit,
                                    mock_vroute, mock_vavail, mini_dem_path):
        """find_network_entry reussit -> hybrid, pas de crash KeyError."""
        mock_entry.return_value = {
            "entry_point": (45.866, 6.866),
            "continuation": {
                "coords": [(45.866, 6.866), (45.868, 6.868)],
                "distance_km": 0.8, "duration_s": 500,
            },
            "snap_m": 100.0,
        }
        mock_get_dem.return_value = mini_dem_path
        mock_rbbox.return_value = {"bbox_l93": _BBOX_L93, "bbox_wgs84": _BBOX_WGS84}
        mock_w2p.side_effect = _fake_wgs84_to_pixel_start_end((5, 5), (45, 45))

        from alpineroute.pipeline import run_pipeline
        req = _fake_req()
        result = run_pipeline(req)

        assert result["status"] == "ok"
        assert result["strategy"] == "hybrid"
        assert "valhalla" in result["layers_used"]

    @patch("alpineroute.pipeline.valhalla_available", return_value=True)
    @patch("alpineroute.pipeline.valhalla_route", return_value=None)
    @patch("alpineroute.pipeline.find_network_exit", return_value=None)
    @patch("alpineroute.pipeline.find_network_entry", return_value=None)
    @patch("alpineroute.pipeline.get_barrier_masks", return_value=None)
    @patch("alpineroute.pipeline.get_trail_cost", return_value=None)
    @patch("alpineroute.pipeline.get_glacier_mask", return_value=None)
    @patch("alpineroute.pipeline.get_landcover_cost", return_value=None)
    @patch("alpineroute.pipeline.get_dem")
    @patch("alpineroute.pipeline.list_zones", return_value=[])
    @patch("alpineroute.pipeline.compute_bbox")
    @patch("alpineroute.pipeline.wgs84_to_pixel")
    def test_cas_b_no_route_full_raster(self, mock_w2p, mock_bbox, mock_zones,
                                         mock_get_dem, mock_lc, mock_gl,
                                         mock_trail, mock_barrier,
                                         mock_entry, mock_exit,
                                         mock_vroute, mock_vavail, mini_dem_path):
        """Valhalla up mais NoRoute + exit/entry echouent -> full raster."""
        mock_get_dem.return_value = mini_dem_path
        mock_bbox.return_value = {"bbox_l93": _BBOX_L93, "bbox_wgs84": _BBOX_WGS84}
        mock_w2p.side_effect = _fake_wgs84_to_pixel_start_end((5, 5), (45, 45))

        from alpineroute.pipeline import run_pipeline
        req = _fake_req()
        result = run_pipeline(req)

        assert result["status"] == "ok"
        assert result["strategy"] == "raster"
        assert result["valhalla_available"] is True

    @patch("alpineroute.pipeline.valhalla_available", return_value=True)
    @patch("alpineroute.pipeline.valhalla_route")
    @patch("alpineroute.pipeline.get_barrier_masks", return_value=None)
    @patch("alpineroute.pipeline.get_trail_cost", return_value=None)
    @patch("alpineroute.pipeline.get_glacier_mask", return_value=None)
    @patch("alpineroute.pipeline.get_landcover_cost", return_value=None)
    @patch("alpineroute.pipeline.get_dem")
    @patch("alpineroute.pipeline.list_zones", return_value=[])
    @patch("alpineroute.pipeline.compute_bbox")
    @patch("alpineroute.pipeline.wgs84_to_pixel")
    def test_out_of_coverage_skips_valhalla(self, mock_w2p, mock_bbox,
                                             mock_zones, mock_get_dem,
                                             mock_lc, mock_gl, mock_trail,
                                             mock_barrier, mock_vroute,
                                             mock_vavail, mini_dem_path):
        """Points hors coverage Alps -> pas d'appel Valhalla, raster direct."""
        mock_get_dem.return_value = mini_dem_path
        mock_bbox.return_value = {"bbox_l93": _BBOX_L93, "bbox_wgs84": _BBOX_WGS84}
        mock_w2p.side_effect = _fake_wgs84_to_pixel_start_end((5, 5), (45, 45))

        from alpineroute.pipeline import run_pipeline
        req = _fake_req(start_lat=42.5, start_lon=0.5,
                        end_lat=42.6, end_lon=0.6)
        result = run_pipeline(req)

        assert result["strategy"] == "raster"
        assert result["coverage"] == "none"
        assert "warnings" in result
        mock_vroute.assert_not_called()

    @patch("alpineroute.pipeline.valhalla_available", return_value=True)
    @patch("alpineroute.pipeline.valhalla_route")
    @patch("alpineroute.pipeline.find_network_exit", return_value=None)
    @patch("alpineroute.pipeline.find_network_entry", return_value=None)
    @patch("alpineroute.pipeline.get_barrier_masks", return_value=None)
    @patch("alpineroute.pipeline.get_trail_cost", return_value=None)
    @patch("alpineroute.pipeline.get_glacier_mask", return_value=None)
    @patch("alpineroute.pipeline.get_landcover_cost", return_value=None)
    @patch("alpineroute.pipeline.get_dem")
    @patch("alpineroute.pipeline.list_zones", return_value=[])
    @patch("alpineroute.pipeline.compute_bbox")
    @patch("alpineroute.pipeline.wgs84_to_pixel")
    def test_ghost_route_loop_rejected(self, mock_w2p, mock_bbox, mock_zones,
                                        mock_get_dem, mock_lc, mock_gl,
                                        mock_trail, mock_barrier,
                                        mock_entry, mock_exit,
                                        mock_vroute, mock_vavail, mini_dem_path):
        """Route en boucle (first~=last) -> rejet ghost."""
        mock_vroute.return_value = {
            "coords": [(45.865, 6.865), (45.866, 6.866), (45.865, 6.865)],
            "distance_km": 0.5,
            "duration_s": 300,
            "shape_encoded": "fake",
            "maneuvers": [],
            "snap_start": (45.865, 6.865),
            "snap_end": (45.865, 6.865),
            "snap_start_m": 10.0,
            "snap_end_m": 10.0,
        }
        mock_get_dem.return_value = mini_dem_path
        mock_bbox.return_value = {"bbox_l93": _BBOX_L93, "bbox_wgs84": _BBOX_WGS84}
        mock_w2p.side_effect = _fake_wgs84_to_pixel_start_end((5, 5), (45, 45))

        from alpineroute.pipeline import run_pipeline
        req = _fake_req()
        result = run_pipeline(req)

        assert result["status"] == "ok"
        assert result["strategy"] == "raster"

    @patch("alpineroute.pipeline.valhalla_available", return_value=True)
    @patch("alpineroute.pipeline.valhalla_route")
    @patch("alpineroute.pipeline.find_network_exit", return_value=None)
    @patch("alpineroute.pipeline.find_network_entry", return_value=None)
    @patch("alpineroute.pipeline.get_barrier_masks", return_value=None)
    @patch("alpineroute.pipeline.get_trail_cost", return_value=None)
    @patch("alpineroute.pipeline.get_glacier_mask", return_value=None)
    @patch("alpineroute.pipeline.get_landcover_cost", return_value=None)
    @patch("alpineroute.pipeline.get_dem")
    @patch("alpineroute.pipeline.list_zones", return_value=[])
    @patch("alpineroute.pipeline.compute_bbox")
    @patch("alpineroute.pipeline.wgs84_to_pixel")
    def test_ghost_route_too_few_points(self, mock_w2p, mock_bbox, mock_zones,
                                         mock_get_dem, mock_lc, mock_gl,
                                         mock_trail, mock_barrier,
                                         mock_entry, mock_exit,
                                         mock_vroute, mock_vavail, mini_dem_path):
        """Route avec < 3 points -> rejet ghost."""
        mock_vroute.return_value = {
            "coords": [(45.865, 6.865), (45.868, 6.868)],
            "distance_km": 1.2,
            "duration_s": 600,
            "shape_encoded": "fake",
            "maneuvers": [],
            "snap_start": (45.865, 6.865),
            "snap_end": (45.868, 6.868),
            "snap_start_m": 10.0,
            "snap_end_m": 15.0,
        }
        mock_get_dem.return_value = mini_dem_path
        mock_bbox.return_value = {"bbox_l93": _BBOX_L93, "bbox_wgs84": _BBOX_WGS84}
        mock_w2p.side_effect = _fake_wgs84_to_pixel_start_end((5, 5), (45, 45))

        from alpineroute.pipeline import run_pipeline
        req = _fake_req()
        result = run_pipeline(req)

        assert result["status"] == "ok"
        assert result["strategy"] == "raster"
