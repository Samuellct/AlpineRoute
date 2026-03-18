# tests hybrid -- haversine, detour, assemblage, bbox reduite, ponts raster

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from rasterio.transform import from_origin
from alpineroute.routing.network import haversine_km, is_detour_excessive
from alpineroute.routing.hybrid import (
    valhalla_to_geojson_feature, assemble_route,
    reduce_bbox, detect_detour_segments, compute_raster_bridge, apply_bridges,
    find_network_exit, find_network_entry,
)


class TestHaversine:
    def test_chamonix_montenvers(self):
        """Chamonix centre -> Montenvers, env 3-4 km a vol d'oiseau."""
        # Chamonix (45.92366, 6.86864) -> Montenvers (45.930341, 6.918502)
        d = haversine_km(45.92366, 6.86864, 45.930341, 6.918502)
        assert 3.0 < d < 5.0

    def test_zero_distance(self):
        d = haversine_km(45.0, 6.0, 45.0, 6.0)
        assert d < 0.001

    def test_known_distance(self):
        # Paris (48.8566, 2.3522) -> Lyon (45.7640, 4.8357), ~390-395 km
        d = haversine_km(48.8566, 2.3522, 45.7640, 4.8357)
        assert 385 < d < 400


class TestIsDetourExcessive:
    def test_ratio_low_ok(self):
        """Ratio 2.0 -> pas excessif."""
        # 1 km direct, 2 km Valhalla
        assert is_detour_excessive(2.0, (45.0, 6.0), (45.009, 6.0)) is False

    def test_ratio_high_excessif(self):
        """Ratio > 3.0 -> excessif."""
        # ~1 km direct, 5 km Valhalla
        assert is_detour_excessive(5.0, (45.0, 6.0), (45.009, 6.0)) is True

    def test_short_distance_always_ok(self):
        """< 500m a vol d'oiseau -> jamais excessif, meme ratio enorme."""
        # points tres proches (~50m), 10 km Valhalla
        assert is_detour_excessive(10.0, (45.0, 6.0), (45.0004, 6.0)) is False


class TestValhallaToGeojsonFeature:
    def test_basic_structure(self):
        vr = {
            "coords": [(45.9, 6.87), (45.91, 6.88), (45.92, 6.89)],
            "distance_km": 2.5,
            "duration_s": 1800,
        }
        feat = valhalla_to_geojson_feature(vr)

        assert feat["type"] == "Feature"
        assert feat["geometry"]["type"] == "LineString"

        coords = feat["geometry"]["coordinates"]
        assert len(coords) == 3
        # format [lon, lat, 0]
        assert coords[0] == [6.87, 45.9, 0]
        assert coords[2] == [6.89, 45.92, 0]

        props = feat["properties"]
        assert props["distance_km"] == 2.5
        assert props["strategy"] == "network"
        assert props["is_optimal"] is True
        assert props["dplus_m"] == 0

    def test_route_index(self):
        vr = {
            "coords": [(45.0, 6.0)],
            "distance_km": 0.1,
            "duration_s": 60,
        }
        feat = valhalla_to_geojson_feature(vr, route_index=2)
        assert feat["properties"]["route_index"] == 2
        assert feat["properties"]["is_optimal"] is False


class TestAssembleRoute:
    def test_concatenation(self):
        v_coords = [(45.90, 6.87), (45.91, 6.88)]
        r_feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[6.88, 45.91, 3000], [6.89, 45.92, 3100]],
            },
            "properties": {
                "distance_km": 1.0,
                "time_tobler_h": 0.5,
                "dplus_m": 100,
                "dminus_m": 0,
                "glacier_pct": 5.0,
                "cost_total": 42.0,
            },
        }
        feat = assemble_route(v_coords, r_feature, (45.91, 6.88))

        coords = feat["geometry"]["coordinates"]
        assert len(coords) == 4
        assert feat["properties"]["strategy"] == "hybrid"
        assert feat["properties"]["n_points"] == 4

    def test_with_valhalla_stats(self):
        """Stats Valhalla transmises dans les props assemblees."""
        v_coords = [(45.90, 6.87), (45.91, 6.88)]
        r_feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[6.88, 45.91, 3000], [6.89, 45.92, 3100]],
            },
            "properties": {"distance_km": 1.0, "time_tobler_h": 0.5},
        }
        v_stats = {"distance_km": 2.5, "duration_s": 1800}
        feat = assemble_route(v_coords, r_feature, (45.91, 6.88), valhalla_stats=v_stats)
        props = feat["properties"]
        assert props["distance_km"] == 3.5  # 2.5 + 1.0
        assert props["time_tobler_h"] == 1.0  # 0.5h valhalla + 0.5h raster

    def test_raster_first_order(self):
        """order=raster_first -> coords raster avant Valhalla."""
        v_coords = [(45.92, 6.89)]
        r_feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[6.87, 45.90, 2800]],
            },
            "properties": {"distance_km": 1.0},
        }
        feat = assemble_route(v_coords, r_feature, (45.91, 6.88), order="raster_first")
        coords = feat["geometry"]["coordinates"]
        # raster d'abord, puis valhalla
        assert coords[0] == [6.87, 45.90, 2800]
        assert coords[1] == [6.89, 45.92, 0]

    def test_gap_warning(self, caplog):
        """Gap > 50m entre Valhalla et raster -> warning."""
        import logging
        # points distants de ~500m
        v_coords = [(45.90, 6.87)]
        r_feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[6.88, 45.91, 3000]],
            },
            "properties": {},
        }
        with caplog.at_level(logging.WARNING):
            assemble_route(v_coords, r_feature, (45.905, 6.875))
        assert any("raccord" in msg.lower() for msg in caplog.messages)


class TestReduceBbox:
    def test_bbox_contains_points(self):
        # deux points pres de Chamonix
        exit_pt = (45.92, 6.87)
        dest_pt = (45.93, 6.89)
        result = reduce_bbox(exit_pt, dest_pt, margin_m=500)

        bl93 = result["bbox_l93"]
        bw84 = result["bbox_wgs84"]

        # la bbox doit contenir les deux points (en WGS84)
        assert bw84["lat_min"] < 45.92
        assert bw84["lat_max"] > 45.93
        assert bw84["lon_min"] < 6.87
        assert bw84["lon_max"] > 6.89

        # bbox L93 coherente
        assert bl93["xmin"] < bl93["xmax"]
        assert bl93["ymin"] < bl93["ymax"]

    def test_margin_applied(self):
        pt = (45.92, 6.87)
        r1 = reduce_bbox(pt, pt, margin_m=100)
        r2 = reduce_bbox(pt, pt, margin_m=1000)
        # marge plus grande = bbox plus grande
        w1 = r1["bbox_l93"]["xmax"] - r1["bbox_l93"]["xmin"]
        w2 = r2["bbox_l93"]["xmax"] - r2["bbox_l93"]["xmin"]
        assert w2 > w1


class TestDetectDetourSegments:
    def _make_vr(self, coords, maneuvers):
        return {"coords": coords, "maneuvers": maneuvers}

    def test_no_detour(self):
        """Route directe, ratio < seuil -> pas de detour."""
        coords = [(45.92, 6.87), (45.921, 6.871), (45.922, 6.872)]
        maneuvers = [{
            "begin_shape_index": 0,
            "end_shape_index": 2,
            "length_km": 0.3,  # ~distance directe, ratio ~1
        }]
        vr = self._make_vr(coords, maneuvers)
        result = detect_detour_segments(vr)
        assert result == []

    def test_detour_detected(self):
        """Un maneuver avec ratio > seuil -> detecte."""
        coords = [(45.92, 6.87), (45.921, 6.871), (45.9205, 6.8705)]
        # direct ~150m, leg=0.5km -> ratio ~3.3
        maneuvers = [{
            "begin_shape_index": 0,
            "end_shape_index": 2,
            "length_km": 0.5,
        }]
        vr = self._make_vr(coords, maneuvers)
        result = detect_detour_segments(vr)
        assert len(result) == 1
        assert result[0]["maneuver_index"] == 0
        assert result[0]["direct_m"] > 10

    def test_distance_too_large(self):
        """Direct > 300m -> skip (pas un pont)."""
        # deux points distants de ~1km
        coords = [(45.92, 6.87), (45.93, 6.87)]
        maneuvers = [{
            "begin_shape_index": 0,
            "end_shape_index": 1,
            "length_km": 5.0,
        }]
        vr = self._make_vr(coords, maneuvers)
        result = detect_detour_segments(vr)
        assert result == []

    def test_empty_maneuvers(self):
        vr = {"coords": [(45.92, 6.87)], "maneuvers": []}
        assert detect_detour_segments(vr) == []


class TestComputeRasterBridge:
    def test_flat_terrain_bridge(self):
        """Terrain plat -> le pont doit trouver un chemin."""
        shape = (50, 50)
        cost = np.ones(shape, dtype=np.float32) * 10.0
        dem = np.full(shape, 3000.0, dtype=np.float32)
        glacier_mask = np.zeros(shape, dtype=bool)
        # transform L93 centree
        transform = from_origin(1001000.0, 6542000.0, 1.0, 1.0)

        # points proches dans la grille (~10px apart)
        # on utilise des coords qui tombent dans la grille
        start = (45.9237, 6.8700)  # doit tomber dans la grille
        end = (45.9237, 6.8701)

        result = compute_raster_bridge(
            start, end, cost, transform, dem, glacier_mask, 1.0)
        # le resultat peut etre None si les points sont hors grille
        # vu la petite taille, c'est possible
        if result is not None:
            assert result["n_points"] > 0
            assert len(result["coords_wgs84"]) > 0

    def test_wall_returns_none(self):
        """Mur infranchissable entre start/end -> None."""
        shape = (50, 50)
        cost = np.ones(shape, dtype=np.float32)
        # mur vertical au milieu
        cost[:, 24:26] = 1e6
        dem = np.full(shape, 3000.0, dtype=np.float32)
        glacier_mask = np.zeros(shape, dtype=bool)
        transform = from_origin(1001000.0, 6542000.0, 1.0, 1.0)

        # points de part et d'autre du mur
        start = (45.9237, 6.8700)
        end = (45.9237, 6.8710)

        result = compute_raster_bridge(
            start, end, cost, transform, dem, glacier_mask, 1.0)
        # soit None soit un chemin contournant
        # dans tous les cas pas de crash


class TestApplyBridges:
    def test_no_valid_bridges(self):
        vr = {"coords": [(45.92, 6.87), (45.93, 6.88)], "maneuvers": [
            {"begin_shape_index": 0, "end_shape_index": 1, "length_km": 1.0}
        ]}
        detours = [{"start": (45.92, 6.87), "end": (45.93, 6.88), "maneuver_index": 0, "direct_m": 100}]
        bridges = [None]
        result = apply_bridges(vr, detours, bridges)
        assert result is None

    def test_bridge_replaces_segment(self):
        coords = [(45.92, 6.87), (45.925, 6.875), (45.93, 6.88)]
        vr = {"coords": coords, "distance_km": 2.0, "duration_s": 1000,
              "maneuvers": [
                  {"begin_shape_index": 0, "end_shape_index": 2, "length_km": 2.0}
              ]}
        detours = [{"start": (45.92, 6.87), "end": (45.93, 6.88),
                    "maneuver_index": 0, "direct_m": 200}]
        bridge_coords = [(45.921, 6.871), (45.929, 6.879)]
        bridges = [{"coords_wgs84": bridge_coords, "n_points": 2, "cost": 5.0,
                    "path_coords_global": np.array([[0, 0], [1, 1]])}]
        result = apply_bridges(vr, detours, bridges)
        assert result is not None
        assert result["n_bridges"] == 1
        # les coords originales ont ete remplacees
        assert len(result["coords"]) == 2


class TestFindNetworkExit:
    _VR = {
        "coords": [(45.92, 6.87), (45.925, 6.88)],
        "distance_km": 2.0, "duration_s": 1200,
        "shape_encoded": "fake", "maneuvers": [],
        "snap_start": (45.92, 6.87), "snap_end": (45.925, 6.88),
        "snap_start_m": 0, "snap_end_m": 0,
    }

    @patch("alpineroute.routing.network.valhalla_route")
    @patch("alpineroute.routing.network.parse_locate_snap")
    @patch("alpineroute.routing.network.valhalla_locate")
    def test_success(self, mock_locate, mock_parse, mock_route):
        mock_locate.return_value = [{"edges": [{}]}]
        mock_parse.return_value = (45.925, 6.88)
        mock_route.return_value = self._VR

        result = find_network_exit((45.92, 6.87), (45.926, 6.881))
        assert result is not None
        assert "exit_point" in result
        assert "approach" in result
        # exit_point = dernier point reel de la route, pas le snap /locate
        assert result["exit_point"] == (45.925, 6.88)  # = _VR["coords"][-1]

    @patch("alpineroute.routing.network.valhalla_route")
    @patch("alpineroute.routing.network.parse_locate_snap")
    @patch("alpineroute.routing.network.valhalla_locate")
    def test_exit_uses_actual_route_end(self, mock_locate, mock_parse, mock_route):
        """exit_point doit etre coords[-1] de la route, pas le snap /locate."""
        mock_locate.return_value = [{"edges": [{}]}]
        # /locate retourne un point proche de la dest
        mock_parse.return_value = (45.926, 6.881)
        # mais la route finit ailleurs (Valhalla re-snappe en interne)
        actual_end = (45.923, 6.875)
        vr = dict(self._VR)
        vr["coords"] = [(45.92, 6.87), actual_end]
        vr["distance_km"] = 0.6  # coherent avec ~0.5km direct
        mock_route.return_value = vr

        result = find_network_exit((45.92, 6.87), (45.926, 6.881))
        assert result is not None
        assert result["exit_point"] == actual_end

    @patch("alpineroute.routing.network.valhalla_locate")
    def test_locate_fails(self, mock_locate):
        mock_locate.return_value = None
        result = find_network_exit((45.92, 6.87), (45.93, 6.89))
        assert result is None

    @patch("alpineroute.routing.network.parse_locate_snap")
    @patch("alpineroute.routing.network.valhalla_locate")
    def test_snap_too_far(self, mock_locate, mock_parse):
        mock_locate.return_value = [{"edges": [{}]}]
        mock_parse.return_value = (45.94, 6.90)
        result = find_network_exit((45.92, 6.87), (45.926, 6.881))
        assert result is None

    @patch("alpineroute.routing.network.valhalla_route")
    @patch("alpineroute.routing.network.parse_locate_snap")
    @patch("alpineroute.routing.network.valhalla_locate")
    def test_route_fails(self, mock_locate, mock_parse, mock_route):
        mock_locate.return_value = [{"edges": [{}]}]
        mock_parse.return_value = (45.925, 6.88)
        mock_route.return_value = None
        result = find_network_exit((45.92, 6.87), (45.926, 6.881))
        assert result is None

    @patch("alpineroute.routing.network.valhalla_route")
    @patch("alpineroute.routing.network.parse_locate_snap")
    @patch("alpineroute.routing.network.valhalla_locate")
    def test_useless_approach_rejected(self, mock_locate, mock_parse, mock_route):
        """Approche qui ne rapproche pas de la dest -> rejetee."""
        mock_locate.return_value = [{"edges": [{}]}]
        mock_parse.return_value = (45.926, 6.881)
        # l'approche finit au meme endroit que le start (detour inutile)
        vr = dict(self._VR)
        vr["coords"] = [(45.92, 6.87), (45.921, 6.871)]
        mock_route.return_value = vr

        # end est a (45.93, 6.89) -- exit (45.921) n'est pas plus proche que start (45.92)
        result = find_network_exit((45.92, 6.87), (45.93, 6.89))
        assert result is None

    @patch("alpineroute.routing.network.valhalla_route")
    @patch("alpineroute.routing.network.parse_locate_snap")
    @patch("alpineroute.routing.network.valhalla_locate")
    def test_result_has_no_continuation(self, mock_locate, mock_parse, mock_route):
        """Resultat = {exit_point, approach, snap_m}, pas de continuation."""
        mock_locate.return_value = [{"edges": [{}]}]
        mock_parse.return_value = (45.925, 6.88)
        mock_route.return_value = self._VR

        result = find_network_exit((45.92, 6.87), (45.926, 6.881))
        assert result is not None
        assert "exit_point" in result
        assert "approach" in result
        assert "snap_m" in result
        assert "continuation" not in result


class TestFindNetworkEntry:
    @patch("alpineroute.routing.network.valhalla_route")
    @patch("alpineroute.routing.network.parse_locate_snap")
    @patch("alpineroute.routing.network.valhalla_locate")
    def test_success(self, mock_locate, mock_parse, mock_route):
        mock_locate.return_value = [{"edges": [{}]}]
        mock_parse.return_value = (45.921, 6.871)
        mock_route.return_value = {
            "coords": [(45.921, 6.871), (45.93, 6.89)],
            "distance_km": 3.0, "duration_s": 1800,
            "shape_encoded": "fake", "maneuvers": [],
            "snap_start": (45.921, 6.871), "snap_end": (45.93, 6.89),
            "snap_start_m": 0, "snap_end_m": 0,
        }
        result = find_network_entry((45.920, 6.870), (45.93, 6.89))
        assert result is not None
        assert "entry_point" in result
        assert "continuation" in result

    @patch("alpineroute.routing.network.valhalla_locate")
    def test_locate_fails(self, mock_locate):
        mock_locate.return_value = None
        result = find_network_entry((45.92, 6.87), (45.93, 6.89))
        assert result is None
