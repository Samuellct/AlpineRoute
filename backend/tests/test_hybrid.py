# tests hybrid -- haversine, detour, assemblage, bbox reduite

import pytest
from alpineroute.routing.network import haversine_km, is_detour_excessive
from alpineroute.routing.hybrid import (
    valhalla_to_geojson_feature, assemble_route, reduce_bbox,
)


class TestHaversine:
    def test_chamonix_montenvers(self):
        """Chamonix centre -> Montenvers, env 3-4 km a vol d'oiseau."""
        # Chamonix (45.924, 6.870) -> Montenvers (45.932, 6.917)
        d = haversine_km(45.924, 6.870, 45.932, 6.917)
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
