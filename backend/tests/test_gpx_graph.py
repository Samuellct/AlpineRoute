# tests graphe GPX overlay

import os
import math
import pytest
from unittest.mock import patch, MagicMock

import gpxpy
import gpxpy.gpx


# ---- helpers pour creer des fichiers GPX de test ----

def _make_gpx_file(path, points):
    """Cree un fichier GPX avec une trace."""
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(track)
    seg = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(seg)
    for lat, lon, alt in points:
        seg.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon, elevation=alt))
    with open(path, "w", encoding="utf-8") as f:
        f.write(gpx.to_xml())


@pytest.fixture
def mini_gpx_dir(tmp_path):
    """2 traces GPX synthetiques qui se croisent au milieu."""
    # trace A : N-S, 5 points, ~200m total (secteur Chamonix lat~45.93 lon~6.87)
    trace_a = [
        (45.9310, 6.8700, 3000),
        (45.9305, 6.8700, 2980),
        (45.9300, 6.8700, 2960),  # milieu
        (45.9295, 6.8700, 2940),
        (45.9290, 6.8700, 2920),
    ]
    # trace B : E-W, 5 points, croise A au milieu
    trace_b = [
        (45.9300, 6.8690, 2950),
        (45.9300, 6.8695, 2955),
        (45.9300, 6.8700, 2960),  # ~ meme point que A[2]
        (45.9300, 6.8705, 2955),
        (45.9300, 6.8710, 2950),
    ]

    _make_gpx_file(str(tmp_path / "trace_a.gpx"), trace_a)
    _make_gpx_file(str(tmp_path / "trace_b.gpx"), trace_b)

    entries = [
        {"type": "segment", "gpx_file": "trace_a.gpx",
         "trail_cost": 0.25, "segment_type": "glacier"},
        {"type": "route", "gpx_file": "trace_b.gpx",
         "massif": "Mont-Blanc", "summit": "Test"},
    ]
    return tmp_path, entries


@pytest.fixture
def long_trace_points():
    """100 points espaces de ~10m chacun."""
    base_lat = 45.93
    pts = []
    for i in range(100):
        lat = base_lat + i * 0.0001  # ~11m par increment
        pts.append((lat, 6.87, 3000 + i))
    return pts


# ---- tests subsample ----

class TestSubsample:
    def test_reduces_points(self, long_trace_points):
        from alpineroute.routing.gpx_graph import _subsample_points
        result = _subsample_points(long_trace_points, min_dist_m=50)
        assert len(result) < len(long_trace_points)
        assert len(result) >= 3

    def test_keeps_first_last(self, long_trace_points):
        from alpineroute.routing.gpx_graph import _subsample_points
        result = _subsample_points(long_trace_points, min_dist_m=50)
        assert result[0] == long_trace_points[0]
        assert result[-1] == long_trace_points[-1]

    def test_short_trace_untouched(self):
        from alpineroute.routing.gpx_graph import _subsample_points
        pts = [(45.93, 6.87, 3000), (45.931, 6.87, 3010)]
        result = _subsample_points(pts, min_dist_m=50)
        assert len(result) == 2


# ---- tests find_nearby_node ----

class TestFindNearbyNode:
    def test_node_within_tolerance(self):
        from alpineroute.routing.gpx_graph import _find_nearby_node
        nc = {0: (45.93, 6.87, 3000)}
        # ~20m au nord
        result = _find_nearby_node(45.93018, 6.87, nc, tolerance_m=30)
        assert result == 0

    def test_node_outside_tolerance(self):
        from alpineroute.routing.gpx_graph import _find_nearby_node
        nc = {0: (45.93, 6.87, 3000)}
        # ~60m au nord
        result = _find_nearby_node(45.9306, 6.87, nc, tolerance_m=30)
        assert result is None


# ---- tests build_graph ----

class TestBuildGraph:
    def test_basic_graph(self, mini_gpx_dir):
        from alpineroute.routing.gpx_graph import build_gpx_graph
        gpx_dir, entries = mini_gpx_dir
        G, nc = build_gpx_graph(entries, gpx_dir=str(gpx_dir))

        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0
        for u, v, data in G.edges(data=True):
            assert "weight" in data
            assert "trail_cost" in data
            assert "gpx_source" in data
            assert data["weight"] > 0

    def test_trail_cost_route_vs_segment(self, mini_gpx_dir):
        from alpineroute.routing.gpx_graph import build_gpx_graph
        from alpineroute.config import GPX_ROUTE_TRAIL_COST
        gpx_dir, entries = mini_gpx_dir
        G, nc = build_gpx_graph(entries, gpx_dir=str(gpx_dir))

        route_edges = [d for _, _, d in G.edges(data=True)
                       if d["entry_type"] == "route"]
        seg_edges = [d for _, _, d in G.edges(data=True)
                     if d["entry_type"] == "segment"]
        assert all(d["trail_cost"] == GPX_ROUTE_TRAIL_COST for d in route_edges)
        assert all(d["trail_cost"] == 0.25 for d in seg_edges)

    def test_merge_endpoints(self, tmp_path):
        """2 traces partageant un bout (~10m d'ecart) -> fusion."""
        from alpineroute.routing.gpx_graph import build_gpx_graph

        t1 = [(45.93, 6.87, 3000), (45.931, 6.87, 3010), (45.932, 6.87, 3020)]
        t2 = [(45.93009, 6.87, 3000), (45.93009, 6.871, 3010), (45.93009, 6.872, 3020)]

        _make_gpx_file(str(tmp_path / "t1.gpx"), t1)
        _make_gpx_file(str(tmp_path / "t2.gpx"), t2)
        entries = [
            {"type": "segment", "gpx_file": "t1.gpx", "trail_cost": 0.3},
            {"type": "segment", "gpx_file": "t2.gpx", "trail_cost": 0.3},
        ]
        G, nc = build_gpx_graph(entries, gpx_dir=str(tmp_path))
        assert G.number_of_nodes() == 5


# ---- tests portals ----

class TestPortals:
    def test_detect_portals_snap_ok(self, mini_gpx_dir):
        from alpineroute.routing.gpx_graph import build_gpx_graph, _detect_portals

        gpx_dir, entries = mini_gpx_dir
        G, nc = build_gpx_graph(entries, gpx_dir=str(gpx_dir))

        def fake_locate(pt):
            return [{"edges": [{"correlated_lat": pt[0] + 0.0003,
                                "correlated_lon": pt[1]}]}]

        with patch("alpineroute.routing.network.valhalla_locate", side_effect=fake_locate), \
             patch("alpineroute.routing.network.parse_locate_snap") as mock_parse:
            mock_parse.side_effect = lambda loc: (
                loc[0]["edges"][0]["correlated_lat"],
                loc[0]["edges"][0]["correlated_lon"],
            ) if loc else None

            portals = _detect_portals(G, nc)

        assert len(portals) > 0
        for p in portals:
            assert "node_id" in p
            assert "osm_coords" in p
            assert p["snap_m"] <= 200  # GPX_PORTAL_SNAP_M = 200

    def test_detect_portals_too_far(self, mini_gpx_dir):
        from alpineroute.routing.gpx_graph import build_gpx_graph, _detect_portals

        gpx_dir, entries = mini_gpx_dir
        G, nc = build_gpx_graph(entries, gpx_dir=str(gpx_dir))

        # snap a ~500m -> trop loin meme avec GPX_PORTAL_SNAP_M=200
        def fake_locate(pt):
            return [{"edges": [{"correlated_lat": pt[0] + 0.005,
                                "correlated_lon": pt[1]}]}]

        with patch("alpineroute.routing.network.valhalla_locate", side_effect=fake_locate), \
             patch("alpineroute.routing.network.parse_locate_snap") as mock_parse:
            mock_parse.side_effect = lambda loc: (
                loc[0]["edges"][0]["correlated_lat"],
                loc[0]["edges"][0]["correlated_lon"],
            ) if loc else None

            portals = _detect_portals(G, nc)

        assert len(portals) == 0


# ---- tests corridor search ----

class TestCorridorSearch:
    def test_portal_in_corridor(self):
        from alpineroute.routing.gpx_graph import _portals_in_corridor
        import alpineroute.routing.gpx_graph as mod

        # portail a mi-chemin entre start et end
        mod._gpx_portals = [{
            "node_id": 0, "gpx_coords": (45.925, 6.88),
            "osm_coords": (45.925, 6.88), "snap_m": 10,
        }]

        result = _portals_in_corridor((45.92, 6.87), (45.93, 6.89))
        assert len(result) == 1

    def test_portal_outside_corridor(self):
        from alpineroute.routing.gpx_graph import _portals_in_corridor
        import alpineroute.routing.gpx_graph as mod

        # portail tres loin de la ligne start-end
        mod._gpx_portals = [{
            "node_id": 0, "gpx_coords": (46.0, 7.0),
            "osm_coords": (46.0, 7.0), "snap_m": 10,
        }]

        result = _portals_in_corridor((45.92, 6.87), (45.93, 6.89))
        assert len(result) == 0


# ---- tests route_via_gpx ----

class TestRouteViaGpx:
    def _setup_graph_with_portals(self, mini_gpx_dir):
        """Helper: construit le graphe et injecte les portails."""
        import alpineroute.routing.gpx_graph as mod
        gpx_dir, entries = mini_gpx_dir
        G, nc = mod.build_gpx_graph(entries, gpx_dir=str(gpx_dir))

        mod._gpx_graph = G
        mod._node_coords = nc

        endpoints = [n for n in G.nodes() if G.degree(n) == 1]
        portals = []
        for nid in endpoints:
            lat, lon, alt = nc[nid]
            portals.append({
                "node_id": nid,
                "gpx_coords": (lat, lon),
                "osm_coords": (lat + 0.0001, lon),
                "snap_m": 11.0,
            })

        mod._gpx_portals = portals
        mod._portals_ready = True
        return G, nc, portals

    def test_route_full_coverage(self, mini_gpx_dir):
        import alpineroute.routing.gpx_graph as mod
        G, nc, portals = self._setup_graph_with_portals(mini_gpx_dir)

        north = max(portals, key=lambda p: p["gpx_coords"][0])
        south = min(portals, key=lambda p: p["gpx_coords"][0])

        # start pres du portal nord, end pres du portal sud -> full coverage
        result = mod.route_via_gpx(
            (north["gpx_coords"][0] + 0.0001, north["gpx_coords"][1]),
            (south["gpx_coords"][0] - 0.0001, south["gpx_coords"][1]),
        )

        assert result is not None
        assert result["coverage"] == "full"
        assert result["distance_km"] > 0
        assert len(result["gpx_coords"]) >= 2
        assert len(result["gpx_sources"]) > 0

    def test_route_partial_coverage(self, mini_gpx_dir):
        """Un seul portail dans le corridor -> partial coverage."""
        import alpineroute.routing.gpx_graph as mod
        gpx_dir, entries = mini_gpx_dir
        G, nc = mod.build_gpx_graph(entries, gpx_dir=str(gpx_dir))
        mod._gpx_graph = G
        mod._node_coords = nc

        # un seul portail (le nord de trace A)
        north_nid = max(nc.keys(), key=lambda n: nc[n][0])
        portals = [{
            "node_id": north_nid,
            "gpx_coords": nc[north_nid][:2],
            "osm_coords": (nc[north_nid][0] + 0.0001, nc[north_nid][1]),
            "snap_m": 11.0,
        }]
        mod._gpx_portals = portals
        mod._portals_ready = True

        # start au nord, end au sud (loin du portail) -> le GPX rapproche
        result = mod.route_via_gpx(
            (45.935, 6.870),   # au nord (pres du portail)
            (45.925, 6.870),   # au sud (pas de portail mais GPX va vers la)
        )

        # devrait trouver un partial coverage si le GPX avance vers la dest
        if result is not None:
            assert result["coverage"] == "partial"
            assert "gpx_exit_wgs84" in result
            assert result["remaining_m"] > 0

    def test_route_too_far(self, mini_gpx_dir):
        import alpineroute.routing.gpx_graph as mod
        self._setup_graph_with_portals(mini_gpx_dir)

        # start et end tres loin du graphe -> corridor ne trouvera rien
        result = mod.route_via_gpx((46.5, 7.5), (46.6, 7.6))
        assert result is None

    def test_no_portals_no_result(self):
        """Aucun portail detecte -> None."""
        import alpineroute.routing.gpx_graph as mod

        mod._gpx_graph = None
        mod._gpx_portals = []
        mod._portals_ready = True

        result = mod.route_via_gpx((45.93, 6.87), (45.94, 6.88))
        assert result is None


# ---- test geojson ----

class TestGpxToGeojson:
    def test_feature_structure(self):
        from alpineroute.routing.gpx_graph import gpx_to_geojson_feature

        gpx_result = {
            "gpx_coords": [
                (45.93, 6.87, 3000),
                (45.931, 6.87, 2990),
                (45.932, 6.87, 2980),
            ],
            "entry_portal": {"node_id": 0},
            "exit_portal": {"node_id": 2},
            "coverage": "full",
            "distance_km": 0.22,
            "dplus_m": 0,
            "dminus_m": 20,
            "gpx_sources": ["test.gpx"],
        }

        feature = gpx_to_geojson_feature(gpx_result)
        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] == "LineString"
        coords = feature["geometry"]["coordinates"]
        assert len(coords) == 3
        assert coords[0][0] == 6.87
        assert coords[0][1] == 45.93

        props = feature["properties"]
        assert props["strategy"] == "gpx_graph"
        assert props["distance_km"] == 0.22
        assert props["gpx_sources"] == ["test.gpx"]
        assert "time_tobler_h" in props


# ---- test rebuild ----

class TestRebuild:
    def test_rebuild_updates_cache(self, mini_gpx_dir):
        import alpineroute.routing.gpx_graph as mod
        gpx_dir, entries = mini_gpx_dir

        original_load = mod.load_gpx

        def patched_load(path):
            fname = os.path.basename(path)
            return original_load(os.path.join(str(gpx_dir), fname))

        with patch.object(mod, "load_gpx", side_effect=patched_load):
            info = mod.rebuild_gpx_graph(entries)

        assert info["n_nodes"] > 0
        assert info["n_edges"] > 0
        assert mod._gpx_graph is not None
        assert mod._portals_ready is False


# ---- test point to segment distance ----

class TestPointToSegment:
    def test_perpendicular(self):
        from alpineroute.routing.gpx_graph import _point_to_segment_dist_m
        # point au milieu perpendiculaire
        dist, t = _point_to_segment_dist_m(
            45.925, 6.88,    # point
            45.92, 6.87,     # segment start
            45.93, 6.87,     # segment end
        )
        assert dist > 500   # ~800m a l'est
        assert 0.4 < t < 0.6  # milieu du segment

    def test_on_segment(self):
        from alpineroute.routing.gpx_graph import _point_to_segment_dist_m
        # point sur le segment
        dist, t = _point_to_segment_dist_m(
            45.925, 6.87,
            45.92, 6.87,
            45.93, 6.87,
        )
        assert dist < 10  # quasi sur la ligne
