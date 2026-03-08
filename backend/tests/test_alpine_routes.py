# tests parsing GPX, stats, GeoJSON alpine
import os
import pytest

from alpineroute.alpine.routes import (
    load_gpx, compute_stats, route_to_geojson, GRADE_ORDINAL,
)

# 3 points pres de Chamonix, 3200-3300m
MINI_GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test">
  <trk><name>test</name><trkseg>
    <trkpt lat="45.8326" lon="6.8652"><ele>3200</ele></trkpt>
    <trkpt lat="45.8340" lon="6.8670"><ele>3250</ele></trkpt>
    <trkpt lat="45.8355" lon="6.8690"><ele>3300</ele></trkpt>
  </trkseg></trk>
</gpx>
"""


@pytest.fixture
def gpx_file(tmp_path):
    f = tmp_path / "test.gpx"
    f.write_text(MINI_GPX, encoding="utf-8")
    return str(f)


class TestLoadGpx:
    def test_valid(self, gpx_file):
        pts = load_gpx(gpx_file)
        assert pts is not None
        assert len(pts) == 3
        assert pts[0] == (45.8326, 6.8652, 3200.0)

    def test_missing_file(self, tmp_path):
        result = load_gpx(str(tmp_path / "nope.gpx"))
        assert result is None


class TestComputeStats:
    def test_basic(self, gpx_file):
        pts = load_gpx(gpx_file)
        dist, dplus = compute_stats(pts)
        assert dist > 0
        assert dplus == 100.0  # 3200 -> 3250 -> 3300

    def test_single_point(self):
        dist, dplus = compute_stats([(45.0, 6.0, 3000)])
        assert dist == 0.0
        assert dplus == 0.0


class TestRouteToGeojson:
    def test_structure(self, gpx_file):
        pts = load_gpx(gpx_file)
        entry = {"massif": "Mont-Blanc", "summit": "Aiguille du Midi",
                 "voie": "Arête des Cosmiques", "grade": "AD"}
        gj = route_to_geojson(entry, pts)

        assert gj["type"] == "Feature"
        assert gj["geometry"]["type"] == "LineString"
        coords = gj["geometry"]["coordinates"]
        assert len(coords) == 3
        # coords = [lon, lat, alt]
        assert coords[0][0] == 6.8652
        assert coords[0][1] == 45.8326
        assert coords[0][2] == 3200.0

        props = gj["properties"]
        assert props["massif"] == "Mont-Blanc"
        assert props["summit"] == "Aiguille du Midi"
        assert props["grade"] == "AD"
        assert props["distance_m"] > 0
        assert props["dplus_m"] == 100.0


class TestGradeOrdinal:
    def test_known_grades(self):
        assert GRADE_ORDINAL["F"] == 1
        assert GRADE_ORDINAL["AD"] == 7
        assert GRADE_ORDINAL["TD"] == 13
        assert GRADE_ORDINAL["ABO+"] == 20
