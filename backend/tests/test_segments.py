# tests segments terrain -- rasterize, merge, load_for_bbox

import os
import json
import tempfile
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from rasterio.transform import from_origin

from alpineroute.alpine.segments import (
    rasterize_segments, merge_trail_layers, load_segments_for_bbox,
)
from alpineroute.db.schema import init_db, get_connection


@pytest.fixture
def seg_transform():
    # 1m resolution, coin NW pres de Chamonix
    return from_origin(1001000.0, 6542000.0, 1.0, 1.0)


@pytest.fixture
def seg_db(tmp_path):
    db_file = str(tmp_path / "seg_test.db")
    init_db(db_file)
    return db_file


class TestRasterizeSegments:
    def test_empty_returns_ones(self, seg_transform):
        result = rasterize_segments([], seg_transform, (20, 20))
        assert result.shape == (20, 20)
        assert np.all(result == 1.0)

    def test_segment_creates_pixels(self, tmp_path):
        # coords WGS84 (45.9237, 6.87) -> L93 ~(999806, 6543349)
        # transform: coin NW couvre cette zone
        transform = from_origin(999700.0, 6543500.0, 1.0, 1.0)
        gpx_content = """<?xml version="1.0"?>
<gpx version="1.1"><trk><trkseg>
<trkpt lat="45.9237" lon="6.8700"><ele>3000</ele></trkpt>
<trkpt lat="45.9237" lon="6.8710"><ele>3010</ele></trkpt>
</trkseg></trk></gpx>"""
        gpx_dir = str(tmp_path / "gpx")
        os.makedirs(gpx_dir, exist_ok=True)
        with open(os.path.join(gpx_dir, "test_seg.gpx"), "w") as f:
            f.write(gpx_content)

        seg = {
            "gpx_path": "test_seg.gpx",
            "trail_cost": 0.25,
        }
        with patch("alpineroute.alpine.segments.GPX_DIR", gpx_dir):
            result = rasterize_segments([seg], transform, (500, 500))

        # au moins quelques pixels < 1.0
        assert np.any(result < 1.0)
        seg_pixels = result[result < 1.0]
        assert np.allclose(seg_pixels, 0.25)

    def test_default_cost_if_none(self, seg_transform, tmp_path):
        gpx_content = """<?xml version="1.0"?>
<gpx version="1.1"><trk><trkseg>
<trkpt lat="45.924" lon="6.870"><ele>3000</ele></trkpt>
<trkpt lat="45.924" lon="6.871"><ele>3010</ele></trkpt>
</trkseg></trk></gpx>"""
        gpx_dir = str(tmp_path / "gpx")
        os.makedirs(gpx_dir, exist_ok=True)
        with open(os.path.join(gpx_dir, "seg2.gpx"), "w") as f:
            f.write(gpx_content)

        seg = {"gpx_path": "seg2.gpx", "trail_cost": None}
        with patch("alpineroute.alpine.segments.GPX_DIR", gpx_dir):
            result = rasterize_segments([seg], seg_transform, (100, 100))
        seg_pixels = result[result < 1.0]
        if len(seg_pixels) > 0:
            assert np.allclose(seg_pixels, 0.3)


class TestMergeTrailLayers:
    def test_minimum_wins(self):
        a = np.array([[1.0, 0.5], [0.8, 1.0]], dtype=np.float32)
        b = np.array([[0.3, 1.0], [1.0, 0.6]], dtype=np.float32)
        result = merge_trail_layers(a, b)
        expected = np.array([[0.3, 0.5], [0.8, 0.6]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_none_osm(self):
        b = np.ones((5, 5), dtype=np.float32) * 0.5
        result = merge_trail_layers(None, b)
        assert result is b

    def test_none_segments(self):
        a = np.ones((5, 5), dtype=np.float32) * 0.5
        result = merge_trail_layers(a, None)
        assert result is a

    def test_both_none(self):
        result = merge_trail_layers(None, None)
        assert result is None


class TestLoadSegmentsForBbox:
    def test_segment_in_bbox(self, seg_db):
        # inserer un segment dans la bbox
        conn = get_connection(seg_db)
        conn.execute("""
            INSERT INTO terrain_segments
                (gpx_path, segment_type, trail_cost,
                 start_lat, start_lon, end_lat, end_lon)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("test.gpx", "trail", 0.3, 45.92, 6.87, 45.93, 6.88))
        conn.commit()
        conn.close()

        # bbox qui contient le segment (L93 autour de Chamonix)
        bbox = {"xmin": 999000, "ymin": 6539000, "xmax": 1003000, "ymax": 6545000}
        segments = load_segments_for_bbox(bbox, db_path=seg_db)
        assert len(segments) == 1
        assert segments[0]["gpx_path"] == "test.gpx"

    def test_segment_outside_bbox(self, seg_db):
        conn = get_connection(seg_db)
        conn.execute("""
            INSERT INTO terrain_segments
                (gpx_path, segment_type, trail_cost,
                 start_lat, start_lon, end_lat, end_lon)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("far.gpx", "trail", 0.3, 47.0, 3.0, 47.01, 3.01))
        conn.commit()
        conn.close()

        bbox = {"xmin": 999000, "ymin": 6539000, "xmax": 1003000, "ymax": 6545000}
        segments = load_segments_for_bbox(bbox, db_path=seg_db)
        assert len(segments) == 0

    def test_empty_db(self, seg_db):
        bbox = {"xmin": 999000, "ymin": 6539000, "xmax": 1003000, "ymax": 6545000}
        segments = load_segments_for_bbox(bbox, db_path=seg_db)
        assert len(segments) == 0
