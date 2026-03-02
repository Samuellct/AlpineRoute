# tests fonctions utilitaires
import numpy as np
import pytest

from alpineroute.config import NODATA_VALUE, BBOX_MAX_SIZE_M
from alpineroute.utils import (
    wgs84_to_l93, l93_to_wgs84,
    wgs84_to_pixel, pixel_to_l93,
    compute_bbox, compute_distance_2d, compute_path_stats,
    make_nodata_mask,
    PointOutOfBoundsError,
)


# ---- projections ----

class TestProjections:
    def test_roundtrip_chamonix(self):
        """WGS84 -> L93 -> WGS84 = identite (a ~1m pres)."""
        lon, lat = 6.8694, 45.9237
        x, y = wgs84_to_l93(lon, lat)
        lon2, lat2 = l93_to_wgs84(x, y)
        assert abs(lon - lon2) < 1e-6
        assert abs(lat - lat2) < 1e-6

    def test_chamonix_in_l93_range(self):
        """Chamonix doit etre dans la zone L93 attendue."""
        x, y = wgs84_to_l93(6.8694, 45.9237)
        # L93 Chamonix: x ~ 1001km, y ~ 6542km
        assert 990_000 < x < 1020_000
        assert 6530_000 < y < 6560_000


# ---- wgs84_to_pixel ----

class TestWgs84ToPixel:
    def test_valid_point(self, fake_transform):
        """Un point dans la grille doit retourner un pixel valide."""
        # la grille commence en (1001000, 6542000) et fait 20x20 a 1m
        # centre de la grille => L93 (1001010, 6541990)
        lon, lat = l93_to_wgs84(1001010.0, 6541990.0)
        row, col, x, y = wgs84_to_pixel(lat, lon, fake_transform, (20, 20))
        assert 0 <= row < 20
        assert 0 <= col < 20

    def test_out_of_bounds(self, fake_transform):
        """Un point tres loin doit lever PointOutOfBoundsError."""
        with pytest.raises(PointOutOfBoundsError):
            wgs84_to_pixel(48.0, 2.0, fake_transform, (20, 20))


# ---- pixel_to_l93 ----

class TestPixelToL93:
    def test_coherent_with_transform(self, fake_transform):
        rows = np.array([0, 10])
        cols = np.array([0, 10])
        xs, ys = pixel_to_l93(rows, cols, fake_transform)
        # pixel (0,0) => coin NW
        assert abs(xs[0] - 1001000.0) < 1.0
        assert abs(ys[0] - 6542000.0) < 1.0


# ---- compute_bbox ----

class TestComputeBbox:
    def test_margin_applied(self):
        start = (45.92, 6.87)
        end = (45.93, 6.88)
        result = compute_bbox(start, end, margin_m=2000)
        bbox = result["bbox_l93"]
        # la bbox doit etre nettement plus grande que la distance entre les pts
        dx = bbox["xmax"] - bbox["xmin"]
        dy = bbox["ymax"] - bbox["ymin"]
        assert dx >= 4000
        assert dy >= 4000

    def test_max_size_clipping(self):
        # pts tres eloignes (Chamonix - Grenoble)
        start = (45.92, 6.87)
        end = (45.19, 5.72)
        result = compute_bbox(start, end, max_size_m=30000)
        bbox = result["bbox_l93"]
        dx = bbox["xmax"] - bbox["xmin"]
        dy = bbox["ymax"] - bbox["ymin"]
        # doit etre clippe a ~30km + alignement
        assert dx <= 32000
        assert dy <= 32000

    def test_alignment_1000m(self):
        start = (45.92, 6.87)
        end = (45.93, 6.88)
        result = compute_bbox(start, end)
        bbox = result["bbox_l93"]
        assert bbox["xmin"] % 1000 == 0
        assert bbox["ymin"] % 1000 == 0


# ---- compute_distance_2d ----

class TestDistance2D:
    def test_triangle_345(self):
        coords = np.array([[0.0, 0.0], [3.0, 4.0]])
        d = compute_distance_2d(coords)
        assert abs(d - 5.0) < 1e-6

    def test_straight_line(self):
        coords = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        d = compute_distance_2d(coords)
        assert abs(d - 20.0) < 1e-6


# ---- compute_path_stats ----

class TestPathStats:
    def test_ascending_path(self, tiny_dem, fake_transform):
        """Chemin montant: dplus > 0, dminus faible."""
        # chemin vertical de haut en bas de la grille (= montee dans tiny_dem)
        path = np.array([[r, 10] for r in range(20)])
        stats, arrays = compute_path_stats(path, tiny_dem, fake_transform)
        assert stats["dplus"] > 0
        assert stats["time_tobler_h"] > 0
        assert stats["n_pixels"] == 20

    def test_flat_path(self, flat_dem, fake_transform):
        """Chemin sur terrain plat: dplus ~ 0."""
        path = np.array([[10, c] for c in range(20)])
        stats, arrays = compute_path_stats(path, flat_dem, fake_transform)
        assert abs(stats["dplus"]) < 0.1
        assert abs(stats["dminus"]) < 0.1


# ---- make_nodata_mask ----

class TestNodataMask:
    def test_nodata_detected(self):
        dem = np.full((10, 10), 3000.0, dtype=np.float32)
        dem[5, 5] = NODATA_VALUE
        mask = make_nodata_mask(dem, dilate=False)
        assert mask[5, 5]
        assert not mask[0, 0]

    def test_dilation_extends(self):
        dem = np.full((10, 10), 3000.0, dtype=np.float32)
        dem[5, 5] = NODATA_VALUE
        mask = make_nodata_mask(dem, dilate=True)
        # les voisins doivent aussi etre masques
        assert mask[4, 5]
        assert mask[5, 6]
        assert mask[6, 6]
