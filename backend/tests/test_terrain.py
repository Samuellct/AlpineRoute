# tests analyse terrain (pente, aspect, rugosite)
import numpy as np
import pytest

from alpineroute.config import NODATA_VALUE
from alpineroute.dem.terrain import compute_slope_aspect, compute_roughness


class TestSlopeAspect:
    def test_flat_dem_zero_slope(self, flat_dem):
        slope, aspect = compute_slope_aspect(flat_dem, resolution=1.0)
        # les bords sont dilates donc on check le centre
        center = slope[5:15, 5:15]
        valid = center != NODATA_VALUE
        assert np.all(center[valid] < 1.0)

    def test_east_facing_slope(self):
        """DEM qui monte vers l'est -> face ouest (aspect ~ 270)."""
        dem = np.zeros((30, 30), dtype=np.float32)
        for c in range(30):
            dem[:, c] = 3000.0 + c * 10.0  # monte vers l'est
        slope, aspect = compute_slope_aspect(dem, resolution=1.0)
        center_slope = slope[10:20, 10:20]
        center_aspect = aspect[10:20, 10:20]
        valid = (center_slope != NODATA_VALUE) & (center_aspect != NODATA_VALUE)
        # pente non nulle
        assert np.mean(center_slope[valid]) > 5.0
        # aspect ~ 270 (face ouest, car la pente descend vers l'ouest)
        mean_aspect = np.mean(center_aspect[valid])
        assert 240 < mean_aspect < 300

    def test_nodata_propagated(self, flat_dem):
        dem = flat_dem.copy()
        dem[10, 10] = NODATA_VALUE
        slope, aspect = compute_slope_aspect(dem, resolution=1.0)
        # le pixel nodata + ses voisins dilates
        assert slope[10, 10] == NODATA_VALUE


class TestRoughness:
    def test_flat_dem_low_roughness(self, flat_dem):
        rough = compute_roughness(flat_dem)
        center = rough[5:15, 5:15]
        valid = center != NODATA_VALUE
        assert np.all(center[valid] < 0.01)

    def test_variable_dem_has_roughness(self):
        """DEM avec variance locale connue => TRI > 0."""
        rng = np.random.default_rng(99)
        dem = 3000.0 + rng.normal(0, 10, (30, 30)).astype(np.float32)
        rough = compute_roughness(dem)
        center = rough[5:25, 5:25]
        valid = center != NODATA_VALUE
        assert np.mean(center[valid]) > 1.0
