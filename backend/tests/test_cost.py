# tests surface de cout
import numpy as np
import pytest

from alpineroute.config import (
    NODATA_VALUE,
    STEEP_ONSET_DEG, STEEP_FULL_DEG, STEEP_MAX_MULTIPLIER,
    GLACIER_COST_FLAT, GLACIER_COST_STEEP, GLACIER_COST_VERY_STEEP,
    ROUGHNESS_SCALE, ROUGHNESS_CLAMP,
)
from alpineroute.cost.surface import (
    compute_slope_cost,
    compute_altitude_cost,
    compute_aspect_cost,
    compute_glacier_cost,
    compute_roughness_cost,
    build_cost_surface,
)


# ---- slope cost (Tobler) ----

class TestSlopeCost:
    def test_flat_terrain_is_baseline(self):
        cost = compute_slope_cost(np.array([0.0]))
        assert abs(cost[0] - 1.0) < 0.05

    def test_moderate_slope_above_one(self):
        cost = compute_slope_cost(np.array([20.0]))
        assert cost[0] > 1.0

    def test_steep_slope_penalized(self):
        # 50 deg: bien dans la zone progressive, cout tres eleve
        cost_mild = compute_slope_cost(np.array([20.0]))
        cost_steep = compute_slope_cost(np.array([50.0]))
        assert cost_steep[0] > cost_mild[0] * 5

    def test_full_steep_multiplier(self):
        cost_55 = compute_slope_cost(np.array([55.0]))
        # a STEEP_FULL_DEG, le facteur doit etre ~STEEP_MAX_MULTIPLIER
        assert cost_55[0] > 50

    def test_array_broadcast(self):
        slopes = np.array([[0, 10], [30, 60]], dtype=np.float32)
        cost = compute_slope_cost(slopes)
        assert cost.shape == (2, 2)
        # plat < raide
        assert cost[0, 0] < cost[1, 1]

    def test_gradient_clipped(self):
        # 89 deg: tan ~= 573, verifier pas d'inf/nan apres clip
        cost = compute_slope_cost(np.array([89.0, 89.9]))
        assert np.all(np.isfinite(cost))
        assert np.all(cost > 0)

    def test_progressive_penalty_onset(self):
        # juste avant le seuil = pas de penalite extra
        cost_34 = compute_slope_cost(np.array([34.0]))[0]
        cost_36 = compute_slope_cost(np.array([36.0]))[0]
        # 34 < onset -> facteur 1, 36 > onset -> facteur > 1
        assert cost_36 > cost_34

    def test_progressive_penalty_full(self):
        # au-dela de STEEP_FULL_DEG, le facteur steep est sature
        cost_55 = compute_slope_cost(np.array([STEEP_FULL_DEG]))[0]
        cost_60 = compute_slope_cost(np.array([60.0]))[0]
        # les deux ont le meme steep_factor, seul Tobler varie
        # le ratio doit rester modere (Tobler seul ~2x entre 55 et 60)
        assert cost_60 / cost_55 < 3.0

    def test_progressive_continuity(self):
        # pas de saut brutal entre 40 et 55 deg (zone progressive interne)
        # on evite le point d'onset (35) ou le facteur passe de 1 a >1
        slopes = np.arange(40, 56, dtype=np.float32)
        costs = compute_slope_cost(slopes)
        # ratio entre valeurs consecutives ne depasse pas 1.4
        for i in range(len(costs) - 1):
            ratio = costs[i + 1] / costs[i]
            assert ratio < 1.4, f"saut a {slopes[i]}->{slopes[i+1]}: ratio={ratio:.2f}"


# ---- altitude cost (hypoxia) ----

class TestAltitudeCost:
    def test_below_threshold_no_penalty(self):
        cost = compute_altitude_cost(np.array([1000.0]))
        assert abs(cost[0] - 1.0) < 1e-5

    def test_high_alt_acclimatized(self):
        cost = compute_altitude_cost(np.array([3000.0]), acclimatized=True)
        assert cost[0] > 1.0

    def test_not_acclimatized_worse(self):
        alt = np.array([3500.0])
        c_acc = compute_altitude_cost(alt, acclimatized=True)
        c_noacc = compute_altitude_cost(alt, acclimatized=False)
        assert c_noacc[0] > c_acc[0]

    def test_moderate_altitude_low_penalty(self):
        # 2000m: dans le palier modere, penalite legere (taux 0.01)
        cost = compute_altitude_cost(np.array([2000.0]))
        # reduction = (2000-1500)*0.01/1000 = 0.005 -> capacity ~0.995
        assert cost[0] > 1.0
        assert cost[0] < 1.02  # tres faible

    def test_high_altitude_full_penalty(self):
        # 4000m: palier modere + palier fort
        cost = compute_altitude_cost(np.array([4000.0]))
        # modere: (2500-1500)*0.01/1000 = 0.01
        # fort: (4000-2500)*0.03/1000 = 0.045
        # total: 0.055, capacity = 0.945
        assert cost[0] > 1.04

    def test_two_tier_boundary(self):
        # a 2500m pile, seul le palier modere s'applique
        cost_2500 = compute_altitude_cost(np.array([2500.0]))[0]
        # reduction = (2500-1500)*0.01/1000 = 0.01
        expected_capacity = 1.0 - 0.01
        expected = 1.0 / expected_capacity
        assert abs(cost_2500 - expected) < 0.001

    def test_hypoxia_continuity(self):
        # pas de saut a 2500m
        alts = np.arange(2400, 2600, 10, dtype=np.float32)
        costs = compute_altitude_cost(alts)
        for i in range(len(costs) - 1):
            assert costs[i + 1] >= costs[i] - 0.001  # monotone croissant
            ratio = costs[i + 1] / costs[i]
            assert ratio < 1.05  # pas de saut


# ---- aspect cost ----

class TestAspectCost:
    def test_summer_south_steep_high_penalty(self):
        # face plein sud, pente raide, altitude haute, mois juillet
        aspect = np.array([180.0])
        slope = np.array([40.0])
        elev = np.array([3000.0])
        cost = compute_aspect_cost(aspect, slope, elev, month=7)
        assert cost[0] > 1.0

    def test_summer_north_no_penalty(self):
        aspect = np.array([0.0])
        slope = np.array([40.0])
        elev = np.array([3000.0])
        cost = compute_aspect_cost(aspect, slope, elev, month=7)
        assert abs(cost[0] - 1.0) < 0.01

    def test_winter_north_steep_penalty(self):
        aspect = np.array([0.0])
        slope = np.array([35.0])
        elev = np.array([2500.0])
        cost = compute_aspect_cost(aspect, slope, elev, month=1)
        assert cost[0] > 1.0


# ---- glacier cost ----

class TestGlacierCost:
    def test_no_glacier_mask(self):
        cost = compute_glacier_cost(None, np.array([15.0]))
        assert abs(cost[0] - 1.0) < 1e-5

    def test_glacier_flat(self):
        mask = np.array([True])
        cost = compute_glacier_cost(mask, np.array([5.0]))
        assert abs(cost[0] - GLACIER_COST_FLAT) < 1e-5

    def test_glacier_steep_25deg(self):
        mask = np.array([True])
        cost = compute_glacier_cost(mask, np.array([25.0]))
        assert abs(cost[0] - GLACIER_COST_STEEP) < 1e-5

    def test_glacier_very_steep(self):
        mask = np.array([True])
        cost = compute_glacier_cost(mask, np.array([35.0]))
        assert abs(cost[0] - GLACIER_COST_VERY_STEEP) < 1e-5


# ---- roughness cost ----

class TestRoughnessCost:
    def test_zero_tri(self):
        cost = compute_roughness_cost(np.array([0.0]))
        assert abs(cost[0] - 1.0) < 1e-5

    def test_moderate_tri(self):
        cost = compute_roughness_cost(np.array([3.0]))
        expected = 1.0 + ROUGHNESS_SCALE * 3.0
        assert abs(cost[0] - expected) < 0.01

    def test_clamp_high_tri(self):
        cost = compute_roughness_cost(np.array([10.0]))
        expected = 1.0 + ROUGHNESS_SCALE * ROUGHNESS_CLAMP
        assert abs(cost[0] - expected) < 0.01


# ---- assemblage complet ----

class TestBuildCostSurface:
    def test_shape_preserved(self, tiny_dem, glacier_mask):
        slope = np.random.uniform(0, 30, (20, 20)).astype(np.float32)
        aspect = np.random.uniform(0, 360, (20, 20)).astype(np.float32)
        roughness = np.random.uniform(0, 2, (20, 20)).astype(np.float32)

        cost, factors, nodata_mask = build_cost_surface(
            tiny_dem, slope, aspect, roughness, glacier_mask)
        assert cost.shape == (20, 20)
        assert len(factors) == 5

    def test_nodata_propagated(self, flat_dem, glacier_mask):
        slope = np.zeros((20, 20), dtype=np.float32)
        slope[5, 5] = NODATA_VALUE
        aspect = np.zeros((20, 20), dtype=np.float32)
        roughness = np.zeros((20, 20), dtype=np.float32)

        cost, factors, nodata_mask = build_cost_surface(
            flat_dem, slope, aspect, roughness, glacier_mask)
        assert cost[5, 5] == NODATA_VALUE
        assert nodata_mask[5, 5]
