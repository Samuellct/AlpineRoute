# tests radiation solaire
import numpy as np
import pytest

from alpineroute.cost.radiation import (
    solar_position,
    compute_horizon_angles,
    is_shadowed,
    direct_irradiance,
    daily_radiation,
    compute_radiation_cost,
)


# ---- position solaire ----

class TestSolarPosition:
    def test_summer_noon_chamonix(self):
        # 21 juin (doy 172), ~12h30 UTC -> midi solaire a Chamonix (lon ~6.9)
        elev, az = solar_position(172, 12.5, 45.92, 6.87)
        # elevation ~65-68 deg en ete au solstice
        assert 55 < elev < 75, f"elevation={elev}"
        # azimut ~180-215 (sud, legerement ouest apres midi solaire)
        assert 150 < az < 220, f"azimut={az}"

    def test_winter_noon(self):
        # 21 dec (doy 355), midi solaire
        elev, az = solar_position(355, 12.0, 45.92, 6.87)
        # elevation basse en hiver ~20-25 deg
        assert 10 < elev < 35, f"elevation={elev}"

    def test_night(self):
        # minuit en ete -> soleil sous l'horizon
        elev, _ = solar_position(172, 1.0, 45.92, 6.87)
        assert elev < 0

    def test_sunrise_positive(self):
        # 7h UTC en ete -> soleil leve
        elev, _ = solar_position(172, 7.0, 45.92, 6.87)
        assert elev > 0


# ---- horizons ----

class TestHorizonAngles:
    def test_flat_terrain(self):
        # DEM plat -> horizons ~0 partout
        dem = np.full((50, 50), 3000.0, dtype=np.float32)
        horizons = compute_horizon_angles(dem, 5.0, n_azimuths=8)
        assert horizons.shape == (8, 50, 50)
        # au centre, loin des bords
        center = horizons[:, 20:30, 20:30]
        assert np.all(center < 2.0), f"max horizon on flat={center.max()}"

    def test_wall_south(self):
        # mur de 500m au sud du centre
        dem = np.full((100, 100), 3000.0, dtype=np.float32)
        dem[70:, :] = 3500.0  # elevation +500m au sud (lignes hautes = sud en raster)
        horizons = compute_horizon_angles(dem, 5.0, n_azimuths=8)
        # azimut 180 (sud) = index 4 pour 8 azimuths
        # les pixels centraux doivent voir un horizon eleve vers le sud
        south_horizon = horizons[4, 40, 50]
        assert south_horizon > 5.0, f"south horizon={south_horizon}"


# ---- ombre ----

class TestShadow:
    def test_shadowed_low_sun(self):
        # horizon a 20 deg, soleil a 10 deg -> ombre
        horizons = np.full((4, 10, 10), 20.0, dtype=np.float32)
        shadow = is_shadowed(horizons, 10.0, 90.0)
        assert shadow[5, 5]

    def test_not_shadowed_high_sun(self):
        # horizon a 5 deg, soleil a 45 deg -> pas d'ombre
        horizons = np.full((4, 10, 10), 5.0, dtype=np.float32)
        shadow = is_shadowed(horizons, 45.0, 90.0)
        assert not shadow[5, 5]


# ---- irradiance ----

class TestIrradiance:
    def test_flat_uniform(self):
        # terrain plat face au soleil
        slope = np.zeros((10, 10), dtype=np.float32)
        aspect = np.zeros((10, 10), dtype=np.float32)
        irr = direct_irradiance(45.0, 180.0, slope, aspect)
        # terrain plat: irr = cos(zenith) = cos(45) ~0.707
        assert np.allclose(irr, np.cos(np.radians(45.0)), atol=0.01)

    def test_shadowed_zero(self):
        slope = np.full((5, 5), 30.0, dtype=np.float32)
        aspect = np.full((5, 5), 180.0, dtype=np.float32)
        shadow = np.ones((5, 5), dtype=bool)
        irr = direct_irradiance(45.0, 180.0, slope, aspect, shadow)
        assert np.all(irr == 0)


# ---- radiation journaliere ----

class TestDailyRadiation:
    def test_flat_uniform(self):
        dem = np.full((20, 20), 3000.0, dtype=np.float32)
        slope = np.zeros((20, 20), dtype=np.float32)
        aspect = np.zeros((20, 20), dtype=np.float32)
        rad = daily_radiation(dem, slope, aspect, 5.0, 45.92, 6.87, 172)
        assert rad.shape == (20, 20)
        # terrain plat: radiation quasi-uniforme
        center = rad[5:15, 5:15]
        assert center.std() < center.mean() * 0.01


# ---- cout radiation ----

class TestRadiationCost:
    def test_summer_high_rad_penalty(self):
        # forte radiation en ete -> penalite
        rad = np.full((10, 10), 10.0, dtype=np.float32)
        slope = np.full((10, 10), 30.0, dtype=np.float32)
        elev = np.full((10, 10), 3000.0, dtype=np.float32)
        cost = compute_radiation_cost(rad, slope, elev, month=7)
        assert np.all(cost >= 1.0)

    def test_winter_low_rad_penalty(self):
        # faible radiation en hiver -> penalite
        rad = np.full((10, 10), 0.5, dtype=np.float32)
        slope = np.full((10, 10), 30.0, dtype=np.float32)
        elev = np.full((10, 10), 3000.0, dtype=np.float32)
        cost = compute_radiation_cost(rad, slope, elev, month=1)
        assert np.all(cost >= 1.0)

    def test_low_slope_no_penalty(self):
        # pente < seuil -> pas de penalite
        rad = np.full((10, 10), 10.0, dtype=np.float32)
        slope = np.full((10, 10), 5.0, dtype=np.float32)
        elev = np.full((10, 10), 3000.0, dtype=np.float32)
        cost = compute_radiation_cost(rad, slope, elev, month=7)
        assert np.allclose(cost, 1.0)

    def test_low_altitude_no_penalty(self):
        # altitude < seuil -> pas de penalite
        rad = np.full((10, 10), 10.0, dtype=np.float32)
        slope = np.full((10, 10), 30.0, dtype=np.float32)
        elev = np.full((10, 10), 1000.0, dtype=np.float32)
        cost = compute_radiation_cost(rad, slope, elev, month=7)
        assert np.allclose(cost, 1.0)
