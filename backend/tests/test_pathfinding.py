# tests pathfinding Dijkstra
import numpy as np
import pytest

from alpineroute.config import NODATA_VALUE
from alpineroute.routing.pathfinding import (
    prepare_cost_grid, run_pathfinding,
    dijkstra_anisotropic, run_aniso_alternatives,
)


class TestPrepareCostGrid:
    def test_nodata_becomes_inf(self):
        cost = np.array([[1.0, NODATA_VALUE], [2.0, 3.0]])
        grid = prepare_cost_grid(cost)
        assert np.isinf(grid[0, 1])

    def test_valid_values_preserved(self):
        cost = np.array([[1.5, 2.0], [3.0, 4.0]])
        grid = prepare_cost_grid(cost)
        assert abs(grid[0, 0] - 1.5) < 1e-5
        assert abs(grid[1, 1] - 4.0) < 1e-5


class TestRunPathfinding:
    def test_uniform_grid_finds_path(self):
        """Sur une grille uniforme, un chemin doit exister."""
        cost = np.ones((20, 20), dtype=np.float64)
        path, path_cost, dt = run_pathfinding(cost, (0, 0), (19, 19))
        assert len(path) > 0
        # premier et dernier pixel
        assert tuple(path[0]) == (0, 0)
        assert tuple(path[-1]) == (19, 19)

    def test_wall_of_inf_forces_detour(self):
        """Un mur d'inf au milieu force le chemin a contourner."""
        cost = np.ones((20, 20), dtype=np.float64)
        # mur vertical au milieu, sauf un passage en haut
        cost[:, 10] = np.inf
        cost[0, 10] = 1.0  # passage libre en haut

        path, path_cost, dt = run_pathfinding(cost, (10, 0), (10, 19))
        assert len(path) > 0
        # le chemin ne doit jamais passer par la colonne 10 sauf en row 0
        for r, c in path:
            if c == 10:
                assert r == 0, f"passage par le mur en ({r}, {c})"

    def test_adjacent_points(self):
        """Deux points adjacents: chemin de 2 pixels."""
        cost = np.ones((5, 5), dtype=np.float64)
        path, _, _ = run_pathfinding(cost, (2, 2), (2, 3))
        assert len(path) == 2

    def test_path_cost_positive(self):
        cost = np.full((10, 10), 2.0, dtype=np.float64)
        _, path_cost, _ = run_pathfinding(cost, (0, 0), (9, 9))
        assert path_cost > 0


class TestDijkstraAnisotropic:
    """Tests Dijkstra anisotrope (cout directionnel par edge)."""

    def test_flat_grid_finds_path(self):
        """Grille plate, base_cost uniforme: on doit trouver un chemin."""
        dem = np.full((20, 20), 2000.0)  # altitude constante
        base = np.ones((20, 20), dtype=np.float64)
        path, cost, dt = dijkstra_anisotropic(dem, base, (0, 0), (19, 19), 1.0)
        assert len(path) > 0
        assert tuple(path[0]) == (0, 0)
        assert tuple(path[-1]) == (19, 19)
        assert cost > 0

    def test_slope_contour(self):
        """Sur un plan incline est->ouest, l'anisotrope devrait
        produire un chemin different de la ligne droite."""
        H, W = 30, 30
        # gradient W->E : altitude augmente vers la droite
        dem = np.zeros((H, W))
        for c in range(W):
            dem[:, c] = 2000 + c * 50  # 50m / pixel = ~87 deg, tres raide

        base = np.ones((H, W), dtype=np.float64)

        # isotrope sur grille a cout=1: quasi diagonale
        iso_grid = np.ones((H, W), dtype=np.float64)
        iso_path, _, _ = run_pathfinding(iso_grid, (0, 0), (29, 29))

        # aniso: devrait contourner la pente
        aniso_path, _, _ = dijkstra_anisotropic(
            dem, base, (0, 0), (29, 29), 1.0)

        assert len(aniso_path) > 0
        # le chemin aniso sera plus long (plus de detours)
        assert len(aniso_path) > len(iso_path)

    def test_nodata_blocks(self):
        """Les pixels inf dans base_cost doivent etre bloques."""
        dem = np.full((10, 10), 2000.0)
        base = np.ones((10, 10), dtype=np.float64)
        # mur vertical au milieu
        base[:, 5] = np.inf
        base[0, 5] = 1.0  # passage en haut

        path, _, _ = dijkstra_anisotropic(dem, base, (5, 0), (5, 9), 1.0)
        assert len(path) > 0
        for r, c in path:
            if c == 5:
                assert r == 0, f"passe par le mur en ({r}, {c})"

    def test_no_path_all_blocked(self):
        """Aucun chemin possible: retourne tableau vide."""
        dem = np.full((10, 10), 2000.0)
        base = np.ones((10, 10), dtype=np.float64)
        # mur complet
        base[:, 5] = np.inf

        path, cost, _ = dijkstra_anisotropic(dem, base, (5, 0), (5, 9), 1.0)
        assert len(path) == 0

    def test_alternatives_penalty(self):
        """Alternatives anisotropes via penalty method."""
        dem = np.full((20, 20), 2000.0)
        base = np.ones((20, 20), dtype=np.float64)

        results = run_aniso_alternatives(
            dem, base, (0, 0), (19, 19), 1.0, n_alt=2)
        assert len(results) >= 1
        # la premiere route doit etre valide
        assert len(results[0][0]) > 0
        # si on a des alternatives, elles doivent couter plus cher
        if len(results) > 1:
            assert results[1][1] >= results[0][1]
