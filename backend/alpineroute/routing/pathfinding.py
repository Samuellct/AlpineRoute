# pathfinding Dijkstra sur surface de cout
# source: T05_pathfinding.py

import math
import time
import logging
import numpy as np
from skimage.graph import route_through_array
from scipy.ndimage import binary_dilation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra
from skimage.morphology import disk

from alpineroute.config import (
    NODATA_VALUE, COST_NODATA_VALUE,
    PENALTY_MULTIPLIER, PENALTY_BUFFER_PX,
    TOBLER_BASE_SPEED_KMH, TOBLER_OPTIMAL_GRADIENT, OFF_TRAIL_FACTOR,
    STEEP_SLOPE_THRESHOLD_DEG, STEEP_SLOPE_MULTIPLIER,
    CRITICAL_SLOPE_DEG, CRITICAL_SLOPE_MULTIPLIER,
)

logger = logging.getLogger(__name__)


def prepare_cost_grid(cost_surface):
    """Prepare la grille de cout pour le pathfinding:
    nodata -> inf, cap valeurs extremes."""
    cost = cost_surface.astype(np.float64)
    nodata_mask = (cost == NODATA_VALUE)
    cost[nodata_mask] = np.inf
    cost = np.clip(cost, 0, COST_NODATA_VALUE)
    cost[nodata_mask] = np.inf
    return cost


def run_pathfinding(cost, start_rc, end_rc):
    """Lance Dijkstra via skimage. Retourne (path_coords, path_cost, elapsed)."""
    logger.info("start=%s end=%s grille=%dx%d (%d px)",
                start_rc, end_rc, cost.shape[0], cost.shape[1], cost.size)

    t0 = time.time()
    path_coords, path_cost = route_through_array(
        cost,
        start=start_rc,
        end=end_rc,
        fully_connected=True,
        geometric=True,
    )
    dt = time.time() - t0

    path_coords = np.array(path_coords)
    logger.info("done: %d px, cout=%.2f, temps=%.1fs",
                len(path_coords), path_cost, dt)

    return path_coords, path_cost, dt


def _apply_penalty(cost_grid, path_coords, multiplier=None, buffer_px=None):
    """Penalise le corridor autour d'un trajet pour forcer des alternatives.
    Modifie cost_grid in-place, ne touche pas aux pixels inf (nodata)."""
    if multiplier is None:
        multiplier = PENALTY_MULTIPLIER
    if buffer_px is None:
        buffer_px = PENALTY_BUFFER_PX

    mask = np.zeros(cost_grid.shape, dtype=bool)
    rows, cols = path_coords[:, 0], path_coords[:, 1]
    mask[rows, cols] = True

    # dilater autour du trajet
    struct = disk(buffer_px)
    corridor = binary_dilation(mask, structure=struct)

    # pas de penalite sur nodata
    valid = np.isfinite(cost_grid) & corridor
    cost_grid[valid] *= multiplier


def run_pathfinding_alternatives(cost_grid, start_rc, end_rc, n_alt=3):
    """Lance le pathfinding optimal + n_alt alternatives via penalty method.
    Retourne [(path_coords, path_cost, elapsed), ...]"""
    results = []

    # route optimale
    path_coords, path_cost, dt = run_pathfinding(cost_grid, start_rc, end_rc)
    results.append((path_coords, path_cost, dt))

    if len(path_coords) == 0:
        return results

    for i in range(n_alt):
        _apply_penalty(cost_grid, results[-1][0])
        try:
            pc, cost, elapsed = run_pathfinding(cost_grid, start_rc, end_rc)
        except Exception as e:
            logger.warning("alt route %d failed: %s", i + 1, e)
            break
        if len(pc) == 0:
            logger.warning("alt route %d: chemin vide, on arrete", i + 1)
            break
        results.append((pc, cost, elapsed))
        logger.info("alternative %d/%d ok (cout=%.2f)", i + 1, n_alt, cost)

    return results


# =====================================================
#  Dijkstra anisotrope (scipy.sparse.csgraph)
# =====================================================
# Construit un graphe oriente creux: chaque arete (r,c)->(nr,nc)
# a un cout directionnel tenant compte du gradient signe.
# Utilise le Dijkstra C de scipy -> ~100x plus rapide que heapq pur.

# 8-connexite
_NEIGHBORS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
_DIAG = math.sqrt(2)
_NEIGHBOR_DISTS = [
    _DIAG, 1.0, _DIAG,
    1.0,        1.0,
    _DIAG, 1.0, _DIAG,
]

# vitesse a plat (ref pour normalisation du cout Tobler)
_V_FLAT = (TOBLER_BASE_SPEED_KMH
           * math.exp(-3.5 * TOBLER_OPTIMAL_GRADIENT)
           * OFF_TRAIL_FACTOR)

# seuils pente en radians
_STEEP_RAD = math.radians(STEEP_SLOPE_THRESHOLD_DEG)
_CRIT_RAD = math.radians(CRITICAL_SLOPE_DEG)


def _tobler_cost_vectorized(gradient):
    """Tobler directionnel vectorise.
    gradient = dz/dist signe, retourne un multiplicateur (1.0 sur du plat)."""
    v = TOBLER_BASE_SPEED_KMH * np.exp(
        -3.5 * np.abs(gradient + TOBLER_OPTIMAL_GRADIENT)
    ) * OFF_TRAIL_FACTOR
    v = np.maximum(v, 0.01)
    cost = _V_FLAT / v

    # penalites pente raide
    slope_rad = np.abs(np.arctan(gradient))
    cost = np.where(
        slope_rad > _CRIT_RAD,
        cost * STEEP_SLOPE_MULTIPLIER * CRITICAL_SLOPE_MULTIPLIER,
        np.where(slope_rad > _STEEP_RAD, cost * STEEP_SLOPE_MULTIPLIER, cost),
    )
    return cost


def _build_aniso_graph(dem, base_cost, resolution):
    """Construit le graphe creux oriente pour le pathfinding anisotrope.
    Retourne (csr_matrix, src_arr, dst_arr, weight_arr) pour pouvoir
    re-penaliser les aretes sans tout reconstruire."""
    H, W = dem.shape
    N = H * W
    src_parts, dst_parts, w_parts = [], [], []

    for k, (dr, dc) in enumerate(_NEIGHBORS):
        edge_m = _NEIGHBOR_DISTS[k] * resolution

        # range source -> dest doit rester dans la grille
        r0, r1 = max(0, -dr), H - max(0, dr)
        c0, c1 = max(0, -dc), W - max(0, dc)

        s_dem = dem[r0:r1, c0:c1]
        d_dem = dem[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        s_bc = base_cost[r0:r1, c0:c1]
        d_bc = base_cost[r0 + dr:r1 + dr, c0 + dc:c1 + dc]

        gradient = (d_dem - s_dem) / edge_m
        tobler = _tobler_cost_vectorized(gradient)
        w = tobler * (s_bc + d_bc) * 0.5 * edge_m

        ok = np.isfinite(s_bc) & np.isfinite(d_bc) & np.isfinite(w)

        # indices source/dest en flat
        rr = np.arange(r0, r1, dtype=np.int64)[:, None] + np.zeros(c1 - c0, dtype=np.int64)
        cc = np.zeros(r1 - r0, dtype=np.int64)[:, None] + np.arange(c0, c1, dtype=np.int64)

        src_flat = (rr * W + cc).ravel()
        dst_flat = ((rr + dr) * W + (cc + dc)).ravel()

        m = ok.ravel()
        src_parts.append(src_flat[m])
        dst_parts.append(dst_flat[m])
        w_parts.append(w.ravel()[m])

    src_arr = np.concatenate(src_parts)
    dst_arr = np.concatenate(dst_parts)
    w_arr = np.concatenate(w_parts)

    graph = csr_matrix((w_arr, (src_arr, dst_arr)), shape=(N, N))

    logger.info("graphe aniso: %d noeuds, %d aretes (%.1f MB)",
                N, len(w_arr), w_arr.nbytes / 1e6)
    return graph, src_arr, dst_arr, w_arr


def _traceback_path(predecessors, start_flat, end_flat, W):
    """Reconstruit le chemin (row, col) a partir du tableau de predecesseurs."""
    path = []
    cur = end_flat
    # -9999 = pas de predecesseur dans scipy
    while cur != start_flat and cur != -9999:
        path.append(cur)
        cur = predecessors[cur]

    if cur == -9999:
        return np.array([])

    path.append(start_flat)
    path.reverse()
    coords = np.array([(idx // W, idx % W) for idx in path])
    return coords


def dijkstra_anisotropic(dem, base_cost, start_rc, end_rc, resolution):
    """Dijkstra anisotrope via scipy.sparse sur graphe oriente.
    Chaque arete a un cout Tobler base sur le gradient signe."""
    H, W = dem.shape
    sr, sc = start_rc
    er, ec = end_rc

    logger.info("aniso sparse: start=(%d,%d) end=(%d,%d) grid=%dx%d res=%.1f",
                sr, sc, er, ec, H, W, resolution)
    t0 = time.time()

    graph, _, _, _ = _build_aniso_graph(dem, base_cost, resolution)
    t_build = time.time() - t0

    start_flat = sr * W + sc
    end_flat = er * W + ec

    t1 = time.time()
    dist, pred = sp_dijkstra(
        graph, directed=True, indices=start_flat,
        return_predecessors=True,
    )
    t_dijk = time.time() - t1

    path_coords = _traceback_path(pred, start_flat, end_flat, W)
    total_cost = dist[end_flat]
    dt = time.time() - t0

    if len(path_coords) == 0:
        logger.warning("aniso: pas de chemin trouve")
        return np.array([]), 0.0, dt

    logger.info("aniso done: %d px, cout=%.2f, build=%.1fs dijk=%.1fs total=%.1fs",
                len(path_coords), total_cost, t_build, t_dijk, dt)
    return path_coords, total_cost, dt


def _apply_penalty_sparse(w_arr, src_arr, dst_arr, path_coords, W,
                          multiplier=None, buffer_px=None):
    """Penalise les aretes du corridor. Retourne un nouveau w_arr."""
    if multiplier is None:
        multiplier = PENALTY_MULTIPLIER
    if buffer_px is None:
        buffer_px = PENALTY_BUFFER_PX

    H = int(src_arr.max() // W) + 2  # approximation hauteur grille
    mask = np.zeros(H * W, dtype=bool)
    flat_idx = path_coords[:, 0] * W + path_coords[:, 1]
    mask[flat_idx] = True

    # dilater en 2D
    mask_2d = mask[:H * W].reshape(H, W)
    struct = disk(buffer_px)
    corridor_2d = binary_dilation(mask_2d, structure=struct)
    corridor_flat = corridor_2d.ravel()

    # penaliser les aretes dont src ou dst dans le corridor
    in_corridor = corridor_flat[src_arr] | corridor_flat[dst_arr]
    w_new = w_arr.copy()
    w_new[in_corridor] *= multiplier
    return w_new


def run_aniso_alternatives(dem, base_cost, start_rc, end_rc,
                           resolution, n_alt=3):
    """Alternatives anisotropes via penalty method sur graphe creux."""
    H, W = dem.shape
    results = []

    logger.info("aniso alternatives: %d demandees, grid=%dx%d", n_alt, H, W)
    t0 = time.time()

    graph, src_arr, dst_arr, w_arr = _build_aniso_graph(dem, base_cost, resolution)
    t_build = time.time() - t0
    logger.info("graphe construit en %.1fs", t_build)

    start_flat = start_rc[0] * W + start_rc[1]
    end_flat = end_rc[0] * W + end_rc[1]
    N = H * W

    # route optimale
    t1 = time.time()
    dist, pred = sp_dijkstra(graph, directed=True, indices=start_flat,
                             return_predecessors=True)
    path_coords = _traceback_path(pred, start_flat, end_flat, W)
    dt = time.time() - t1

    if len(path_coords) == 0:
        logger.warning("aniso: pas de chemin principal")
        return [(np.array([]), 0.0, dt)]

    results.append((path_coords, dist[end_flat], dt + t_build))

    # alternatives via penalites
    current_w = w_arr
    for i in range(n_alt):
        current_w = _apply_penalty_sparse(
            current_w, src_arr, dst_arr, results[-1][0], W)
        penalized = csr_matrix((current_w, (src_arr, dst_arr)), shape=(N, N))

        try:
            t1 = time.time()
            dist_p, pred_p = sp_dijkstra(
                penalized, directed=True, indices=start_flat,
                return_predecessors=True)
            pc = _traceback_path(pred_p, start_flat, end_flat, W)
            dt_p = time.time() - t1
        except Exception as e:
            logger.warning("aniso alt %d failed: %s", i + 1, e)
            break

        if len(pc) == 0:
            logger.warning("aniso alt %d: chemin vide", i + 1)
            break

        results.append((pc, dist_p[end_flat], dt_p))
        logger.info("aniso alt %d/%d ok (cout=%.2f, %.1fs)",
                    i + 1, n_alt, dist_p[end_flat], dt_p)

    return results
