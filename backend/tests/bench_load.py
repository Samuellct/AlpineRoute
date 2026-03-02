#!/usr/bin/env python3
"""
Benchmark performance sur grilles synthetiques.
Pas un test pytest -- script standalone.

Usage:
    cd backend
    python tests/bench_load.py
    python tests/bench_load.py --json
"""
import sys
import time
import tracemalloc
import numpy as np

# hack path pour import depuis backend/
sys.path.insert(0, ".")

from alpineroute.dem.terrain import compute_slope_aspect, compute_roughness
from alpineroute.cost.surface import build_cost_surface, build_base_cost
from alpineroute.routing.pathfinding import (
    prepare_cost_grid, run_pathfinding, dijkstra_anisotropic,
)


def make_synthetic_dem(size, seed=42):
    """Genere un DEM realiste: plan incline + bruit perlin-like."""
    rng = np.random.default_rng(seed)
    rows, cols = size, size
    # gradient global 2500-3500m
    base = np.linspace(2500, 3500, rows).reshape(-1, 1)
    dem = np.broadcast_to(base, (rows, cols)).copy().astype(np.float32)
    # bruit a differentes echelles
    dem += rng.normal(0, 20, dem.shape).astype(np.float32)
    return dem


def make_chamonix_dem(size, seed=123):
    """DEM avec pentes variees: fond de vallee + versants + arete."""
    rng = np.random.default_rng(seed)
    rows, cols = size, size
    # forme en V: vallee au centre, pentes de chaque cote
    x = np.linspace(-1, 1, cols)
    profile = 2500 + 500 * np.abs(x)  # vallee a 2500, cretes a 3000
    dem = np.broadcast_to(profile, (rows, cols)).copy().astype(np.float32)
    # gradient N-S aussi
    ns = np.linspace(0, 300, rows).reshape(-1, 1)
    dem += ns
    # bruit
    dem += rng.normal(0, 15, dem.shape).astype(np.float32)
    return dem


def bench_terrain(dem, resolution=1.0):
    t0 = time.time()
    slope, aspect = compute_slope_aspect(dem, resolution)
    dt_slope = time.time() - t0

    t0 = time.time()
    roughness = compute_roughness(dem)
    dt_rough = time.time() - t0

    return slope, aspect, roughness, dt_slope + dt_rough


def bench_cost(dem, slope, aspect, roughness):
    t0 = time.time()
    cost, factors, nodata = build_cost_surface(
        dem, slope, aspect, roughness, glacier_mask=None)
    dt = time.time() - t0
    return cost, dt


def bench_pathfinding(cost):
    grid = prepare_cost_grid(cost)
    n = cost.shape[0]
    start = (1, 1)
    end = (n - 2, n - 2)
    t0 = time.time()
    path, path_cost, _ = run_pathfinding(grid, start, end)
    dt = time.time() - t0
    return len(path), dt


def bench_pathfinding_aniso(dem, slope, aspect, roughness, resolution=1.0):
    """Benchmark dijkstra anisotrope."""
    base_cost, _ = build_base_cost(
        dem, slope, aspect, roughness, glacier_mask=None)
    n = dem.shape[0]
    start = (1, 1)
    end = (n - 2, n - 2)
    t0 = time.time()
    path, path_cost, _ = dijkstra_anisotropic(
        dem, base_cost, start, end, resolution)
    dt = time.time() - t0
    return len(path), dt


def run_bench(size, include_aniso=False, dem_func=make_synthetic_dem):
    tracemalloc.start()

    dem = dem_func(size)
    slope, aspect, roughness, dt_terrain = bench_terrain(dem)
    cost, dt_cost = bench_cost(dem, slope, aspect, roughness)
    n_pts, dt_path = bench_pathfinding(cost)

    result = {
        "size": size,
        "terrain_s": round(dt_terrain, 2),
        "cost_s": round(dt_cost, 2),
        "path_iso_s": round(dt_path, 2),
        "path_iso_pts": n_pts,
    }

    if include_aniso:
        try:
            n_pts_a, dt_a = bench_pathfinding_aniso(
                dem, slope, aspect, roughness)
            result["path_aniso_s"] = round(dt_a, 2)
            result["path_aniso_pts"] = n_pts_a
        except Exception as e:
            result["path_aniso_s"] = None
            result["path_aniso_error"] = str(e)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result["peak_mb"] = round(peak / 1024 / 1024, 1)

    return result


if __name__ == "__main__":
    json_mode = "--json" in sys.argv

    # grilles standard
    sizes = [1000, 5000, 10000]
    # aniso seulement sur petites grilles (RAM)
    aniso_max = 2000
    results = []

    if not json_mode:
        print(f"{'size':>8} | {'terrain':>8} | {'cost':>8} | {'iso':>9} | {'aniso':>9} | {'pts':>6} | {'peak MB':>8}")
        print("-" * 78)

    for s in sizes:
        if not json_mode:
            print(f"  running {s}x{s}...", flush=True)
        try:
            r = run_bench(s, include_aniso=(s <= aniso_max))
            results.append(r)
            if not json_mode:
                aniso_str = f"{r['path_aniso_s']:>8.1f}s" if r.get("path_aniso_s") else "     -  "
                print(f"{r['size']:>8} | {r['terrain_s']:>7.1f}s | {r['cost_s']:>7.1f}s | {r['path_iso_s']:>8.1f}s | {aniso_str} | {r['path_iso_pts']:>6} | {r['peak_mb']:>7.0f}")
        except Exception as e:
            if not json_mode:
                print(f"{s:>8} | FAILED: {e}")

    # scenario "Chamonix" avec pentes variees
    if not json_mode:
        print("\n--- Chamonix (pentes variees) ---")
    for s in [1000, 3000]:
        if not json_mode:
            print(f"  running chamonix {s}x{s}...", flush=True)
        try:
            r = run_bench(s, include_aniso=(s <= aniso_max),
                          dem_func=make_chamonix_dem)
            r["scenario"] = "chamonix"
            results.append(r)
            if not json_mode:
                aniso_str = f"{r['path_aniso_s']:>8.1f}s" if r.get("path_aniso_s") else "     -  "
                print(f"{r['size']:>8} | {r['terrain_s']:>7.1f}s | {r['cost_s']:>7.1f}s | {r['path_iso_s']:>8.1f}s | {aniso_str} | {r['path_iso_pts']:>6} | {r['peak_mb']:>7.0f}")
        except Exception as e:
            if not json_mode:
                print(f"{s:>8} | FAILED: {e}")

    # verif basique perf
    std_results = [r for r in results if r.get("scenario") is None]
    if std_results and std_results[-1]["size"] == 10000:
        dt = std_results[-1]["path_iso_s"]
        if not json_mode:
            if dt > 60:
                print(f"\nWARNING: pathfinding 10k trop lent ({dt:.0f}s > 60s)")
            else:
                print(f"\nOK: 10k pathfinding en {dt:.0f}s")

    if json_mode:
        import json
        print(json.dumps(results, indent=2))
