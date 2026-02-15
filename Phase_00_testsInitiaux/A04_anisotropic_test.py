# A04 - test pathfinding avec MCP_Flexible

import os
import sys
import time
import numpy as np
from skimage.graph import MCP_Geometric, route_through_array
import rasterio

from config import (
    DEM_DIR, DERIVED_DIR, DEM_RESOLUTION, NODATA_VALUE,
    CRS_L93, CRS_WGS84, START_POINT_WGS84, END_POINT_WGS84,
)


# =====================================================
#  Test 1 : DEM synthetique (plan incline)
# =====================================================

def test_synthetic():
    """Plan incline 30x30, pente ~15 deg vers le bas (y croissant).
    Montee = aller vers y=0, descente = aller vers y=29.
    En anisotrope, le path devrait eviter de monter droit."""

    print("\n--- Test 1 : DEM synthetique ---")
    size = 30
    # plan incline: altitude decroit de haut en bas
    elevation = np.zeros((size, size), dtype=np.float64)
    for r in range(size):
        elevation[r, :] = 3000 - r * 10  # 3000m en haut, 2710m en bas

    res = 1.0  # 1m
    res_synth = 40.0

    # calcul de la pente pour info
    grad = 10.0 / res_synth
    slope_deg = np.degrees(np.arctan(grad))
    print(f"  grille: {size}x{size}, resolution {res_synth}m")
    print(f"  gradient: {grad:.3f}, pente: {slope_deg:.1f} deg")

    # --- isotrope : surface de cout basee sur pente abs ---
    # Tobler isotrope : v = 6 * exp(-3.5 * |grad + 0.05|) * 0.6
    v_flat = 6.0 * np.exp(-3.5 * 0.05) * 0.6
    v_iso = 6.0 * np.exp(-3.5 * abs(grad + 0.05)) * 0.6
    cost_iso = np.full((size, size), v_flat / max(v_iso, 0.01), dtype=np.float64)

    start = (0, 0)     # haut-gauche (altitude haute)
    end = (29, 29)      # bas-droite (altitude basse)

    print(f"\n  Isotrope (route_through_array):")
    path_iso, cost_val_iso = route_through_array(
        cost_iso, start=start, end=end, fully_connected=True, geometric=True)
    path_iso = np.array(path_iso)
    print(f"    path: {len(path_iso)} pixels, cout={cost_val_iso:.2f}")
    print(f"    trajet: ligne droite ?" if np.allclose(path_iso[:, 0] - path_iso[:, 1], 0) else "    trajet: pas en diagonale pure")

    # --- anisotrope : MCP_Geometric avec offsets manuels ---
    # on pre-calcule 8 surfaces de cout, une par direction : (dr, dc) avec dr,dc in {-1,0,1}
    print(f"\n  Anisotrope (8 surfaces de cost):")

    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    # pour chaque offset on calcule le gradient signe
    # dr negatif = on va vers le haut = montee (altitude augmente)
    # dr positif = on va vers le bas = descente (altitude diminue)

    from skimage.graph import MCP_Flexible

    class AnisotropicMCP(MCP_Flexible):
        """MCP avec cout Tobler signe (montee != descente)."""

        def __init__(self, elev, cell_size, *args, **kwargs):
            # le "costs" pour MCP doit etre >0 partout
            placeholder = np.ones_like(elev, dtype=np.float64)
            super().__init__(placeholder, *args, **kwargs)
            self.elev = elev
            self.cell_size = cell_size
            self.v_flat = 6.0 * np.exp(-3.5 * 0.05) * 0.6

        def _travel_cost(self, old_cost, new_cost, offset_length):
            return old_cost + new_cost

    cost_arrays = {}
    for dr, dc in offsets:
        dist_2d = np.sqrt((dr * res_synth)**2 + (dc * res_synth)**2)
        # dz pour aller de (r,c) a (r+dr, c+dc)
        # elevation[r+dr,c+dc] - elevation[r,c]
        # pour simplifier: dz = dr * (-10) car altitude = 3000 - r*10
        dz = dr * (-10.0)  # dr>0 -> on descend -> dz<0
        gradient_signed = dz / dist_2d
        # Tobler signe: v = 6 * exp(-3.5 * (gradient + 0.05))
        v = 6.0 * np.exp(-3.5 * (gradient_signed + 0.05)) * 0.6
        v = max(v, 0.01)
        cost_val = (v_flat / v) * dist_2d
        cost_arrays[(dr, dc)] = cost_val

    print(f"    couts par direction (plan incline):")
    for (dr, dc), c in sorted(cost_arrays.items()):
        direction = ""
        if dr < 0: direction += "haut"
        elif dr > 0: direction += "bas"
        if dc < 0: direction += "-gauche"
        elif dc > 0: direction += "-droite"
        else:
            if not direction: direction = "neutre"
        print(f"      ({dr:+d},{dc:+d}) {direction:>12}: {c:.2f}")

    # le cout de descente (dr=+1) doit etre < cout de montee (dr=-1)
    cost_down = cost_arrays[(1, 0)]
    cost_up = cost_arrays[(-1, 0)]
    print(f"\n    ratio montee/descente: {cost_up/cost_down:.2f}x")
    assert cost_up > cost_down, "Tobler signe doit penaliser la montee plus que la descente"
    print("    [OK] montee bien plus chere que descente")

    return True


# =====================================================
#  Test 2 : MCP_Geometric sur le vrai DEM
# =====================================================

def test_real_dem():
    """Benchmark MCP_Geometric vs route_through_array sur le vrai DEM."""
    print("\n--- Test 2 : Benchmark sur le vrai DEM ---")

    # charge la cost surface
    cost_path = os.path.join(DERIVED_DIR, f"cost_surface_{DEM_RESOLUTION}m.tif")
    if not os.path.exists(cost_path):
        print("  [skip] cost surface introuvable, lancer T04")
        return False

    with rasterio.open(cost_path) as ds:
        cost = ds.read(1).astype(np.float64)
        transform = ds.transform

    # prep nodata
    nodata_mask = (cost == NODATA_VALUE)
    cost[nodata_mask] = np.inf
    cost = np.clip(cost, 0.001, 1e6)
    cost[nodata_mask] = np.inf

    # coords -> pixels
    from pyproj import Transformer
    proj = Transformer.from_crs(CRS_WGS84, CRS_L93, always_xy=True)

    sx, sy = proj.transform(START_POINT_WGS84[1], START_POINT_WGS84[0])
    sc, sr = ~transform * (sx, sy)
    start_rc = (int(round(sr)), int(round(sc)))

    ex, ey = proj.transform(END_POINT_WGS84[1], END_POINT_WGS84[0])
    ec, er = ~transform * (ex, ey)
    end_rc = (int(round(er)), int(round(ec)))

    print(f"  grille: {cost.shape} ({cost.size:,} pixels)")
    print(f"  start: {start_rc}, end: {end_rc}")

    # --- route_through_array (baseline) ---
    print(f"\n  route_through_array (isotrope):")
    t0 = time.time()
    path1, cost1 = route_through_array(
        cost, start=start_rc, end=end_rc,
        fully_connected=True, geometric=True)
    dt1 = time.time() - t0
    path1 = np.array(path1)
    print(f"    temps: {dt1:.2f}s, path: {len(path1)} px, cout: {cost1:.1f}")

    # --- MCP_Geometric (equivalent, mais plus flexible) ---
    print(f"\n  MCP_Geometric (meme chose mais via la classe):")
    t0 = time.time()
    mcp = MCP_Geometric(cost, fully_connected=True)
    cum_costs, traceback = mcp.find_costs([start_rc])
    dt2_costs = time.time() - t0

    t0 = time.time()
    path2 = mcp.traceback(end_rc)
    dt2_trace = time.time() - t0
    path2 = np.array(path2)
    cost2 = cum_costs[end_rc]

    print(f"    find_costs: {dt2_costs:.2f}s")
    print(f"    traceback:  {dt2_trace:.3f}s")
    print(f"    total:      {dt2_costs + dt2_trace:.2f}s")
    print(f"    path: {len(path2)} px, cout: {cost2:.1f}")

    # comparaison
    print(f"\n  Comparaison:")
    print(f"    route_through_array: {dt1:.2f}s")
    print(f"    MCP_Geometric:       {dt2_costs + dt2_trace:.2f}s")
    print(f"    ratio: {(dt2_costs + dt2_trace) / dt1:.2f}x")
    print(f"    couts: {cost1:.1f} vs {cost2:.1f} (diff={abs(cost1-cost2):.2f})")

    # avantage MCP: on a deja calcule les couts vers TOUS les pixels
    # donc si on veut plusieurs destinations c'est quasi gratuit
    print(f"\n  Avantage MCP: cum_costs disponible pour TOUTE la grille")
    print(f"    -> routes alternatives quasi-gratuites (traceback depuis n'importe quel point)")

    return True


# =====================================================
#  Test 3 : simulation anisotrope par double passe
# =====================================================

def test_double_pass():
    """Approche anisotrope simplifiee: deux cost surfaces (montee vs descente)
    puis selection du meilleur chemin."""
    print("\n--- Test 3 : Approche double-passe (proto anisotrope) ---")

    dem_path = os.path.join(DEM_DIR, f"dem_aiguille_du_midi_{DEM_RESOLUTION}m.tif")
    if not os.path.exists(dem_path):
        print("  [skip] DEM manquant")
        return False

    with rasterio.open(dem_path) as ds:
        dem = ds.read(1).astype(np.float64)
        transform = ds.transform

    # charge slope
    slope_path = os.path.join(DERIVED_DIR, f"slope_{DEM_RESOLUTION}m.tif")
    with rasterio.open(slope_path) as ds:
        slope = ds.read(1).astype(np.float64)

    nodata_mask = (slope == NODATA_VALUE) | np.isnan(slope)

    # --- surface de cout isotrope (comme T04) ---
    slope_clean = np.where(nodata_mask, 0, slope)
    slope_rad = np.radians(np.clip(slope_clean, 0, 89.9))
    gradient = np.tan(slope_rad)
    v_flat = 6.0 * np.exp(-3.5 * 0.05) * 0.6
    v_iso = 6.0 * np.exp(-3.5 * np.abs(gradient + 0.05)) * 0.6
    v_iso = np.maximum(v_iso, 0.01)
    cost_iso = v_flat / v_iso
    cost_iso[nodata_mask] = np.inf

    # --- surface "montee" : penalise les pentes montantes ---
    # approx: on multiplie le cout par un facteur selon l'altitude relative
    # les pixels a haute altitude sont plus chers (car on monte pour y arriver)
    alt_norm = (dem - dem[~nodata_mask].min()) / max(dem[~nodata_mask].max() - dem[~nodata_mask].min(), 1)
    alt_norm = np.clip(alt_norm, 0, 1)
    cost_uphill = cost_iso * (1 + 0.5 * alt_norm)  # max 50% plus cher en altitude
    cost_uphill[nodata_mask] = np.inf

    # --- surface "descente" : penalise les descentes raides ---
    cost_downhill = cost_iso * (1 + 0.3 * (1 - alt_norm))
    cost_downhill[nodata_mask] = np.inf

    # coords
    from pyproj import Transformer
    proj = Transformer.from_crs(CRS_WGS84, CRS_L93, always_xy=True)
    sx, sy = proj.transform(START_POINT_WGS84[1], START_POINT_WGS84[0])
    sc, sr = ~transform * (sx, sy)
    start_rc = (int(round(sr)), int(round(sc)))
    ex, ey = proj.transform(END_POINT_WGS84[1], END_POINT_WGS84[0])
    ec, er = ~transform * (ex, ey)
    end_rc = (int(round(er)), int(round(ec)))

    # comparer les 3 paths
    for name, c in [("isotrope", cost_iso), ("uphill_bias", cost_uphill), ("downhill_bias", cost_downhill)]:
        t0 = time.time()
        path, path_cost = route_through_array(
            c, start=start_rc, end=end_rc,
            fully_connected=True, geometric=True)
        dt = time.time() - t0
        path = np.array(path)

        # stats elevation
        elevs = dem[path[:, 0], path[:, 1]]
        dz = np.diff(elevs)
        dplus = np.sum(dz[dz > 0])
        dminus = abs(np.sum(dz[dz < 0]))

        print(f"\n  {name}:")
        print(f"    temps: {dt:.2f}s, path: {len(path)} px, cout: {path_cost:.1f}")
        print(f"    D+={dplus:.0f}m, D-={dminus:.0f}m")

    print("\n  Note: l'approche double-passe est une approximation grossiere.")
    print("  Pour un vrai anisotrope il faudrait integrer la direction dans le graphe.")

    return True


# =====================================================

def main():
    print("=" * 60)
    print("A04 - Pathfinding anisotrope")
    print("=" * 60)

    test_synthetic()
    test_real_dem()
    test_double_pass()

    # conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
  1. MCP_Flexible._travel_cost ne fournit PAS les coordonnees du pixel
     -> impossible de lire l'elevation pour calculer le gradient signe
     -> sous-classement MCP_Flexible n'est PAS viable pour l'anisotrope

  2. MCP_Geometric est equivalent a route_through_array en perf/resultats
     mais offre find_costs() qui calcule les couts vers TOUTE la grille
     -> utile pour routes alternatives (traceback depuis plusieurs endpoints)

  3. Pour un vrai pathfinding anisotrope (montee != descente):
     - Option A : 8 cost surfaces pre-calculees (une par direction 8-connexe)
       + MCP custom qui selectionne la bonne surface selon l'offset
       Faisable mais il faut ecrire un MCP custom en Cython/Rust
     - Option B : graphe oriente explicite (NetworkX/custom)
       Trop lent pour 36M de noeuds
     - Option C : A* anisotrope en Cython
       Meilleure option pour V1, mais effort de dev significatif
     - Option D : approximation par bias altimetrique (test 3)
       Rapide a implementer, resultats approximatifs
    """)

    print("Done!")


if __name__ == "__main__":
    main()
