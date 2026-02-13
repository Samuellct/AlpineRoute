# T05 - test Dijkstra
# route Refuge du Requin -> col du Midi avec skimage

import os
import sys
import time
import json
import numpy as np
import rasterio
from pyproj import Transformer
from skimage.graph import route_through_array
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LightSource

from config import (
    DEM_DIR, DERIVED_DIR, FIGURES_DIR, MAPS_DIR,
    DEM_RESOLUTION, NODATA_VALUE, CRS_L93, CRS_WGS84,
    START_POINT_WGS84, END_POINT_WGS84, UHD_DPI,
)


# ==============================================
#  Chargement des donnees
# ==============================================

def load_cost_surface():
    path = os.path.join(DERIVED_DIR, f"cost_surface_{DEM_RESOLUTION}m.tif")
    if not os.path.exists(path):
        print(f"[error] cost surface introuvable: {path}")
        print("  -> lancer T04 d'abord")
        sys.exit(1)
    with rasterio.open(path) as ds:
        cost = ds.read(1).astype(np.float64)
        transform = ds.transform
        profile = ds.profile.copy()
    return cost, transform, profile


def load_dem():
    path = os.path.join(DEM_DIR, f"dem_aiguille_du_midi_{DEM_RESOLUTION}m.tif")
    if not os.path.exists(path):
        print(f"[error] DEM introuvable: {path}")
        sys.exit(1)
    with rasterio.open(path) as ds:
        dem = ds.read(1).astype(np.float32)
    return dem


def load_glacier_mask():
    path = os.path.join(DERIVED_DIR, f"glacier_mask_{DEM_RESOLUTION}m.tif")
    if not os.path.exists(path):
        print("[warn] pas de masque glacier")
        return None
    with rasterio.open(path) as ds:
        return ds.read(1).astype(bool)


# ==============================================
#  Conversion coords WGS84 -> pixel
# ==============================================

def wgs84_to_pixel(lat, lon, transform, shape):
    """Convertit lat/lon WGS84 en row/col pixel sur la grille L93."""
    # pyproj always_xy=True => (lon, lat) -> (x, y)
    proj = Transformer.from_crs(CRS_WGS84, CRS_L93, always_xy=True)
    x_l93, y_l93 = proj.transform(lon, lat)

    # inverse transform: coords L93 -> pixel
    col, row = ~transform * (x_l93, y_l93)
    row, col = int(round(row)), int(round(col))

    print(f"  WGS84: ({lat:.6f}, {lon:.6f})")
    print(f"  L93:   ({x_l93:.2f}, {y_l93:.2f})")
    print(f"  pixel: (row={row}, col={col})")

    # check bornes
    h, w = shape
    if not (0 <= row < h and 0 <= col < w):
        print(f"  [FAIL] pixel hors grille ({h}x{w})")
        sys.exit(1)

    return row, col, x_l93, y_l93


# ==============================================
#  Pathfinding
# ==============================================

def run_pathfinding(cost, start_rc, end_rc):
    print(f"\n  start pixel: {start_rc}")
    print(f"  end pixel:   {end_rc}")
    print(f"  grille: {cost.shape[0]}x{cost.shape[1]} = {cost.size:,} pixels")

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
    print(f"  temps: {dt:.1f}s")
    print(f"  pixels dans le path: {len(path_coords)}")
    print(f"  cout total: {path_cost:.2f}")

    return path_coords, path_cost, dt


# ==============================================
#  Stats
# ==============================================

def compute_path_stats(path_coords, dem, cost, glacier_mask, transform):
    rows = path_coords[:, 0]
    cols = path_coords[:, 1]

    elevations = dem[rows, cols]
    costs_along = cost[rows, cols]

    # coords L93 de chaque pixel du path
    xs = np.array([transform * (c, r) for r, c in zip(rows, cols)])
    x_l93 = xs[:, 0]
    y_l93 = xs[:, 1]

    # distance
    dx = np.diff(x_l93)
    dy = np.diff(y_l93)
    dz = np.diff(elevations)
    seg_dist_2d = np.sqrt(dx**2 + dy**2)
    seg_dist_3d = np.sqrt(dx**2 + dy**2 + dz**2)

    dist_2d = np.sum(seg_dist_2d)
    dist_3d = np.sum(seg_dist_3d)
    cum_dist = np.concatenate([[0], np.cumsum(seg_dist_2d)])

    # D+ / D-
    dplus = np.sum(dz[dz > 0])
    dminus = np.sum(dz[dz < 0])

    # pente locale (degres) pour chaque segment
    slopes = np.degrees(np.arctan2(np.abs(dz), seg_dist_2d))
    # evite div/0 si 2 pixels identiques (rare)
    slopes = np.where(np.isnan(slopes), 0, slopes)

    # estimation Tobler pour chaque segment
    # gradient = dz/dx (signe : positif = montee)
    gradient = np.where(seg_dist_2d > 0, dz / seg_dist_2d, 0)
    v_tobler = 6.0 * np.exp(-3.5 * np.abs(gradient + 0.05)) * 0.6  # hors-piste
    v_tobler = np.maximum(v_tobler, 0.01)  # conv km/h
    seg_time_h = (seg_dist_2d / 1000.0) / v_tobler
    total_time_h = np.sum(seg_time_h)

    # % sur glacier
    if glacier_mask is not None:
        on_glacier = glacier_mask[rows, cols]
        glacier_pct = on_glacier.sum() / len(rows) * 100
    else:
        glacier_pct = -1
        on_glacier = np.zeros(len(rows), dtype=bool)

    stats = {
        'n_pixels': len(path_coords),
        'dist_2d_m': dist_2d,
        'dist_3d_m': dist_3d,
        'dplus': dplus,
        'dminus': abs(dminus),
        'elev_start': elevations[0],
        'elev_end': elevations[-1],
        'elev_min': elevations.min(),
        'elev_max': elevations.max(),
        'cost_total': costs_along.sum(),
        'cost_mean': costs_along.mean(),
        'cost_median': np.median(costs_along),
        'glacier_pct': glacier_pct,
        'time_tobler_h': total_time_h,
    }

    # debug
    print("\n--- Stats du trajet ---")
    print(f"  pixels:       {stats['n_pixels']:,}")
    print(f"  distance 2D:  {stats['dist_2d_m']:.0f} m ({stats['dist_2d_m']/1000:.2f} km)")
    print(f"  distance 3D:  {stats['dist_3d_m']:.0f} m ({stats['dist_3d_m']/1000:.2f} km)")
    print(f"  D+:           {stats['dplus']:.0f} m")
    print(f"  D-:           {stats['dminus']:.0f} m")
    print(f"  altitude:     {stats['elev_start']:.0f}m (depart) -> {stats['elev_end']:.0f}m (arrivee)")
    print(f"  alt min/max:  {stats['elev_min']:.0f}m / {stats['elev_max']:.0f}m")
    print(f"  cout moyen:   {stats['cost_mean']:.2f}")
    print(f"  cout median:  {stats['cost_median']:.2f}")
    print(f"  glacier:      {stats['glacier_pct']:.1f}%")
    print(f"  temps Tobler: {stats['time_tobler_h']:.1f}h ({stats['time_tobler_h']*60:.0f} min)")

    return stats, cum_dist, elevations, slopes, x_l93, y_l93


# ==============================================
#  Export GeoJSON
# ==============================================

def export_geojson(x_l93, y_l93, stats):
    os.makedirs(MAPS_DIR, exist_ok=True)
    path = os.path.join(MAPS_DIR, f"route_requin_to_aiguille_{DEM_RESOLUTION}m.geojson")

    # coordonnees en L93 (pas besoin de re-transformer)
    coords = [[float(x), float(y)] for x, y in zip(x_l93, y_l93)]

    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
            "properties": {
                "name": "Refuge du Requin -> col du Midi",
                "distance_m": round(stats['dist_2d_m']),
                "distance_km": round(stats['dist_2d_m'] / 1000, 2),
                "dplus_m": round(stats['dplus']),
                "dminus_m": round(stats['dminus']),
                "cost_total": round(stats['cost_total'], 1),
                "time_tobler_h": round(stats['time_tobler_h'], 1),
                "glacier_pct": round(stats['glacier_pct'], 1),
                "resolution_m": DEM_RESOLUTION,
                "crs": CRS_L93,
            },
        }],
    }

    with open(path, 'w') as f:
        json.dump(geojson, f, indent=2)

    size_kb = os.path.getsize(path) / 1024
    print(f"[export] {path} ({size_kb:.0f} KB)")
    return path


# ==============================================
#  Plot
# ==============================================

def _make_hillshade(dem):
    dem_display = np.where(dem == NODATA_VALUE, np.nan, dem)
    dem_filled = np.where(np.isnan(dem_display), 0, dem_display)
    ls = LightSource(azdeg=315, altdeg=45)
    return ls.hillshade(dem_filled, vert_exag=2,
                        dx=DEM_RESOLUTION, dy=DEM_RESOLUTION)


def plot_route_map(path_coords, cost, dem, start_rc, end_rc, stats):
    """Fig 1 : vue globale du trace sur fond hillshade + cost."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    hillshade = _make_hillshade(dem)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.imshow(hillshade, cmap='gray', alpha=0.5)

    # cost surface en fond
    nodata_mask = (cost == np.inf) | (cost >= 1e6)
    cost_display = np.where(nodata_mask, np.nan, cost)
    ax.imshow(cost_display, cmap='RdYlGn_r', norm=LogNorm(vmin=1, vmax=500),
              alpha=0.4)

    rows = path_coords[:, 0]
    cols = path_coords[:, 1]
    ax.plot(cols, rows, color='cyan', linewidth=2.5, alpha=0.9, label='Route optimale')

    ax.plot(start_rc[1], start_rc[0], 'o', color='#2ecc71', markersize=12,
            markeredgecolor='white', markeredgewidth=2, zorder=10, label='Depart (Requin)')
    ax.plot(end_rc[1], end_rc[0], 's', color='#e74c3c', markersize=12,
            markeredgecolor='white', markeredgewidth=2, zorder=10, label='Arrivee (Aig. du Midi)')

    dist_km = stats['dist_2d_m'] / 1000
    ax.set_title(f"Route optimale Requin -> col du Midi ({DEM_RESOLUTION}m)\n"
                 f"dist={dist_km:.1f} km, D+={stats['dplus']:.0f}m, "
                 f"temps~{stats['time_tobler_h']:.1f}h", fontsize=11)
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlabel('col (px)')
    ax.set_ylabel('row (px)')

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "route_on_map.png")
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    uhd_dir = os.path.join(FIGURES_DIR, "uhd")
    os.makedirs(uhd_dir, exist_ok=True)
    fig.savefig(os.path.join(uhd_dir, "route_on_map.pdf"), dpi=UHD_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[plot] {out}")


def plot_elevation_profile(cum_dist, elevations, slopes, stats):
    """Fig 2 : profil altimetrique + pente locale."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[3, 1],
                                    sharex=True)

    dist_km = cum_dist / 1000.0

    # profil altitude
    ax1.fill_between(dist_km, elevations, elevations.min() - 50,
                     color='#3498db', alpha=0.3)
    ax1.plot(dist_km, elevations, color='#2c3e50', linewidth=1.5)

    ax1.set_ylabel('Altitude (m)')
    ax1.set_title(f"Profil altimetrique - Requin -> col du Midi\n"
                  f"D+={stats['dplus']:.0f}m, D-={stats['dminus']:.0f}m", fontsize=11)
    ax1.grid(True, alpha=0.3)

    # pts depart/arrivee
    ax1.annotate(f"Depart\n{elevations[0]:.0f}m", xy=(dist_km[0], elevations[0]),
                 fontsize=8, ha='left', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#2ecc71', alpha=0.7))
    ax1.annotate(f"Arrivee\n{elevations[-1]:.0f}m", xy=(dist_km[-1], elevations[-1]),
                 fontsize=8, ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#e74c3c', alpha=0.7))

    # point le plus haut
    idx_max = np.argmax(elevations)
    ax1.annotate(f"Max\n{elevations[idx_max]:.0f}m", xy=(dist_km[idx_max], elevations[idx_max]),
                 fontsize=8, ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7))

    # pente locale
    # on a len(slopes) = len(dist_km) - 1, on prend les midpoints
    dist_mid = (dist_km[:-1] + dist_km[1:]) / 2
    ax2.fill_between(dist_mid, slopes, color='#e67e22', alpha=0.4)
    ax2.plot(dist_mid, slopes, color='#d35400', linewidth=0.8)
    ax2.set_ylabel('Pente locale (deg)')
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylim(0, min(slopes.max() * 1.2, 80))
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=30, color='red', ls='--', lw=0.8, alpha=0.5, label='30 deg')
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "route_elevation_profile.png")
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    uhd_dir = os.path.join(FIGURES_DIR, "uhd")
    os.makedirs(uhd_dir, exist_ok=True)
    fig.savefig(os.path.join(uhd_dir, "route_elevation_profile.pdf"), dpi=UHD_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[plot] {out}")


def plot_detail_zones(path_coords, cost, dem, start_rc, end_rc):
    """Fig 3 : zoom depart + arrivee."""
    hillshade = _make_hillshade(dem)
    win = 250

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    rows = path_coords[:, 0]
    cols = path_coords[:, 1]

    nodata_mask = (cost == np.inf) | (cost >= 1e6)
    cost_display = np.where(nodata_mask, np.nan, cost)

    for ax, (rc, title, color) in zip(
        [ax1, ax2],
        [(start_rc, "Depart - Refuge du Requin", '#2ecc71'),
         (end_rc, "Arrivee - col du Midi", '#e74c3c')]
    ):
        r, c = rc
        r_lo = max(0, r - win)
        r_hi = min(dem.shape[0], r + win)
        c_lo = max(0, c - win)
        c_hi = min(dem.shape[1], c + win)

        ax.imshow(hillshade[r_lo:r_hi, c_lo:c_hi], cmap='gray', alpha=0.5)
        ax.imshow(cost_display[r_lo:r_hi, c_lo:c_hi], cmap='RdYlGn_r',
                  norm=LogNorm(vmin=1, vmax=500), alpha=0.5)

        mask = (rows >= r_lo) & (rows < r_hi) & (cols >= c_lo) & (cols < c_hi)
        ax.plot(cols[mask] - c_lo, rows[mask] - r_lo,
                color='cyan', linewidth=2.5, alpha=0.9)

        ax.plot(c - c_lo, r - r_lo, 'o', color=color, markersize=14,
                markeredgecolor='white', markeredgewidth=2, zorder=10)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f'col ({c_lo}-{c_hi})')
        ax.set_ylabel(f'row ({r_lo}-{r_hi})')

    fig.suptitle(f"Zoom depart/arrivee - {DEM_RESOLUTION}m", fontsize=12, y=1.01)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "route_detail_zones.png")
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    uhd_dir = os.path.join(FIGURES_DIR, "uhd")
    os.makedirs(uhd_dir, exist_ok=True)
    fig.savefig(os.path.join(uhd_dir, "route_detail_zones.pdf"), dpi=UHD_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[plot] {out}")


# ==============================================
#  Validation du path
# ==============================================

def validate_path(path_coords, cost, start_rc, end_rc, stats):
    print("\n--- Validation ---")
    ok = True

    # path non vide
    if len(path_coords) == 0:
        print("  [FAIL] path vide!")
        return False

    # commence/finit aux bons pixels
    if tuple(path_coords[0]) != start_rc:
        print(f"  [FAIL] debut path {tuple(path_coords[0])} != start {start_rc}")
        ok = False
    else:
        print("  [OK] debut du path correct")

    if tuple(path_coords[-1]) != end_rc:
        print(f"  [FAIL] fin path {tuple(path_coords[-1])} != end {end_rc}")
        ok = False
    else:
        print("  [OK] fin du path correcte")

    # continuite : chaque step a max 1px de distance (8-connexe)
    diffs = np.abs(np.diff(path_coords, axis=0))
    max_step = diffs.max()
    if max_step > 1:
        bad = np.where(diffs.max(axis=1) > 1)[0]
        print(f"  [FAIL] {len(bad)} sauts > 1px (max step={max_step})")
        ok = False
    else:
        print("  [OK] path continu (8-connexe)")

    # aucun pixel sur inf
    rows = path_coords[:, 0]
    cols = path_coords[:, 1]
    costs_on_path = cost[rows, cols]
    n_inf = np.sum(np.isinf(costs_on_path))
    if n_inf > 0:
        print(f"  [WARN] {n_inf} pixels du path sur nodata (inf)")
    else:
        print("  [OK] aucun pixel sur nodata")

    # distance plausible (debug de test avec distance réelle connue)
    dist_km = stats['dist_2d_m'] / 1000
    if 3 < dist_km < 20:
        print(f"  [OK] distance ok: {dist_km:.1f} km")
    else:
        print(f"  [WARN] distance inattendue: {dist_km:.1f} km (attendu 5-15 km)")

    # D+ plausible (Requin 2300m -> Midi 3800m, donc 1500m min + detours)
    if 500 < stats['dplus'] < 3000:
        print(f"  [OK] D+ ok: {stats['dplus']:.0f} m")
    else:
        print(f"  [WARN] D+ inattendue: {stats['dplus']:.0f} m (attendu +- 1500m)")

    return ok


# ==============================================

def main():
    t_total = time.time()

    print("=" * 55)
    print("T05 - Pathfinding (Dijkstra)")
    print(f"  resolution: {DEM_RESOLUTION}m")
    print(f"  start: {START_POINT_WGS84}  (Refuge du Requin)")
    print(f"  end:   {END_POINT_WGS84}  (col du Midi)")
    print("=" * 55)

    print("\n--- Chargement cost surface ---")
    t0 = time.time()
    cost, transform, profile = load_cost_surface()
    print(f"  shape: {cost.shape}")
    print(f"  chargement: {time.time()-t0:.1f}s")

    nodata_mask = (cost == NODATA_VALUE)
    cost[nodata_mask] = np.inf
    # cap a 1e6 pour eviter overflow
    cost = np.clip(cost, 0, 1e6)
    cost[nodata_mask] = np.inf

    valid = cost[~np.isinf(cost)]
    print(f"  valid pixels: {len(valid):,} / {cost.size:,}")
    print(f"  cout: min={valid.min():.3f}, max={valid.max():.1f}, median={np.median(valid):.2f}")

    print("\n--- Chargement DEM ---")
    dem = load_dem()

    print("\n--- Chargement masque glacier ---")
    glacier_mask = load_glacier_mask()
    if glacier_mask is not None:
        print(f"  couverture: {glacier_mask.sum()/glacier_mask.size*100:.1f}%")

    print("\n--- Conversion start ---")
    start_row, start_col, _, _ = wgs84_to_pixel(
        START_POINT_WGS84[0], START_POINT_WGS84[1], transform, cost.shape)
    print(f"  elevation: {dem[start_row, start_col]:.1f}m")
    print(f"  cout:      {cost[start_row, start_col]:.3f}")

    print("\n--- Conversion end ---")
    end_row, end_col, _, _ = wgs84_to_pixel(
        END_POINT_WGS84[0], END_POINT_WGS84[1], transform, cost.shape)
    print(f"  elevation: {dem[end_row, end_col]:.1f}m")
    print(f"  cout:      {cost[end_row, end_col]:.3f}")

    start_rc = (start_row, start_col)
    end_rc = (end_row, end_col)

    print("\n--- Dijkstra (route_through_array) ---")
    path_coords, path_cost, dt_pathfind = run_pathfinding(cost, start_rc, end_rc)

    stats, cum_dist, elevations, slopes, x_l93, y_l93 = compute_path_stats(
        path_coords, dem, cost, glacier_mask, transform)

    print("\n--- Export GeoJSON ---")
    export_geojson(x_l93, y_l93, stats)

    print("\n--- Visualisations ---")
    plot_route_map(path_coords, cost, dem, start_rc, end_rc, stats)
    plot_elevation_profile(cum_dist, elevations, slopes, stats)
    plot_detail_zones(path_coords, cost, dem, start_rc, end_rc)
    validate_path(path_coords, cost, start_rc, end_rc, stats)

    dt_total = time.time() - t_total
    print(f"\nDone! ({dt_total:.1f}s total, dont {dt_pathfind:.1f}s pathfinding)")


if __name__ == "__main__":
    main()
