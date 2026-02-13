# T02 - analyse de terrain avec pente, aspect, rugosite

import os
import sys
import numpy as np
from scipy.ndimage import convolve, binary_dilation
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (
    DEM_DIR, DERIVED_DIR, FIGURES_DIR, DEM_RESOLUTION, NODATA_VALUE, CRS_L93,
    UHD_DPI,
)


# =====================================================
#  Chargement DEM
# =====================================================

def load_dem():
    path = os.path.join(DEM_DIR, f"dem_aiguille_du_midi_{DEM_RESOLUTION}m.tif")
    if not os.path.exists(path):
        print(f"[error] DEM introuvable: {path}")
        print("  -> lancer T01_dem_download.py d'abord")
        sys.exit(1)

    with rasterio.open(path) as ds:
        dem = ds.read(1)
        profile = ds.profile.copy()

    print(f"[load] DEM charge: {dem.shape}, res={DEM_RESOLUTION}m")
    return dem, profile


# =====================================================
#  Masque nodata + dilatation
# =====================================================

def make_nodata_mask(dem, dilate=True):
    # masque base : la ou le DEM est nodata
    mask = (dem == NODATA_VALUE) | np.isnan(dem)

    if dilate:
        # augmnte de 1px pour exclure les pixels où 1 voisin est nodata (fix pour eviter les effets de bord)
        struct = np.ones((3, 3), dtype=bool)
        mask = binary_dilation(mask, structure=struct)

    return mask


# =====================================================
#  Pente + Aspect (Horn's method)
# =====================================================

def compute_slope_aspect(dem):
    res = DEM_RESOLUTION

    # remplace nodata par nan pour le padding
    work = dem.astype(np.float64)
    nodata_base = (dem == NODATA_VALUE) | np.isnan(dem)
    work[nodata_base] = np.nan

    # pad reflect pour eviter les artefacts de bord
    padded = np.pad(work, 1, mode='reflect')

    # Horn kernels - normalises par 8*resolution
    gx_kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float64) / (8.0 * res)

    gy_kernel = np.array([[-1, -2, -1],
                          [ 0,  0,  0],
                          [ 1,  2,  1]], dtype=np.float64) / (8.0 * res)

    # !! scipy.ndimage.convolve propage pas les NaN correctement mais on masque apres de toute facon via le dilate on met les nan a 0 pour la convolution
    padded_clean = np.where(np.isnan(padded), 0.0, padded)

    gx_full = convolve(padded_clean, gx_kernel, mode='constant', cval=0.0)
    gy_full = convolve(padded_clean, gy_kernel, mode='constant', cval=0.0)

    # trim padding
    gx = gx_full[1:-1, 1:-1]
    gy = gy_full[1:-1, 1:-1]

    # pente en degres
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # aspect : convention 0=Nord, 90=Est, 180=Sud, 270=Ouest
    aspect_rad = np.arctan2(-gy, gx)
    aspect_deg = np.degrees(aspect_rad)

    # conversion en [0, 360] avec 0=Nord, arctan2 donne 0=Est, on doit tourner de 90
    aspect_deg = 90.0 - aspect_deg
    aspect_deg = np.mod(aspect_deg, 360.0).astype(np.float32)

    # cas terrain plat (gx=gy=0) -> aspect indefini
    flat_mask = (gx == 0) & (gy == 0)
    aspect_deg[flat_mask] = NODATA_VALUE

    # applique le masque nodata dilate
    mask = make_nodata_mask(dem, dilate=True)
    slope_deg[mask] = NODATA_VALUE
    aspect_deg[mask] = NODATA_VALUE

    print(f"[slope] done - {np.sum(~mask)} pixels valides")
    print(f"[aspect] done - {np.sum(flat_mask & ~mask)} pixels plats (aspect=nodata)")

    return slope_deg, aspect_deg


# =====================================================
#  Rugosite (TRI - Riley et al., 1999)
# =====================================================

def compute_roughness(dem):
    # Terrain Roughness Index : ecart-type local sur fenetre 3x3
    # convolution NaN-aware pour eviter les artefacts pres du nodata
    # on convolve separement les valeurs et un masque de validite
    # comme ca les pixels nodata (mis a 0) ne polluent pas la moyenne

    nodata_base = (dem == NODATA_VALUE) | np.isnan(dem)
    valid_px = (~nodata_base).astype(np.float64)
    work = np.where(nodata_base, 0.0, dem.astype(np.float64))

    kernel = np.ones((3, 3), dtype=np.float64)

    # nb de voisins valides par pixel
    count = convolve(valid_px, kernel, mode='reflect')
    count = np.maximum(count, 1.0)  # avoid div/0

    sum_z = convolve(work, kernel, mode='reflect')
    sum_z2 = convolve(work ** 2, kernel, mode='reflect')

    mean_z = sum_z / count
    mean_z2 = sum_z2 / count

    variance = np.maximum(mean_z2 - mean_z ** 2, 0.0)
    roughness = np.sqrt(variance).astype(np.float32)

    mask = make_nodata_mask(dem, dilate=True)
    roughness[mask] = NODATA_VALUE

    print(f"[roughness] done - TRI 3x3")

    return roughness


# =====================================================
#  Export GeoTIFF
# =====================================================

def save_raster(data, name, profile):
    os.makedirs(DERIVED_DIR, exist_ok=True)
    path = os.path.join(DERIVED_DIR, f"{name}_{DEM_RESOLUTION}m.tif")

    out_profile = profile.copy()
    out_profile.update({
        "dtype": "float32",
        "count": 1,
        "nodata": NODATA_VALUE,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
    })

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data.astype(np.float32), 1)

    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"[save] {path} ({size_mb:.1f} MB)")
    return path


# =====================================================
#  Validation
# =====================================================

def validate():
    print("\n--- Validation ---")
    ok = True

    checks = {
        # 89+ possible sur les parois quasi-verticales a 1m (ex : aiguilles, seracs)
        "slope": {"expected_min": 0, "expected_max": 90, "warn_median": 75},
        "aspect": {"expected_min": 0, "expected_max": 360, "warn_median": None},
        "roughness": {"expected_min": 0, "expected_max": 25, "warn_median": 5},
    }

    for layer, params in checks.items():
        path = os.path.join(DERIVED_DIR, f"{layer}_{DEM_RESOLUTION}m.tif")
        if not os.path.exists(path):
            print(f"  [{layer}] FAIL - fichier absent")
            ok = False
            continue

        with rasterio.open(path) as ds:
            data = ds.read(1)
            valid = data[(data != NODATA_VALUE) & ~np.isnan(data)]

            if len(valid) == 0:
                print(f"  [{layer}] FAIL - que du nodata")
                ok = False
                continue

            nodata_pct = (1 - len(valid) / data.size) * 100
            p95 = np.percentile(valid, 95)
            p99 = np.percentile(valid, 99)
            p999 = np.percentile(valid, 99.9)

            print(f"  [{layer}] min={valid.min():.2f}, max={valid.max():.2f}, "
                  f"median={np.median(valid):.2f}, nodata={nodata_pct:.1f}%")
            print(f"    p95={p95:.2f}, p99={p99:.2f}, p99.9={p999:.2f}")

            if valid.min() < params["expected_min"]:
                print(f"    [warn] min < {params['expected_min']}")
            if valid.max() > params["expected_max"]:
                n_extreme = np.sum(valid > params["expected_max"])
                print(f"    [info] {n_extreme} pixels > {params['expected_max']} "
                      f"({n_extreme/len(valid)*100:.4f}%) - artefacts DEM probables")
            if params["warn_median"] and np.median(valid) > params["warn_median"]:
                print(f"    [warn] mediane elevee, a verifier")

    # verif % nodata coherent entre les couches
    dem_path = os.path.join(DEM_DIR, f"dem_aiguille_du_midi_{DEM_RESOLUTION}m.tif")
    if os.path.exists(dem_path):
        with rasterio.open(dem_path) as ds:
            dem_data = ds.read(1)
            dem_nodata_pct = np.sum(dem_data == NODATA_VALUE) / dem_data.size * 100
        print(f"\n  [ref] nodata DEM original: {dem_nodata_pct:.1f}%")

    return ok


# =====================================================
#  plot
# =====================================================

def plot_results():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    layers = {
        'slope': {'cmap': 'YlOrRd', 'vmin': 0, 'vmax': 80, 'label': 'Pente (deg)'},
        'aspect': {'cmap': 'hsv', 'vmin': 0, 'vmax': 360, 'label': 'Aspect (deg)'},
        'roughness': {'cmap': 'viridis', 'vmin': 0, 'vmax': 10, 'label': 'Rugosite TRI (m)'},
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (name, params) in zip(axes, layers.items()):
        path = os.path.join(DERIVED_DIR, f"{name}_{DEM_RESOLUTION}m.tif")
        if not os.path.exists(path):
            ax.set_title(f"{name} - ABSENT")
            continue

        with rasterio.open(path) as ds:
            data = ds.read(1)

        display = np.where((data == NODATA_VALUE) | np.isnan(data), np.nan, data)
        if name == 'roughness':
            display = np.clip(display, params['vmin'], params['vmax'])

        im = ax.imshow(display, cmap=params['cmap'],
                       vmin=params['vmin'], vmax=params['vmax'])
        fig.colorbar(im, ax=ax, shrink=0.7, label=params['label'])
        ax.set_title(name.capitalize(), fontsize=11)
        ax.set_xlabel('pixels')
        ax.set_ylabel('pixels')

    fig.suptitle(f'Terrain Analysis - {DEM_RESOLUTION}m', fontsize=13, y=1.02)
    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "terrain_analysis.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    uhd_dir = os.path.join(FIGURES_DIR, "uhd")
    os.makedirs(uhd_dir, exist_ok=True)
    fig.savefig(os.path.join(uhd_dir, "terrain_analysis.png"), dpi=UHD_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[plot] {out_path}")


# =====================================================

def main():
    print("=" * 50)
    print("T02 - Terrain Analysis")
    print(f"  resolution: {DEM_RESOLUTION}m")
    print("=" * 50)

    dem, profile = load_dem()

    print("\n--- Slope + Aspect ---")
    slope, aspect = compute_slope_aspect(dem)

    print("\n--- Roughness ---")
    roughness = compute_roughness(dem)

    print("\n--- Export ---")
    save_raster(slope, "slope", profile)
    save_raster(aspect, "aspect", profile)
    save_raster(roughness, "roughness", profile)

    validate()

    print("\n--- Visualisation ---")
    plot_results()

    print("\nDone!")


if __name__ == "__main__":
    main()
