# T04 - surface de cout multi-criteres
# combine pente, altitude, aspect, glacier, rugosite en un seul raster de cout

import os
import sys
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LightSource

from config import (
    DEM_DIR, DERIVED_DIR, FIGURES_DIR, DEM_RESOLUTION, NODATA_VALUE,
    DEFAULT_SEASON_MONTH, DEFAULT_ACCLIMATIZED, UHD_DPI,
)


# -----------------------------------
#  Chargement des couches
# -----------------------------------

def load_layer(name, dtype=np.float32):
    path = os.path.join(DERIVED_DIR, f"{name}_{DEM_RESOLUTION}m.tif")
    if not os.path.exists(path):
        print(f"[error] couche manquante: {path}")
        print("  -> lancer T02/T03 d'abord")
        sys.exit(1)
    with rasterio.open(path) as ds:
        data = ds.read(1).astype(dtype)
        profile = ds.profile.copy()
    return data, profile


def load_dem():
    path = os.path.join(DEM_DIR, f"dem_aiguille_du_midi_{DEM_RESOLUTION}m.tif")
    if not os.path.exists(path):
        print(f"[error] DEM introuvable: {path}")
        sys.exit(1)
    with rasterio.open(path) as ds:
        dem = ds.read(1).astype(np.float32)
        profile = ds.profile.copy()
    return dem, profile


def load_glacier_mask():
    path = os.path.join(DERIVED_DIR, f"glacier_mask_{DEM_RESOLUTION}m.tif")
    if not os.path.exists(path):
        print(f"[warn] pas de masque glacier, on continue sans")
        return None
    with rasterio.open(path) as ds:
        mask = ds.read(1)
    return mask.astype(bool)


# -----------------------------------
#  F_pente

def compute_slope_cost(slope_deg):
    """
    Tobler hiking function, adapte hors-sentier (x0.6).
    Cout isotrope
    """

    # conversion degres gradient (dz/dx)
    slope_rad = np.radians(np.clip(slope_deg, 0, 89.9))
    gradient = np.tan(slope_rad)

    # vitesse Tobler hors-piste
    v = 6.0 * np.exp(-3.5 * np.abs(gradient + 0.05)) * 0.6
    v = np.maximum(v, 0.01)  # evite div/0 sur les pentes extremes

    # normalise: cout=1.0 sur terrain plat (gradient=0)
    v_flat = 6.0 * np.exp(-3.5 * 0.05) * 0.6
    cost = v_flat / v

    # au dela de 45deg add penalite supplementaire
    steep = slope_deg > 45
    cost = np.where(steep, cost * 5.0, cost)

    # au-dela de 60deg classé comme quasi infranchissable a pied
    very_steep = slope_deg > 60
    cost = np.where(very_steep, cost * 50.0, cost)

    return cost.astype(np.float32)


# -----------------------------------
#  F_altitude

def compute_altitude_cost(elevation, acclimatized=True):
    """Reduction de perf liee a l'altitude"""

    rate = 0.03 if acclimatized else 0.063  # perte par 1000m au dessus de 1500
    reduction = np.maximum(0, (elevation - 1500.0) * rate / 1000.0)
    capacity = np.maximum(1.0 - reduction, 0.3)
    cost = 1.0 / capacity

    return cost.astype(np.float32)


# -----------------------------------
#  F_aspect - orientation + saison (chutes de pierres / neige)

def compute_aspect_cost(aspect_deg, slope_deg, elevation, month=7):
    """Penalite aspect/saison. En ete, les faces S/SW au dessus de 2500m
    sur pente raide = risque accru de chutes de pierres."""

    cost = np.ones_like(slope_deg, dtype=np.float32)

    # masque nodata aspect (pixels "plats")
    valid = (aspect_deg != NODATA_VALUE) & ~np.isnan(aspect_deg)

    if month in [6, 7, 8, 9]:  # ete
        # penalite faces sud sur pente > 30 au dessus de 2500m
        south_exp = np.cos(np.radians(aspect_deg - 180.0))
        penalty = 1.0 + 0.5 * np.maximum(south_exp, 0)
        mask = valid & (slope_deg > 30) & (elevation > 2500)
        cost = np.where(mask, penalty, cost)
    else:
        # hiver/printemps : faces nord
        north_exp = np.cos(np.radians(aspect_deg))
        penalty = 1.0 + 0.3 * np.maximum(north_exp, 0)
        mask = valid & (slope_deg > 25)
        cost = np.where(mask, penalty, cost)

    return cost


# -----------------------------------
#  F_glacier

def compute_glacier_cost(glacier_mask, slope_deg):
    """Penalite simplifiee pour les zones glaciaires.
    Pas de detection de crevasses (prevu V1 ou V2), juste surcout uniforme par tranche
    de pente."""

    if glacier_mask is None:
        return np.ones_like(slope_deg, dtype=np.float32)

    cost = np.ones_like(slope_deg, dtype=np.float32)

    # pente faible : glacier plat
    cost = np.where(glacier_mask & (slope_deg < 10), 1.3, cost)
    # pente moderee : progression + technique
    cost = np.where(glacier_mask & (slope_deg >= 10) & (slope_deg < 20), 2.0, cost)
    # pente forte : terrain glaciaire raide
    cost = np.where(glacier_mask & (slope_deg >= 20) & (slope_deg < 30), 4.0, cost)
    # > 30 : zone tres raide sur glace
    cost = np.where(glacier_mask & (slope_deg >= 30), 10.0, cost)

    return cost


# -----------------------------------
#  F_rugosite - terrain technique

def compute_roughness_cost(roughness):
    """Penalite liee a la rugosite/technicité du terrain (TRI).
    Terrain lisse ~1.0, eboulis/moraine ~2-3, blocs ~5."""

    # clamp a 5m pour eviter que les artefacts DEM explosent le cout
    r = np.minimum(roughness, 5.0)
    cost = 1.0 + 0.8 * r

    return cost.astype(np.float32)


# -----------------------------------
#  Assemblage surface de cout
# -----------------------------------

def build_cost_surface(dem, slope, aspect, roughness, glacier_mask,
                       month=7, acclimatized=True):
    print(f"[params] month={month}, acclimatized={acclimatized}")

    # masque nodata global
    nodata_mask = (slope == NODATA_VALUE) | np.isnan(slope)

    # on travaille en float32 sur les pixels valides
    slope_clean = np.where(nodata_mask, 0, slope)
    aspect_clean = np.where(nodata_mask, 0, aspect)
    rough_clean = np.where(nodata_mask, 0, roughness)
    dem_clean = np.where(nodata_mask, 0, dem)

    # calcul de chaque facteur
    print("  -> f_slope (Tobler hors-sentier)")
    f_slope = compute_slope_cost(slope_clean)

    print("  -> f_altitude (hypoxie)")
    f_alt = compute_altitude_cost(dem_clean, acclimatized)

    print("  -> f_aspect (orientation/saison)")
    f_aspect = compute_aspect_cost(aspect_clean, slope_clean, dem_clean, month)

    print("  -> f_glacier (penalite glaciaire)")
    f_glacier = compute_glacier_cost(glacier_mask, slope_clean)

    print("  -> f_roughness (terrain)")
    f_rough = compute_roughness_cost(rough_clean)

    # produit de tous les facteurs
    cost = f_slope * f_alt * f_aspect * f_glacier * f_rough

    # remet nodata la ou c'etait nodata
    cost[nodata_mask] = NODATA_VALUE

    print(f"\n[cost] surface calculee: {cost.shape}")

    # on renvoie aussi les facteurs individuels pour la visu
    factors = {
        'slope': f_slope, 'altitude': f_alt, 'aspect': f_aspect,
        'glacier': f_glacier, 'roughness': f_rough,
    }
    for name, f in factors.items():
        f[nodata_mask] = NODATA_VALUE

    return cost, factors, nodata_mask


# -----------------------------------
#  Sauvegarde GeoTIFF
# -----------------------------------

def save_cost_surface(cost, profile):
    os.makedirs(DERIVED_DIR, exist_ok=True)
    path = os.path.join(DERIVED_DIR, f"cost_surface_{DEM_RESOLUTION}m.tif")

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
        dst.write(cost.astype(np.float32), 1)

    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"[save] {path} ({size_mb:.1f} MB)")
    return path


# -----------------------------------
#  Validation
# -----------------------------------

def validate(cost, nodata_mask, factors):
    print("\n--- Validation ---")

    valid = cost[~nodata_mask]
    if len(valid) == 0:
        print("  [FAIL] que du nodata")
        return False

    nodata_pct = nodata_mask.sum() / cost.size * 100

    print(f"  total pixels: {cost.size:,}")
    print(f"  nodata: {nodata_pct:.2f}%")
    print(f"  cout min:    {valid.min():.3f}")
    print(f"  cout max:    {valid.max():.1f}")
    print(f"  cout median: {np.median(valid):.3f}")
    print(f"  cout mean:   {valid.mean():.3f}")

    # percentiles
    for p in [50, 75, 90, 95, 99, 99.9]:
        print(f"    p{p}: {np.percentile(valid, p):.3f}")

    # repartition par tranches de cout
    bins = [(0, 2, "facile"), (2, 5, "modere"), (5, 20, "difficile"),
            (20, 100, "tres difficile"), (100, np.inf, "infranchissable")]
    print("\n  repartition:")
    for lo, hi, label in bins:
        n = np.sum((valid >= lo) & (valid < hi))
        pct = n / len(valid) * 100
        print(f"    [{lo:>6.0f} - {hi:>6.0f}] {label:20s} : {pct:5.1f}% ({n:>10,} px)")

    # stats par facteur
    print("\n  facteurs individuels (median / max sur pixels valides):")
    for name, f in factors.items():
        fv = f[~nodata_mask]
        print(f"    {name:12s}: median={np.median(fv):.3f}, max={fv.max():.1f}")

    return True


# -----------------------------------
#  Plot
# -----------------------------------

def _make_hillshade(dem):
    """Hillshade pour fond de carte, reutilise dans tous les plots."""
    dem_display = np.where(dem == NODATA_VALUE, np.nan, dem)
    dem_filled = np.where(np.isnan(dem_display), 0, dem_display)
    ls = LightSource(azdeg=315, altdeg=45)
    return ls.hillshade(dem_filled, vert_exag=2,
                        dx=DEM_RESOLUTION, dy=DEM_RESOLUTION)


def plot_results(cost, factors, nodata_mask, dem):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    hillshade = _make_hillshade(dem)

    # --- Figure 1 : surface de cout globale ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    ax.imshow(hillshade, cmap='gray', alpha=0.4)
    cost_display = np.where(nodata_mask, np.nan, cost)
    im = ax.imshow(cost_display, cmap='RdYlGn_r', norm=LogNorm(vmin=1, vmax=500),
                   alpha=0.75)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Multiplicateur de cout (1 = terrain plat)')
    cbar.ax.text(1.3, 0.0, 'facile', transform=cbar.ax.transAxes,
                 fontsize=8, va='bottom', color='green')
    cbar.ax.text(1.3, 1.0, 'quasi-impraticable', transform=cbar.ax.transAxes,
                 fontsize=8, va='top', color='red')
    ax.set_title('Surface de cout totale', fontsize=11)

    # histogramme
    ax = axes[1]
    valid = cost[~nodata_mask]
    ax.hist(valid, bins=np.logspace(0, 4, 80), color='steelblue',
            edgecolor='none', alpha=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Cout (multiplicateur)')
    ax.set_ylabel('Nombre de pixels')
    ax.set_title('Distribution des couts', fontsize=11)
    ax.axvline(np.median(valid), color='red', ls='--', lw=1.2,
               label=f'median={np.median(valid):.2f}')
    ax.legend(fontsize=9)

    fig.suptitle(f'Cost Surface - {DEM_RESOLUTION}m (month={DEFAULT_SEASON_MONTH})',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    path1 = os.path.join(FIGURES_DIR, "cost_surface.png")
    fig.savefig(path1, dpi=150, bbox_inches='tight', facecolor='white')
    uhd_dir = os.path.join(FIGURES_DIR, "uhd")
    os.makedirs(uhd_dir, exist_ok=True)
    fig.savefig(os.path.join(uhd_dir, "cost_surface.png"), dpi=UHD_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[plot] {path1}")

    # --- Figure 2 : les 5 facteurs individuels ---
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 11))
    axes2 = axes2.flatten()

    # description par facteur pour la colorbar
    factor_params = {
        'slope': {
            'cmap': 'YlOrRd', 'vmin': 1, 'vmax': 50, 'log': True,
            'title': 'F_pente',
            'cbar_label': 'Multiplicateur (1=plat)',
            'lo_text': 'plat / pente douce', 'hi_text': 'paroi raide',
        },
        'altitude': {
            'cmap': 'Purples', 'vmin': 1, 'vmax': 1.3, 'log': False,
            'title': 'F_altitude',
            'cbar_label': 'Multiplicateur',
            'lo_text': '< 1500m, pas de penalite', 'hi_text': '~4000m, -10% perf',
        },
        'aspect': {
            'cmap': 'OrRd', 'vmin': 1, 'vmax': 1.5, 'log': False,
            'title': 'F_aspect (orientation, juillet)',
            'cbar_label': 'Multiplicateur',
            'lo_text': 'face N ou pente faible', 'hi_text': 'face S raide >2500m',
        },
        'glacier': {
            'cmap': 'Blues', 'vmin': 1, 'vmax': 10, 'log': True,
            'title': 'F_glacier (penalite uniforme)',
            'cbar_label': 'Multiplicateur',
            'lo_text': 'hors glacier / glacier plat', 'hi_text': 'glacier tres raide',
        },
        'roughness': {
            'cmap': 'YlGn', 'vmin': 1, 'vmax': 5, 'log': False,
            'title': 'F_rugosite (terrain)',
            'cbar_label': 'Multiplicateur',
            'lo_text': 'lisse (neige, pelouse)', 'hi_text': 'blocs / moraine',
        },
    }

    for idx, (name, params) in enumerate(factor_params.items()):
        ax = axes2[idx]
        f = factors[name]
        f_display = np.where(nodata_mask, np.nan, f)

        if params['log']:
            norm = LogNorm(vmin=params['vmin'], vmax=params['vmax'])
        else:
            norm = None
            f_display = np.clip(f_display, params['vmin'], params['vmax'])

        kwargs = {'cmap': params['cmap'], 'alpha': 0.85}
        if norm:
            kwargs['norm'] = norm
        else:
            kwargs['vmin'] = params['vmin']
            kwargs['vmax'] = params['vmax']

        ax.imshow(hillshade, cmap='gray', alpha=0.3)
        im = ax.imshow(f_display, **kwargs)
        cbar = fig2.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label(params['cbar_label'], fontsize=8)
        cbar.ax.text(1.3, 0.0, params['lo_text'], transform=cbar.ax.transAxes,
                     fontsize=7, va='bottom', style='italic')
        cbar.ax.text(1.3, 1.0, params['hi_text'], transform=cbar.ax.transAxes,
                     fontsize=7, va='top', style='italic')
        ax.set_title(params['title'], fontsize=10)

    # plot cout total
    ax = axes2[5]
    ax.imshow(hillshade, cmap='gray', alpha=0.4)
    cost_d = np.where(nodata_mask, np.nan, cost)
    im = ax.imshow(cost_d, cmap='RdYlGn_r', norm=LogNorm(vmin=1, vmax=500), alpha=0.75)
    cbar = fig2.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Cout total (produit)', fontsize=8)
    cbar.ax.text(1.3, 0.0, 'facile', transform=cbar.ax.transAxes,
                 fontsize=7, va='bottom', style='italic', color='green')
    cbar.ax.text(1.3, 1.0, 'tres couteux', transform=cbar.ax.transAxes,
                 fontsize=7, va='top', style='italic', color='red')
    ax.set_title('Cout total (produit des 5 facteurs)', fontsize=10)

    fig2.suptitle(f'Facteurs de cout individuels - {DEM_RESOLUTION}m',
                  fontsize=13, y=1.01)
    fig2.tight_layout()
    path2 = os.path.join(FIGURES_DIR, "cost_factors.png")
    fig2.savefig(path2, dpi=150, bbox_inches='tight', facecolor='white')
    fig2.savefig(os.path.join(uhd_dir, "cost_factors.png"), dpi=UHD_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"[plot] {path2}")

    # --- Figure 3 : decoupage par niveaux de cout ---
    cost_levels = [
        (0, 2,     'Facile (1-2x)',         '#2ecc71', 'terrain plat, pente douce'),
        (2, 5,     'Modere (2-5x)',         '#f1c40f', 'pente moderee, hors-sentier'),
        (5, 20,    'Difficile (5-20x)',     '#e67e22', 'pente forte, glacier, eboulis'),
        (20, 100,  'Tres difficile (20-100x)', '#e74c3c', 'terrain technique'),
        (100, np.inf, 'Quasi-impraticable (>100x)', '#8e44ad', 'parois, seracs, falaises'),
    ]

    fig3, axes3 = plt.subplots(1, 5, figsize=(25, 5))

    valid_cost = cost[~nodata_mask]
    total_valid = len(valid_cost)

    for ax, (lo, hi, label, color, desc) in zip(axes3, cost_levels):
        ax.imshow(hillshade, cmap='gray', alpha=0.5)

        level_mask = (~nodata_mask) & (cost >= lo) & (cost < hi)
        overlay = np.full(cost.shape + (4,), 0.0)
        r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
        overlay[level_mask] = [r, g, b, 0.7]

        ax.imshow(overlay)

        n = level_mask.sum()
        pct = n / total_valid * 100
        ax.set_title(f'{label}\n{pct:.1f}% ({n:,} px)', fontsize=9)
        ax.text(0.5, -0.02, desc, transform=ax.transAxes, fontsize=7,
                ha='center', va='top', style='italic', color='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    fig3.suptitle(f'Repartition spatiale par niveau de cout - {DEM_RESOLUTION}m',
                  fontsize=13, y=1.05)
    fig3.tight_layout()
    path3 = os.path.join(FIGURES_DIR, "cost_levels.png")
    fig3.savefig(path3, dpi=150, bbox_inches='tight', facecolor='white')
    fig3.savefig(os.path.join(uhd_dir, "cost_levels.png"), dpi=UHD_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig3)
    print(f"[plot] {path3}")


# -----------------------------------

def main():
    print("=" * 50)
    print("T04 - Cost Surface")
    print(f"  resolution: {DEM_RESOLUTION}m")
    print(f"  season: month={DEFAULT_SEASON_MONTH}")
    print(f"  acclimatized: {DEFAULT_ACCLIMATIZED}")
    print("=" * 50)

    print("\n--- Chargement des couches ---")
    dem, profile = load_dem()
    slope, _ = load_layer("slope")
    aspect, _ = load_layer("aspect")
    roughness, _ = load_layer("roughness")
    glacier_mask = load_glacier_mask()

    print(f"  DEM: {dem.shape}")
    print(f"  glacier: {'oui' if glacier_mask is not None else 'non'}")

    print("\n--- Construction surface de cout ---")
    cost, factors, nodata_mask = build_cost_surface(
        dem, slope, aspect, roughness, glacier_mask,
        month=DEFAULT_SEASON_MONTH,
        acclimatized=DEFAULT_ACCLIMATIZED,
    )

    print("\n--- Export ---")
    save_cost_surface(cost, profile)

    validate(cost, nodata_mask, factors)

    print("\n--- Visualisation ---")
    plot_results(cost, factors, nodata_mask, dem)



if __name__ == "__main__":
    main()
