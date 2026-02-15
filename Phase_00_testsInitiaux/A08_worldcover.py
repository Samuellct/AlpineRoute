# A08 - test ESA WorldCover

import os
import sys
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.windows import from_bounds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from config import (
    DEM_DIR, DERIVED_DIR, FIGURES_DIR,
    DEM_RESOLUTION, NODATA_VALUE,
    BBOX_L93, BBOX_WGS84, CRS_L93, CRS_WGS84,
)


# WorldCover classes
WORLDCOVER_CLASSES = {
    10: ("Tree cover", "#006400"),
    20: ("Shrubland", "#ffbb22"),
    30: ("Grassland", "#ffff4c"),
    40: ("Cropland", "#f096ff"),
    50: ("Built-up", "#fa0000"),
    60: ("Bare / sparse vegetation", "#b4b4b4"),
    70: ("Snow and ice", "#f0f0f0"),
    80: ("Permanent water bodies", "#0064c8"),
    90: ("Herbaceous wetland", "#0096a0"),
    95: ("Mangroves", "#00cf75"),
    100: ("Moss and lichen", "#fae6a0"),
}

# mapping classes
COST_MULTIPLIERS = {
    10: 2.5,    # foret
    20: 1.8,    # buissons
    30: 1.2,    # herbe, alpage
    40: 1.0,    # cultures
    50: 1.0,    # bati
    60: 1.5,    # moraine, eboulis
    70: 1.3,    # neige/glace (overlap avec masque glacier)
    80: 50.0,   # eau
    90: 3.0,    # zone humide
    95: 3.0,    # mangrove
    100: 1.3,   # mousse/lichen
    0: 1.0,     # nodata
}

WORLDCOVER_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N45E006_Map.tif"


def load_worldcover_windowed():
    """Charge juste la fenetre Chamonix via /vsicurl/ (pas de dl complet)."""
    print(f"[worldcover] lecture fenetree via /vsicurl/")
    print(f"  url: {WORLDCOVER_URL}")

    vsicurl = f"/vsicurl/{WORLDCOVER_URL}"

    try:
        with rasterio.open(vsicurl) as ds:
            print(f"  tuile complete: shape={ds.shape}, crs={ds.crs}, res={ds.res}")
            print(f"  bounds: {ds.bounds}")

            # fenetre correspondant a notre bbox de test
            bbox = BBOX_WGS84
            window = from_bounds(
                bbox["lon_min"], bbox["lat_min"],
                bbox["lon_max"], bbox["lat_max"],
                ds.transform,
            )
            print(f"  window: {window}")

            data = ds.read(1, window=window)
            win_transform = ds.window_transform(window)

            print(f"  crop: shape={data.shape}")
            return data, win_transform, ds.crs

    except Exception as e:
        print(f"  [ERREUR] {e}")
        print("  Verifier que GDAL a le support curl (GDAL_HTTP_UNSAFESSL ?)")
        return None, None, None


def reproject_to_l93(data, src_transform, src_crs):
    """Reprojette en Lambert-93 a 10m (nearest neighbor pour categorique)."""
    print("\n[reproject] WGS84 -> L93 @ 10m")

    bbox = BBOX_L93
    dst_res = 10  # WorldCover est a 10m, on garde cette resolution
    width = int((bbox["xmax"] - bbox["xmin"]) / dst_res)
    height = int((bbox["ymax"] - bbox["ymin"]) / dst_res)

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, CRS_L93,
        data.shape[1], data.shape[0],
        left=BBOX_WGS84["lon_min"], bottom=BBOX_WGS84["lat_min"],
        right=BBOX_WGS84["lon_max"], top=BBOX_WGS84["lat_max"],
        dst_width=width, dst_height=height,
    )

    dst = np.zeros((height, width), dtype=np.uint8)

    reproject(
        source=data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=CRS_L93,
        resampling=Resampling.nearest,  # nearest pour donnees categoriques
    )

    print(f"  result: {dst.shape}, {width}x{height} @ {dst_res}m")
    return dst


def analyze_classes(data):
    """Stats sur les classes presentes."""
    print("\n--- Repartition des classes ---")
    total = data.size
    for code, (name, _) in sorted(WORLDCOVER_CLASSES.items()):
        count = np.sum(data == code)
        if count > 0:
            pct = count / total * 100
            print(f"  {code:>3} {name:30s} : {count:>8} px ({pct:5.1f}%)")

    # pixels sans classe connue
    known = sum(np.sum(data == c) for c in WORLDCOVER_CLASSES)
    unknown = total - known
    if unknown > 0:
        print(f"  ???  Unknown/nodata             : {unknown:>8} px ({unknown/total*100:5.1f}%)")


def compare_with_glacier_mask(lc_data, lc_res=10):
    """Compare classe 70 (snow/ice) avec le masque glacier RGI de T03."""
    print("\n--- Comparaison snow/ice (70) vs masque glacier RGI ---")

    glacier_path = os.path.join(DERIVED_DIR, f"glacier_mask_{DEM_RESOLUTION}m.tif")
    if not os.path.exists(glacier_path):
        print("  [skip] masque glacier introuvable")
        return

    with rasterio.open(glacier_path) as ds:
        glacier = ds.read(1).astype(bool)

    # le masque glacier est a DEM_RESOLUTION (1m), le land cover a 10m
    # on down-sample le glacier a 10m pour comparer
    ratio = int(lc_res / DEM_RESOLUTION)
    if ratio > 1:
        # block mean : >50% glacier = glacier
        h, w = lc_data.shape
        glacier_ds = np.zeros((h, w), dtype=bool)
        for r in range(h):
            for c in range(w):
                r0, r1 = r * ratio, min((r + 1) * ratio, glacier.shape[0])
                c0, c1 = c * ratio, min((c + 1) * ratio, glacier.shape[1])
                if r1 > r0 and c1 > c0:
                    block = glacier[r0:r1, c0:c1]
                    glacier_ds[r, c] = block.mean() > 0.5
    else:
        glacier_ds = glacier[:lc_data.shape[0], :lc_data.shape[1]]

    snow_ice = (lc_data == 70)

    # overlap
    both = snow_ice & glacier_ds
    only_lc = snow_ice & ~glacier_ds
    only_rgi = glacier_ds & ~snow_ice

    total = max(np.sum(snow_ice | glacier_ds), 1)
    print(f"  Snow/ice (WC):     {snow_ice.sum():>8} px")
    print(f"  Glacier (RGI):     {glacier_ds.sum():>8} px")
    print(f"  Les deux:          {both.sum():>8} px ({both.sum()/total*100:.1f}% de l'union)")
    print(f"  WC only:           {only_lc.sum():>8} px (neige saisonniere ?)")
    print(f"  RGI only:          {only_rgi.sum():>8} px (glacier sous debris ?)")

    # Jaccard index
    jaccard = both.sum() / total if total > 0 else 0
    print(f"  Jaccard index:     {jaccard:.3f}")


def plot_landcover(data):
    """Visualise le land cover avec les couleurs ESA."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # build colormap
    codes = sorted(WORLDCOVER_CLASSES.keys())
    colors = [WORLDCOVER_CLASSES[c][1] for c in codes]
    cmap = ListedColormap(colors)
    bounds = codes + [max(codes) + 1]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(data, cmap=cmap, norm=norm, interpolation='nearest')

    # colorbar avec labels
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, ticks=codes)
    cbar.ax.set_yticklabels([
        f"{c}: {WORLDCOVER_CLASSES[c][0]}" for c in codes
    ], fontsize=8)
    ax.set_title('ESA WorldCover 2021 - Chamonix (10m)')

    out = os.path.join(FIGURES_DIR, "worldcover_chamonix.png")
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[plot] {out}")


def build_cost_layer(data):
    """Construit le multiplicateur de cout a partir du land cover."""
    print("\n--- Couche de cout land cover ---")
    cost = np.ones_like(data, dtype=np.float32)
    for code, mult in COST_MULTIPLIERS.items():
        cost[data == code] = mult

    valid = cost[cost > 0]
    print(f"  min={valid.min():.1f}, max={valid.max():.1f}, mean={valid.mean():.2f}")
    print(f"  pixels eau (cout 50x): {np.sum(data == 80)}")
    print(f"  pixels foret (cout 2.5x): {np.sum(data == 10)}")
    return cost


def main():
    print("=" * 60)
    print("A08 - ESA WorldCover 10m")
    print("=" * 60)

    # charge la tuile (fenetree)
    data, transform, crs = load_worldcover_windowed()
    if data is None:
        print("\n[FAIL] impossible de charger WorldCover")
        sys.exit(1)

    # reproject
    data_l93 = reproject_to_l93(data, transform, crs)

    # analyse
    analyze_classes(data_l93)

    # comparaison glacier
    compare_with_glacier_mask(data_l93)

    # cout
    cost = build_cost_layer(data_l93)

    # visu
    print("\n--- Visualisation ---")
    plot_landcover(data_l93)

    # conclusion
    print("\n--- Conclusion ---")
    n_forest = np.sum(data_l93 == 10)
    n_bare = np.sum(data_l93 == 60)
    n_snow = np.sum(data_l93 == 70)
    total = data_l93.size

    useful = (n_forest + n_bare + n_snow) / total * 100
    print(f"  Classes utiles pour le cout (foret+moraine+neige): {useful:.1f}% des pixels")
    if useful > 10:
        print("  -> WorldCover maybe utile en V1")
    else:
        print("  -> useless")

    print("\nDone")


if __name__ == "__main__":
    main()
