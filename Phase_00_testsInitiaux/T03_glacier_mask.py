# T03 - masque glaciaire depuis la data RGI 7.0
# rasterise les polygones glacier sur la grille DEM pour la surface de cout

import os
import sys
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

from config import (
    RGI_DIR, DERIVED_DIR, DEM_DIR, FIGURES_DIR,
    BBOX_L93, CRS_L93, DEM_RESOLUTION, NODATA_VALUE,
)


# =====================================================
#  Verif donnees RGI
# =====================================================

def check_rgi_data():
    if not os.path.isdir(RGI_DIR):
        os.makedirs(RGI_DIR, exist_ok=True)

    shp_paths = [] #cherche les .shp dans rgi/ et ses sous-dossiers
    for root, dirs, files in os.walk(RGI_DIR):
        for f in files:
            if f.endswith('.shp'):
                shp_paths.append(os.path.join(root, f))

    if not shp_paths:
        print("[error] Aucun shapefile RGI trouve dans", RGI_DIR)
        print()
        print("  Telecharger RGI 7.0 region 11 (Central Europe) depuis NSIDC :")
        print("  https://nsidc.org/data/nsidc-0770/versions/7")
        print("  -> RGI2000-v7.0-G-11_central_europe.zip")
        print()
        print(f"  Extraire les fichiers dans : {RGI_DIR}")
        sys.exit(1)

    return shp_paths


def load_rgi(shp_paths):
    # charge le premier .shp trouve
    shp_path = shp_paths[0]
    print(f"[rgi] chargement {os.path.basename(shp_path)}...")

    gdf = gpd.read_file(shp_path)
    print(f"  {len(gdf)} glaciers total dans le fichier")

    # reproject L93 si necessaire
    if gdf.crs and gdf.crs.to_epsg() != 2154:
        print(f"  reprojection {gdf.crs} -> L93")
        gdf = gdf.to_crs(CRS_L93)

    # filtre spatial sur notre bbox
    bbox = BBOX_L93
    gdf = gdf.cx[bbox['xmin']:bbox['xmax'], bbox['ymin']:bbox['ymax']]
    print(f"  {len(gdf)} glaciers dans la bbox")

    return gdf


# =====================================================
#  Chargement profil DEM (pas les donnees)
# =====================================================

def load_dem_profile():
    path = os.path.join(DEM_DIR, f"dem_aiguille_du_midi_{DEM_RESOLUTION}m.tif")
    if not os.path.exists(path):
        print(f"[error] DEM introuvable: {path}")
        print("  -> lancer T01_dem_download.py d'abord")
        sys.exit(1)

    with rasterio.open(path) as ds:
        profile = ds.profile.copy()
        transform = ds.transform
        shape = ds.shape

    return profile, transform, shape


# =====================================================
#  Rasterisation
# =====================================================

def rasterize_glaciers(gdf, transform, shape):
    if len(gdf) == 0:
        print("[warn] aucun glacier a rasteriser")
        return np.zeros(shape, dtype=np.uint8)

    # geometries en format (geom, value)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]

    mask = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
    )

    n_glacier_px = np.sum(mask == 1)
    pct = n_glacier_px / mask.size * 100
    print(f"[rasterize] {n_glacier_px} pixels glacier ({pct:.1f}% de la zone)")

    return mask


# =====================================================
#  Sauvegarde
# =====================================================

def save_raster(mask, profile):
    os.makedirs(DERIVED_DIR, exist_ok=True)
    path = os.path.join(DERIVED_DIR, f"glacier_mask_{DEM_RESOLUTION}m.tif")

    out_profile = profile.copy()
    out_profile.update({
        "dtype": "uint8",
        "count": 1,
        "nodata": None,
        "compress": "deflate",
        "tiled": True,
    })
    # virer predictor si present (normalmet pas utile pour uint8 binaire)
    out_profile.pop("predictor", None)

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(mask, 1)

    size_kb = os.path.getsize(path) / 1024
    print(f"[save] {path} ({size_kb:.0f} KB)")
    return path


# =====================================================
#  Visualisation
# =====================================================

def plot_results(mask, transform, shape):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # charger le DEM pour le hillshade de fond
    dem_path = os.path.join(DEM_DIR, f"dem_aiguille_du_midi_{DEM_RESOLUTION}m.tif")
    with rasterio.open(dem_path) as ds:
        dem = ds.read(1)

    dem_display = np.where(dem == NODATA_VALUE, np.nan, dem)

    ls = LightSource(azdeg=315, altdeg=45)
    # hillshade sur le DEM (nan-safe)
    dem_filled = np.where(np.isnan(dem_display), 0, dem_display)
    hillshade = ls.hillshade(dem_filled, vert_exag=2, dx=DEM_RESOLUTION, dy=DEM_RESOLUTION)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.imshow(hillshade, cmap='gray', alpha=0.6)

    # superpose le masque glacier en bleu transparent
    glacier_overlay = np.ma.masked_where(mask == 0, mask)
    ax.imshow(glacier_overlay, cmap='Blues', alpha=0.5, vmin=0, vmax=1)

    # contours des glaciers
    ax.contour(mask, levels=[0.5], colors='dodgerblue', linewidths=0.8)

    ax.set_title(f'Masque glaciaire - {DEM_RESOLUTION}m', fontsize=12)
    ax.set_xlabel('pixels')
    ax.set_ylabel('pixels')

    out_path = os.path.join(FIGURES_DIR, "glacier_mask.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[plot] {out_path}")


# =====================================================
#  Validation
# =====================================================

def validate(gdf, mask):
    print("\n--- Validation ---")

    n_glaciers = len(gdf)
    n_glacier_px = np.sum(mask == 1)
    pct = n_glacier_px / mask.size * 100

    # surface totale (chaque pixel = DEM_RESOLUTION^2 m2)
    surface_km2 = n_glacier_px * (DEM_RESOLUTION ** 2) / 1e6

    print(f"  glaciers dans la bbox: {n_glaciers}")
    print(f"  pixels glacier: {n_glacier_px}")
    print(f"  couverture: {pct:.1f}%")
    print(f"  surface glaciaire: {surface_km2:.2f} km2")

    # noms si dispo
    name_col = None
    for col in ['glac_name', 'name', 'Name', 'GLAC_NAME']:
        if col in gdf.columns:
            name_col = col
            break

    if name_col:
        named = gdf[gdf[name_col].notna() & (gdf[name_col] != '')]
        if len(named) > 0:
            print(f"\n  glaciers nommes ({len(named)}):")
            for _, row in named.iterrows():
                area = row.get('area_km2', row.get('Area', '?'))
                print(f"    - {row[name_col]} ({area} km2)")

    # surface depuis les attributs RGI
    area_col = None
    for col in ['area_km2', 'Area', 'AREA']:
        if col in gdf.columns:
            area_col = col
            break

    if area_col:
        rgi_total = gdf[area_col].sum()
        print(f"\n  surface RGI (attributs): {rgi_total:.2f} km2")

    # sanity checks
    if pct < 1:
        print("  [warn] < 1% de couverture glaciaire, verifier le fichier RGI")
    if pct > 50:
        print("  [warn] > 50% de couverture glaciaire, un peu beaucoup non?")

    return True


# =====================================================

def main():
    print("=" * 50)
    print("T03 - Glacier Mask (RGI 7.0)")
    print(f"  resolution: {DEM_RESOLUTION}m")
    print("=" * 50)

    # check RGI
    shp_files = check_rgi_data()

    # load + filtre
    gdf = load_rgi(shp_files)

    # profil DEM
    profile, transform, shape = load_dem_profile()

    # rasterise
    print("\n--- Rasterisation ---")
    mask = rasterize_glaciers(gdf, transform, shape)

    # save
    print("\n--- Export ---")
    save_raster(mask, profile)

    # visu
    print("\n--- Visualisation ---")
    plot_results(mask, transform, shape)

    # validation
    validate(gdf, mask)

    print("\nDone!")


if __name__ == "__main__":
    main()
