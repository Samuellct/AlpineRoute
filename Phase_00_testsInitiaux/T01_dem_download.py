# T01 - dl et preparation du DEM pour la zone de test

import os
import sys
import time
import math
from urllib.parse import urlparse, parse_qs, urlencode
import httpx
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

from config import (
    DEM_DIR, FIGURES_DIR, BBOX_L93, BBOX_WGS84, CRS_L93,
    DEM_RESOLUTION, NODATA_VALUE,
)


# -- params IGN WFS/WMS-R
WFS_URL = "https://data.geopf.fr/wfs/ows"
WFS_TYPENAME = "IGNF_MNT-LIDAR-HD:dalle"

HTTP_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAY = 2


def get_output_path():
    return os.path.join(DEM_DIR, f"dem_aiguille_du_midi_{DEM_RESOLUTION}m.tif")


def check_cache():
    path = get_output_path()
    if os.path.exists(path):
        with rasterio.open(path) as ds:
            data = ds.read(1)
            valid = data[data != NODATA_VALUE]
            if len(valid) > 0:
                print(f"[cache] DEM deja present: {path}")
                print(f"  shape={ds.shape}, alti min={valid.min():.0f}m, max={valid.max():.0f}m")
                return True
    return False


# =====================================================
#  Lidar HD : WFS discovery + WMS-R download
# =====================================================

def wfs_discover_dalles():
    # recup la liste des dalles qui intersectent la bbox
    bbox = BBOX_L93
    bbox_str = f"{bbox['xmin']},{bbox['ymin']},{bbox['xmax']},{bbox['ymax']}"

    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": WFS_TYPENAME,
        "OUTPUTFORMAT": "application/json",
        "BBOX": f"{bbox_str},EPSG:2154",
        "COUNT": "500",
    }

    print(f"[wfs] requete dalles pour bbox L93: {bbox_str}")
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        resp = client.get(WFS_URL, params=params)
        resp.raise_for_status()

    data = resp.json()
    features = data.get("features", [])
    print(f"[wfs] {len(features)} dalles trouvees")
    return features


def build_dalle_url(dalle_feature):
    # le WFS renvoie directement l'URL WMS-R complete dans properties.url
    # juste besoin de la modifier pour ajuster WIDTH/HEIGHT selon la resolution voulue
    props = dalle_feature.get("properties", {})
    base_url = props.get("url", "")
    if not base_url:
        return None

    # parse l'url pour modifier width/height
    parsed = urlparse(base_url)
    params = parse_qs(parsed.query, keep_blank_values=True)

    bbox_str = props.get("bbox", "") # bbox L93 de la dalle (string "xmin,ymin,xmax,ymax")
    if bbox_str:
        parts = [float(x) for x in bbox_str.split(",")]
        width_m = parts[2] - parts[0]
        height_m = parts[3] - parts[1]
    else:
        width_m = 1000 # dimension de la tuile, 1km par defaut
        height_m = 1000

    width_px = int(round(width_m / DEM_RESOLUTION))
    height_px = int(round(height_m / DEM_RESOLUTION))

    # flatten les params (parse_qs renvoie des listes)
    flat_params = {k: v[0] for k, v in params.items()}
    flat_params["WIDTH"] = str(width_px)
    flat_params["HEIGHT"] = str(height_px)

    new_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{urlencode(flat_params)}"
    return new_url, width_px, height_px


def download_dalle_wmsr(dalle_feature, out_path):
    props = dalle_feature.get("properties", {})
    dalle_name = props.get("name", "unknown")

    result = build_dalle_url(dalle_feature)
    if result is None:
        print(f"  [warn] pas d'url pour {dalle_name}, skip")
        return False

    url, width_px, height_px = result
    print(f"  [wmsr] download {dalle_name} ({width_px}x{height_px}px)...")

    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                resp = client.get(url)
                resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "tiff" not in content_type and "image" not in content_type:
                # maybe erreur XML dans le parsing
                print(f"  [warn] reponse inattendue ({content_type}): {resp.text[:200]}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return False

            with open(out_path, "wb") as f:
                f.write(resp.content)

            # verif rapide
            with rasterio.open(out_path) as ds:
                _ = ds.shape
            return True

        except Exception as e:
            print(f"  [retry {attempt+1}/{MAX_RETRIES}] {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    return False


def try_ign_lidar():
    # tente le dl via Lidar HD
    os.makedirs(DEM_DIR, exist_ok=True)
    raw_dir = os.path.join(DEM_DIR, "raw_ign")
    os.makedirs(raw_dir, exist_ok=True)

    try:
        features = wfs_discover_dalles()
    except Exception as e:
        print(f"[ign] WFS discovery failed: {e}")
        return []

    if not features:
        print("[ign] aucune dalle trouvee pour cette bbox")
        return []

    downloaded = []
    for i, feat in enumerate(features):
        props = feat.get("properties", {})
        dalle_name = props.get("name", f"dalle_{i}")
        out_path = os.path.join(raw_dir, f"{dalle_name}.tif")

        # skip si deja telecharge
        if os.path.exists(out_path):
            try:
                with rasterio.open(out_path) as ds:
                    _ = ds.shape
                print(f"  [cache] {dalle_name} deja present")
                downloaded.append(out_path)
                continue
            except:
                os.remove(out_path)

        if download_dalle_wmsr(feat, out_path):
            downloaded.append(out_path)
            # un petit sleep pour pas spam l'API
            time.sleep(0.2)
        else:
            print(f"  [fail] {dalle_name}")

    return downloaded


# =====================================================
#  Fallback Copernicus GLO-30
# =====================================================

def try_copernicus():
    os.makedirs(DEM_DIR, exist_ok=True)

    # la zone Chamonix tombe dans la tuile N45_E006
    lat = int(math.floor(BBOX_WGS84["lat_min"]))
    lon = int(math.floor(BBOX_WGS84["lon_min"]))

    lat_str = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
    lon_str = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"

    tile_name = f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"
    s3_url = f"https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{tile_name}/{tile_name}.tif"

    out_path = os.path.join(DEM_DIR, "copernicus_glo30_raw.tif")

    if os.path.exists(out_path):
        print(f"[copernicus] tuile deja en cache: {out_path}")
        return [out_path]

    print(f"[copernicus] download {tile_name}...")

    try:
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            resp = client.get(s3_url)
            resp.raise_for_status()

        with open(out_path, "wb") as f:
            f.write(resp.content)
        print(f"[copernicus] ok ({len(resp.content) / 1024 / 1024:.1f} MB)")
        return [out_path]

    except Exception as e:
        print(f"[copernicus] echec: {e}")
        return []


# =====================================================
#  Merge + reproject + crop
# =====================================================

def merge_and_process(tile_paths):
    if not tile_paths:
        print("[error] aucune dalle a traiter")
        return False

    output_path = get_output_path()
    bbox = BBOX_L93

    # merge si plusieurs dalles
    print(f"[merge] {len(tile_paths)} dalles...")
    datasets = [rasterio.open(p) for p in tile_paths]

    if len(datasets) == 1:
        mosaic = datasets[0].read(1)
        mosaic_transform = datasets[0].transform
        mosaic_crs = datasets[0].crs
        src_nodata = datasets[0].nodata
    else:
        mosaic, mosaic_transform = merge(datasets, nodata=NODATA_VALUE)
        mosaic = mosaic[0]  # premiere bande
        mosaic_crs = datasets[0].crs
        src_nodata = NODATA_VALUE

    for ds in datasets:
        ds.close()

    # target transform pour la bbox L93 a la resolution voulue
    width = int(round((bbox["xmax"] - bbox["xmin"]) / DEM_RESOLUTION))
    height = int(round((bbox["ymax"] - bbox["ymin"]) / DEM_RESOLUTION))

    dst_transform = from_bounds(
        bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"],
        width, height,
    )

    dst_array = np.full((height, width), NODATA_VALUE, dtype=np.float32)

    print(f"[reproject] -> L93, {width}x{height}px, res={DEM_RESOLUTION}m")

    # les dalles lidar sont deja en L93, mais on reproject quand meme pour gerer le crop + resample proprement
    reproject(
        source=mosaic.astype(np.float32),
        destination=dst_array,
        src_transform=mosaic_transform,
        src_crs=mosaic_crs,
        dst_transform=dst_transform,
        dst_crs=CRS_L93,
        resampling=Resampling.bilinear,
        src_nodata=src_nodata if src_nodata is not None else NODATA_VALUE,
        dst_nodata=NODATA_VALUE,
    )

    # nettoyage: valeurs aberrantes = artefacts d'interpolation pres du nodata
    bad_mask = dst_array < -100
    if bad_mask.any():
        print(f"  [fix] {bad_mask.sum()} pixels aberrants -> nodata")
        dst_array[bad_mask] = NODATA_VALUE

    # ecriture du resultat
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": width,
        "height": height,
        "count": 1,
        "crs": CRS_L93,
        "transform": dst_transform,
        "nodata": NODATA_VALUE,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(dst_array, 1)

    print(f"[ok] DEM ecrit: {output_path}")
    return True


def validate():
    path = get_output_path()
    if not os.path.exists(path):
        print("[validation] FAIL - fichier absent")
        return False

    with rasterio.open(path) as ds:
        data = ds.read(1)
        valid = data[data != NODATA_VALUE]

        if len(valid) == 0:
            print("[validation] FAIL - que du nodata")
            return False

        nodata_pct = (1 - len(valid) / data.size) * 100
        alt_min = valid.min()
        alt_max = valid.max()

        print(f"[validation] shape: {ds.shape}")
        print(f"  resolution: {ds.res}")
        print(f"  CRS: {ds.crs}")
        print(f"  altitude: {alt_min:.0f}m - {alt_max:.0f}m")
        print(f"  nodata: {nodata_pct:.1f}%")
        print(f"  taille fichier: {os.path.getsize(path) / 1024 / 1024:.1f} MB")

        # verif coherence pour le secteur Aiguille du Midi
        if alt_max < 3000:
            print("  [warn] alt max < 3000m, bizarre pour Chamonix")
        if alt_min < 0:
            print("  [warn] altitudes negatives detectees")
        if nodata_pct > 50:
            print("  [warn] beaucoup de nodata (>50%)")

        # c'est du copernicus 30m si la resolution est >10m
        if ds.res[0] > 10:
            print("  [info] resolution grossiere, probablement fallback Copernicus")

    return True


# =====================================================
#  Visu hillshade
# =====================================================

def plot_results():
    path = get_output_path()
    if not os.path.exists(path):
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)

    with rasterio.open(path) as ds:
        dem = ds.read(1)

    dem_display = np.where(dem == NODATA_VALUE, np.nan, dem)

    ls = LightSource(azdeg=315, altdeg=45)
    dem_filled = np.where(np.isnan(dem_display), 0, dem_display)
    hillshade = ls.hillshade(dem_filled, vert_exag=2, dx=DEM_RESOLUTION, dy=DEM_RESOLUTION)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # hillshade en gris + altitude en couleur par dessus
    ax.imshow(hillshade, cmap='gray')
    im = ax.imshow(dem_display, cmap='terrain', alpha=0.5,
                   vmin=np.nanmin(dem_display), vmax=np.nanmax(dem_display))

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, label='Altitude (m)')
    ax.set_title(f'DEM Aiguille du Midi - {DEM_RESOLUTION}m', fontsize=12)
    ax.set_xlabel('pixels')
    ax.set_ylabel('pixels')

    out_path = os.path.join(FIGURES_DIR, "dem_hillshade.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[plot] {out_path}")


# =====================================================

def main():
    print("=" * 50)
    print("T01 - Download DEM Aiguille du Midi")
    print(f"  resolution cible: {DEM_RESOLUTION}m")
    print(f"  bbox L93: {BBOX_L93}")
    print("=" * 50)

    os.makedirs(DEM_DIR, exist_ok=True)

    # check cache
    if check_cache():
        print("\nDEM deja pret, rien a faire")
        plot_results()
        return

    # check dalles brutes deja presentes (dl manuel par ex)
    raw_dir = os.path.join(DEM_DIR, "raw_ign")
    existing_raw = []
    if os.path.isdir(raw_dir):
        existing_raw = [
            os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
            if f.endswith(".tif")
        ]
        if existing_raw:
            print(f"[local] {len(existing_raw)} dalles brutes trouvees dans {raw_dir}")

    tile_paths = []

    if not existing_raw:
        # essai IGN
        print("\n--- Tentative IGN Lidar HD ---")
        tile_paths = try_ign_lidar()

        if not tile_paths:
            # fallback copernicus
            print("\n--- Fallback Copernicus GLO-30 ---")
            tile_paths = try_copernicus()
    else:
        tile_paths = existing_raw

    if not tile_paths:
        print("\n[ECHEC] aucune source DEM disponible")
        sys.exit(1)

    # merge + reproject + crop
    print("\n--- Traitement ---")
    if not merge_and_process(tile_paths):
        sys.exit(1)

    # validation
    print("\n--- Validation ---")
    if not validate():
        sys.exit(1)

    # visu
    print("\n--- Visualisation ---")
    plot_results()

    print("\nDone!")


if __name__ == "__main__":
    main()
