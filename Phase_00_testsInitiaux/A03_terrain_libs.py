# A03 - comparaison richdem vs scipy pour slope/aspect/TRI (test avec gemini)

import os
import sys
import time
import numpy as np
import rasterio
import richdem as rd

from config import DEM_DIR, DEM_RESOLUTION, NODATA_VALUE

from T02_terrain_analysis import compute_slope_aspect, compute_roughness, make_nodata_mask


def load_dem():
    path = os.path.join(DEM_DIR, f"dem_aiguille_du_midi_{DEM_RESOLUTION}m.tif")
    if not os.path.exists(path):
        print(f"[error] DEM introuvable: {path}")
        sys.exit(1)
    with rasterio.open(path) as ds:
        dem = ds.read(1)
    return dem


def dem_to_rdarray(dem):
    """Convertit numpy array en rdarray pour richdem."""
    # richdem attend un rdarray avec metadata
    arr = rd.rdarray(dem.astype(np.float64), no_data=NODATA_VALUE)
    # on doit set la geotransform (meme fake, il en a besoin pour le cell_size)
    arr.geotransform = (0, DEM_RESOLUTION, 0, 0, 0, -DEM_RESOLUTION)
    return arr


def compare_arrays(a, b, name, mask):
    """Compare deux arrays en ignorant les nodata."""
    valid = ~mask
    # filtre aussi les nodata de richdem (souvent -1 ou nan)
    a_valid = a[valid].astype(np.float64)
    b_valid = b[valid].astype(np.float64)

    # pour l'aspect, il faut gerer la circularite (0 ~ 360)
    if "aspect" in name.lower():
        diff = np.abs(a_valid - b_valid)
        diff = np.minimum(diff, 360 - diff)
    else:
        diff = np.abs(a_valid - b_valid)

    print(f"\n  [{name}]")
    print(f"    pixels valides: {len(a_valid):,}")
    print(f"    diff absolue: mean={diff.mean():.4f}, median={np.median(diff):.4f}, "
          f"max={diff.max():.2f}")
    print(f"    diff <0.5:  {(diff < 0.5).sum() / len(diff) * 100:.2f}%")
    print(f"    diff <1.0:  {(diff < 1.0).sum() / len(diff) * 100:.2f}%")
    print(f"    diff <2.0:  {(diff < 2.0).sum() / len(diff) * 100:.2f}%")

    # stats individuelles
    print(f"    scipy: min={a_valid.min():.2f}, max={a_valid.max():.2f}, "
          f"mean={a_valid.mean():.2f}")
    print(f"    richdem: min={b_valid.min():.2f}, max={b_valid.max():.2f}, "
          f"mean={b_valid.mean():.2f}")

    return diff


def main():
    print("=" * 60)
    print("A03 - Comparaison richdem vs scipy (T02)")
    print(f"  resolution: {DEM_RESOLUTION}m")
    print("=" * 60)

    dem = load_dem()
    mask = make_nodata_mask(dem, dilate=True)
    print(f"[load] DEM: {dem.shape}, {mask.sum()} pixels nodata")

    # =========================================
    #  Baseline scipy (T02)
    # =========================================
    print("\n--- Baseline scipy (T02) ---")
    t0 = time.time()
    slope_scipy, aspect_scipy = compute_slope_aspect(dem)
    dt_scipy_sa = time.time() - t0
    print(f"  slope+aspect: {dt_scipy_sa:.2f}s")

    t0 = time.time()
    rough_scipy = compute_roughness(dem)
    dt_scipy_r = time.time() - t0
    print(f"  roughness: {dt_scipy_r:.2f}s")

    # =========================================
    #  richdem
    # =========================================
    print("\n--- richdem ---")
    rda = dem_to_rdarray(dem)

    # slope
    t0 = time.time()
    slope_rd = rd.TerrainAttribute(rda, attrib='slope_degrees')
    dt_rd_slope = time.time() - t0
    slope_rd_np = np.array(slope_rd, dtype=np.float32)
    # richdem met -1 pour nodata
    slope_rd_np[slope_rd_np < 0] = NODATA_VALUE
    print(f"  slope: {dt_rd_slope:.2f}s")

    # aspect
    t0 = time.time()
    aspect_rd = rd.TerrainAttribute(rda, attrib='aspect')
    dt_rd_aspect = time.time() - t0
    aspect_rd_np = np.array(aspect_rd, dtype=np.float32)
    aspect_rd_np[aspect_rd_np < 0] = NODATA_VALUE
    print(f"  aspect: {dt_rd_aspect:.2f}s")

    # TRI (terrain ruggedness index)
    # richdem n'a pas de TRI direct, mais a le "roughness" qui est different
    # testons ce qui est dispo
    has_tri = False
    try:
        t0 = time.time()
        # essayons differents noms
        for attr_name in ['tri', 'ruggedness', 'roughness']:
            try:
                rough_rd = rd.TerrainAttribute(rda, attrib=attr_name)
                dt_rd_rough = time.time() - t0
                rough_rd_np = np.array(rough_rd, dtype=np.float32)
                rough_rd_np[rough_rd_np < 0] = NODATA_VALUE
                print(f"  {attr_name}: {dt_rd_rough:.2f}s")
                has_tri = True
                break
            except Exception:
                continue
        if not has_tri:
            print("  TRI/roughness: pas disponible dans richdem")
    except Exception as e:
        print(f"  TRI test: {e}")

    # =========================================
    #  Comparaison
    # =========================================
    print("\n--- Comparaison ---")

    # slope
    diff_slope = compare_arrays(slope_scipy, slope_rd_np, "Slope", mask)

    # aspect
    diff_aspect = compare_arrays(aspect_scipy, aspect_rd_np, "Aspect", mask)

    # roughness (si dispo)
    if has_tri:
        diff_rough = compare_arrays(rough_scipy, rough_rd_np, "Roughness/TRI", mask)

    # =========================================
    #  Benchmark timing
    # =========================================
    print("\n--- Benchmark ---")
    print(f"  scipy  slope+aspect: {dt_scipy_sa:.2f}s")
    print(f"  richdem slope:       {dt_rd_slope:.2f}s")
    print(f"  richdem aspect:      {dt_rd_aspect:.2f}s")
    print(f"  richdem total:       {dt_rd_slope + dt_rd_aspect:.2f}s")
    print(f"  scipy  roughness:    {dt_scipy_r:.2f}s")

    # =========================================
    #  Gestion nodata
    # =========================================
    print("\n--- Gestion nodata ---")
    # check: est-ce que richdem propage mieux les nodata pres des bords ?
    nodata_scipy = (slope_scipy == NODATA_VALUE)
    nodata_rd = (slope_rd_np == NODATA_VALUE) | np.isnan(slope_rd_np)
    print(f"  nodata scipy slope: {nodata_scipy.sum():,} px")
    print(f"  nodata richdem slope: {nodata_rd.sum():,} px")

    # pixels valides dans richdem mais nodata dans scipy (= richdem gere mieux les bords ?)
    extra_rd = nodata_scipy & ~nodata_rd
    extra_scipy = ~nodata_scipy & nodata_rd
    print(f"  richdem a des valeurs la ou scipy a nodata: {extra_rd.sum():,} px")
    print(f"  scipy a des valeurs la ou richdem a nodata: {extra_scipy.sum():,} px")

    # =========================================
    #  Decision
    # =========================================
    print("\n--- Conclusion ---")

    # critere principal : diff slope < 0.5 deg pour > 99%
    valid_mask = ~mask
    slope_diff_valid = np.abs(slope_scipy[valid_mask].astype(np.float64) -
                              slope_rd_np[valid_mask].astype(np.float64))
    pct_sub_05 = (slope_diff_valid < 0.5).sum() / len(slope_diff_valid) * 100

    print(f"  Slopes concordent a <0.5 deg: {pct_sub_05:.2f}%")
    if pct_sub_05 > 99:
        print("  -> CRITERE VALIDE (>99%)")
    else:
        print(f"  -> critere non atteint ({pct_sub_05:.1f}% < 99%)")

    print("\nDone!")


if __name__ == "__main__":
    main()
