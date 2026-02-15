# T06 - export GPX + GeoJSON

import os
import sys
import json
import time
import numpy as np
import rasterio
import gpxpy
import gpxpy.gpx
from pyproj import Transformer
from shapely.geometry import LineString

from config import (
    DEM_DIR, MAPS_DIR, GPX_DIR, FIGURES_DIR,
    DEM_RESOLUTION, NODATA_VALUE,
    CRS_L93, CRS_WGS84,
    SIMPLIFY_TOLERANCE_M,
    START_POINT_WGS84, END_POINT_WGS84,
)


# =====================================================
#  Chargement
# =====================================================

def load_route_geojson():
    path = os.path.join(MAPS_DIR, f"route_requin_to_aiguille_{DEM_RESOLUTION}m.geojson")
    if not os.path.exists(path):
        print(f"[error] GeoJSON introuvable: {path}")
        print("  -> lancer T05 d'abord")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    feature = data["features"][0]
    coords_l93 = feature["geometry"]["coordinates"]  # [[x, y], ..
    props = feature.get("properties", {})

    coords = np.array(coords_l93)
    print(f"[load] {len(coords)} points, CRS={props.get('crs', '?')}")
    return coords, props


def load_dem():
    path = os.path.join(DEM_DIR, f"dem_aiguille_du_midi_{DEM_RESOLUTION}m.tif")
    if not os.path.exists(path):
        print(f"[error] DEM introuvable: {path}")
        sys.exit(1)
    ds = rasterio.open(path)
    return ds


#  reproj + elevation

def reproject_to_wgs84(coords_l93):
    """L93 [x, y] -> WGS84 [lon, lat]."""
    proj = Transformer.from_crs(CRS_L93, CRS_WGS84, always_xy=True)
    xs = coords_l93[:, 0]
    ys = coords_l93[:, 1]
    lons, lats = proj.transform(xs, ys)
    return np.column_stack([lons, lats])


def sample_elevations(coords_l93, dem_ds):
    """Echantillonne l'altitude du DEM pour chaque point L93."""
    elevations = np.zeros(len(coords_l93), dtype=np.float64)
    dem_data = dem_ds.read(1)

    for i, (x, y) in enumerate(coords_l93):
        col, row = ~dem_ds.transform * (x, y)
        row, col = int(round(row)), int(round(col))
        row = max(0, min(row, dem_data.shape[0] - 1))
        col = max(0, min(col, dem_data.shape[1] - 1))
        val = dem_data[row, col]
        elevations[i] = val if val != NODATA_VALUE else 0.0

    return elevations


#  Simplification Douglas-Pecker

def simplify_route(coords_l93, elevations, tolerance_m):
    """Simplifie en L93 (metres), retourne les indices gardes."""
    line = LineString(coords_l93)
    simplified = line.simplify(tolerance_m, preserve_topology=True)
    # retrouver les indices des points gardes
    # on matche par distance minimale (pas parfait mais ok pour export)
    simp_coords = np.array(simplified.coords)
    print(f"[simplify] {len(coords_l93)} -> {len(simp_coords)} pts (tol={tolerance_m}m)")
    return simp_coords


# =====================================================
#  Stats
# =====================================================

def compute_stats(coords_wgs84, elevations):
    """Stats basiques pour les metadata GPX/GeoJSON."""
    dz = np.diff(elevations)
    # distance en m (approx via coords L93 deja calculees, on reutilise les elevations)
    # on recalcule a partir des coords WGS84 pour etre propre ou sinon tester direct en l93

    dplus = float(np.sum(dz[dz > 0]))
    dminus = float(abs(np.sum(dz[dz < 0])))

    return {
        "dplus_m": round(dplus),
        "dminus_m": round(dminus),
        "elev_min": round(float(elevations.min())),
        "elev_max": round(float(elevations.max())),
        "elev_start": round(float(elevations[0])),
        "elev_end": round(float(elevations[-1])),
        "n_points": len(elevations),
    }


def compute_distance_2d(coords_l93):
    dx = np.diff(coords_l93[:, 0])
    dy = np.diff(coords_l93[:, 1])
    return float(np.sum(np.sqrt(dx**2 + dy**2)))


# =====================================================
#  Export GPX
# =====================================================

def export_gpx(coords_wgs84, elevations, stats, dist_m, suffix, props):
    os.makedirs(GPX_DIR, exist_ok=True)

    gpx = gpxpy.gpx.GPX()
    gpx.name = "Requin -> Aiguille du Midi"
    gpx.description = (
        f"Route optimale {DEM_RESOLUTION}m | "
        f"dist={dist_m/1000:.1f}km D+={stats['dplus_m']}m D-={stats['dminus_m']}m | "
        f"temps~{props.get('time_tobler_h', '?')}h"
    )
    gpx.creator = "AlpineRoute Optimizer"

    track = gpxpy.gpx.GPXTrack()
    track.name = gpx.name
    gpx.tracks.append(track)

    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)

    for (lon, lat), elev in zip(coords_wgs84, elevations):
        pt = gpxpy.gpx.GPXTrackPoint(
            latitude=float(lat),
            longitude=float(lon),
            elevation=float(elev),
        )
        segment.points.append(pt)

    fname = f"route_requin_to_aiguille_{DEM_RESOLUTION}m_{suffix}.gpx"
    path = os.path.join(GPX_DIR, fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(gpx.to_xml())

    size_kb = os.path.getsize(path) / 1024
    print(f"[gpx] {path} ({size_kb:.0f} KB, {len(coords_wgs84)} pts)")
    return path


# =====================================================
#  Export GeoJSON
# =====================================================

def export_geojson_wgs84(coords_wgs84, elevations, stats, dist_m, suffix, props):
    os.makedirs(MAPS_DIR, exist_ok=True)

    # coords 3D: [lon, lat, elevation]
    coords_3d = [
        [round(float(lon), 7), round(float(lat), 7), round(float(elev), 1)]
        for (lon, lat), elev in zip(coords_wgs84, elevations)
    ]

    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords_3d,
            },
            "properties": {
                "name": "Refuge du Requin -> Aiguille du Midi",
                "distance_m": round(dist_m),
                "distance_km": round(dist_m / 1000, 2),
                "dplus_m": stats["dplus_m"],
                "dminus_m": stats["dminus_m"],
                "elev_min": stats["elev_min"],
                "elev_max": stats["elev_max"],
                "n_points": stats["n_points"],
                "time_tobler_h": props.get("time_tobler_h"),
                "glacier_pct": props.get("glacier_pct"),
                "resolution_m": DEM_RESOLUTION,
                "crs": CRS_WGS84,
                "variant": suffix,
            },
        }],
    }

    fname = f"route_requin_to_aiguille_{DEM_RESOLUTION}m_wgs84_{suffix}.geojson"
    path = os.path.join(MAPS_DIR, fname)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2)

    size_kb = os.path.getsize(path) / 1024
    print(f"[geojson] {path} ({size_kb:.0f} KB, {stats['n_points']} pts)")
    return path


def validate_exports(files_created):
    print("\n--- Validation exports ---")
    ok = True

    for path in files_created:
        if not os.path.exists(path):
            print(f"  [FAIL] fichier manquant: {path}")
            ok = False
            continue

        size = os.path.getsize(path)
        if size == 0:
            print(f"  [FAIL] fichier vide: {path}")
            ok = False
            continue

        if path.endswith('.gpx'):
            with open(path, 'r') as f:
                content = f.read()
            n_trkpt = content.count('<trkpt')
            print(f"  [OK] {os.path.basename(path)} : {n_trkpt} trackpoints, {size/1024:.0f} KB")
            if n_trkpt == 0:
                print(f"    [FAIL] aucun trackpoint!")
                ok = False

        elif path.endswith('.geojson'):
            with open(path, 'r') as f:
                data = json.load(f)
            coords = data["features"][0]["geometry"]["coordinates"]
            n_pts = len(coords)
            # verif bbox coherente (secteur Chamonix normalemnt)
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            bbox_ok = (6.8 < min(lons) < 7.0 and 45.8 < min(lats) < 46.0)
            status = "OK" if bbox_ok else "WARN"
            print(f"  [{status}] {os.path.basename(path)} : {n_pts} pts, "
                  f"bbox=[{min(lons):.4f},{min(lats):.4f}]-[{max(lons):.4f},{max(lats):.4f}]")
            if not bbox_ok:
                print(f"    bbox hors zone Chamonix attendue")
                ok = False

            # verif elevations dispo
            has_elev = all(len(c) >= 3 for c in coords)
            if has_elev:
                elevs = [c[2] for c in coords]
                print(f"    elevations: {min(elevs):.0f}m - {max(elevs):.0f}m")
            else:
                print(f"    [WARN] pas d'elevation 3D")

    return ok


# =====================================================

def main():
    t0 = time.time()

    print("=" * 55)
    print("T06 - Export GPX / GeoJSON WGS84")
    print(f"  resolution: {DEM_RESOLUTION}m")
    print(f"  simplification: {SIMPLIFY_TOLERANCE_M}m (Douglas-Peucker)")
    print("=" * 55)

    print("\n--- Chargement route L93 ---")
    coords_l93, props = load_route_geojson()

    print("\n--- Chargement DEM ---")
    dem_ds = load_dem()
    print(f"  shape: {dem_ds.shape}")

    print("\n--- Echantillonnage elevations ---")
    elevations_full = sample_elevations(coords_l93, dem_ds)
    print(f"  elev range: {elevations_full.min():.0f}m - {elevations_full.max():.0f}m")

    print("\n--- Reprojection L93 -> WGS84 ---")
    coords_wgs84_full = reproject_to_wgs84(coords_l93)
    print(f"  lon: [{coords_wgs84_full[:,0].min():.5f}, {coords_wgs84_full[:,0].max():.5f}]")
    print(f"  lat: [{coords_wgs84_full[:,1].min():.5f}, {coords_wgs84_full[:,1].max():.5f}]")

    print("\n--- Simplification (Douglas-Peucker) ---")
    coords_l93_simp = simplify_route(coords_l93, elevations_full, SIMPLIFY_TOLERANCE_M)
    elevations_simp = sample_elevations(coords_l93_simp, dem_ds)
    coords_wgs84_simp = reproject_to_wgs84(coords_l93_simp)

    dem_ds.close()

    dist_full = compute_distance_2d(coords_l93)
    dist_simp = compute_distance_2d(coords_l93_simp)
    stats_full = compute_stats(coords_wgs84_full, elevations_full)
    stats_simp = compute_stats(coords_wgs84_simp, elevations_simp)
    stats_full["distance_m"] = round(dist_full)
    stats_simp["distance_m"] = round(dist_simp)

    print(f"\n  full:       {stats_full['n_points']} pts, {dist_full/1000:.2f} km")
    print(f"  simplified: {stats_simp['n_points']} pts, {dist_simp/1000:.2f} km")
    ratio = (1 - stats_simp['n_points'] / stats_full['n_points']) * 100
    print(f"  reduction:  {ratio:.1f}%")

    # --- exports ---
    print("\n--- Export GPX ---")
    files = []
    files.append(export_gpx(coords_wgs84_full, elevations_full, stats_full, dist_full, "full", props))
    files.append(export_gpx(coords_wgs84_simp, elevations_simp, stats_simp, dist_simp, "simplified", props))

    print("\n--- Export GeoJSON WGS84 ---")
    files.append(export_geojson_wgs84(coords_wgs84_full, elevations_full, stats_full, dist_full, "full", props))
    files.append(export_geojson_wgs84(coords_wgs84_simp, elevations_simp, stats_simp, dist_simp, "simplified", props))

    validate_exports(files)

    dt = time.time() - t0
    print(f"\nDone! ({dt:.1f}s)")


if __name__ == "__main__":
    main()
