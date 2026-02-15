# A02 - test endpoints IGN (WCS, WFS) + fallback GLO-30

import os
import sys
import time
import tempfile
import httpx
import rasterio

from config import BASE_DIR, BBOX_L93, DEM_DIR

HTTP_TIMEOUT = 30

# -- endpoints a tester
WFS_URL = "https://data.geopf.fr/wfs/ows"
WCS_URL = "https://data.geopf.fr/wcs/ows"
WFS_TYPENAME = "IGNF_MNT-LIDAR-HD:dalle"


def test_wfs():
    """Verifie que le WFS LIDAR HD repond toujours."""
    print("\n--- Test WFS (dalles Lidar HD) ---")

    # GetCapabilities
    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetCapabilities",
    }

    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            resp = client.get(WFS_URL, params=params)
            resp.raise_for_status()

        # check que c'est du XML valide avec le typename qu'on cherche
        content = resp.text[:5000]
        has_lidar = "IGNF_MNT-LIDAR-HD" in content or "MNT-LIDAR-HD" in content
        print(f"  GetCapabilities: OK ({len(resp.text)} chars)")
        print(f"  Layer MNT-LIDAR-HD present: {'OUI' if has_lidar else 'a verifier (peut etre plus loin dans le XML)'}")
    except Exception as e:
        print(f"  GetCapabilities: ERREUR - {e}")
        return False

    # GetFeature sur Chamonix (deja valide en T01)
    bbox = BBOX_L93
    bbox_str = f"{bbox['xmin']},{bbox['ymin']},{bbox['xmax']},{bbox['ymax']}"
    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": WFS_TYPENAME,
        "OUTPUTFORMAT": "application/json",
        "BBOX": f"{bbox_str},EPSG:2154",
        "COUNT": "5",
    }

    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            resp = client.get(WFS_URL, params=params)
            resp.raise_for_status()

        data = resp.json()
        features = data.get("features", [])
        print(f"  GetFeature Chamonix: {len(features)} dalles (limit 5)")

        if features:
            f = features[0]
            props = f.get("properties", {})
            print(f"    sample: name={props.get('name')}, url={props.get('url', '?')[:80]}...")
            # verif que l'url WMS-R est bien presente
            if props.get("url"):
                print("    [OK] URL WMS-R presente dans les features")
            else:
                print("    [WARN] pas d'URL dans les properties")
        return True

    except Exception as e:
        print(f"  GetFeature: ERREUR - {e}")
        return False


def test_wcs():
    """Cherche si un WCS existe pour le Lidar HD (serait plus simple que dalle par dalle)."""
    print("\n--- Test WCS (GetCoverage) ---")

    # GetCapabilities
    params = {
        "SERVICE": "WCS",
        "VERSION": "2.0.1",
        "REQUEST": "GetCapabilities",
    }

    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            resp = client.get(WCS_URL, params=params)
            resp.raise_for_status()

        content = resp.text
        print(f"  GetCapabilities: OK ({len(content)} chars)")

        # cherche les coverages liees au MNT / altitude / lidar
        keywords = ["LIDAR", "MNT", "ELEVATION", "ALTI"]
        found = []
        for kw in keywords:
            if kw.lower() in content.lower():
                found.append(kw)
        print(f"  Keywords trouves dans le XML: {found if found else 'aucun pertinent'}")

        # cherche les CoverageId
        import re
        coverage_ids = re.findall(r'<(?:wcs:)?CoverageId>([^<]+)</(?:wcs:)?CoverageId>', content)
        elevation_coverages = [c for c in coverage_ids if any(
            kw in c.upper() for kw in ["ELEV", "MNT", "LIDAR", "ALTI"]
        )]

        if elevation_coverages:
            print(f"  Coverages MNT/elevation trouvees:")
            for c in elevation_coverages[:10]:
                print(f"    - {c}")
        else:
            # affiche les premieres pour voir ce qui existe
            print(f"  Pas de coverage Lidar HD specifique.")
            if coverage_ids:
                print(f"  Coverages disponibles ({len(coverage_ids)} total, premiers 10):")
                for c in coverage_ids[:10]:
                    print(f"    - {c}")

    except Exception as e:
        print(f"  GetCapabilities: ERREUR - {e}")
        return False

    # si on a trouve une coverage elevation, test un GetCoverage
    if elevation_coverages:
        cov_id = elevation_coverages[0]
        print(f"\n  Test GetCoverage sur '{cov_id}'...")
        bbox = BBOX_L93
        # subset en L93 sur une petite zone (100m x 100m)
        params_gc = {
            "SERVICE": "WCS",
            "VERSION": "2.0.1",
            "REQUEST": "GetCoverage",
            "COVERAGEID": cov_id,
            "SUBSET": [
                f"x({bbox['xmin']},{bbox['xmin']+100})",
                f"y({bbox['ymin']},{bbox['ymin']+100})",
            ],
            "FORMAT": "image/tiff",
        }
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                resp = client.get(WCS_URL, params=params_gc)
                resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            print(f"    content-type: {content_type}")
            print(f"    taille: {len(resp.content)} bytes")

            if "tiff" in content_type or "image" in content_type:
                # essaie de lire avec rasterio
                tmp = os.path.join(tempfile.gettempdir(), "a02_wcs_test.tif")
                with open(tmp, "wb") as f:
                    f.write(resp.content)
                try:
                    with rasterio.open(tmp) as ds:
                        data = ds.read(1)
                        print(f"    [OK] GeoTIFF valide: shape={ds.shape}, crs={ds.crs}")
                        print(f"    valeurs: min={data.min():.1f}, max={data.max():.1f}")
                        return True
                except Exception as e:
                    print(f"    [WARN] GeoTIFF illisible: {e}")
                finally:
                    os.remove(tmp)
            else:
                # prob une erreur XML
                print(f"    [WARN] reponse non-TIFF: {resp.text[:300]}")
        except Exception as e:
            print(f"    GetCoverage ERREUR: {e}")

    return False


def test_copernicus_glo30():
    """Test dl d'une dalle Copernicus hors France (Gran Paradiso)."""
    print("\n--- Test Copernicus GLO-30 (hors France) ---")

    # Gran Paradiso (Italie): N45_E007
    tile_name = "Copernicus_DSM_COG_10_N45_00_E007_00_DEM"
    s3_url = f"https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{tile_name}/{tile_name}.tif"

    print(f"  URL: {s3_url}")

    try:
        # on teste juste le HEAD pour verifier que le fichier existe
        with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
            resp = client.head(s3_url)
            print(f"  HEAD status: {resp.status_code}")

            if resp.status_code == 200:
                size = resp.headers.get("content-length", "?")
                content_type = resp.headers.get("content-type", "?")
                print(f"  content-type: {content_type}")
                print(f"  taille: {int(size)/1024/1024:.1f} MB" if size != "?" else f"  taille: ?")

                # dl les premiers bytes pour verifier que c'est un TIFF (magic bytes)
                resp2 = client.get(s3_url, headers={"Range": "bytes=0-3"})
                if resp2.content[:2] == b'II' or resp2.content[:2] == b'MM':
                    print("  [OK] c'est bien un GeoTIFF (magic bytes OK)")
                else:
                    print(f"  [WARN] magic bytes inattendus: {resp2.content[:4]}")

                # test lecture via /vsicurl/ (rasterio)
                print("\n  Test lecture fenetree /vsicurl/...")
                try:
                    with rasterio.open(f"/vsicurl/{s3_url}") as ds:
                        print(f"    shape: {ds.shape}, crs: {ds.crs}")
                        print(f"    bounds: {ds.bounds}")
                        # lecture d'une petite fenetre (100x100 px)
                        from rasterio.windows import Window
                        win = Window(0, 0, 100, 100)
                        data = ds.read(1, window=win)
                        valid = data[data != ds.nodata] if ds.nodata else data.flatten()
                        if len(valid) > 0:
                            print(f"    sample: min={valid.min():.0f}m, max={valid.max():.0f}m")
                        print("    [OK] lecture fenetree OK")
                        return True
                except Exception as e:
                    print(f"    /vsicurl/ ERREUR: {e}")
                    print("    (normal si GDAL pas compile avec curl support)")
            else:
                print(f"  [FAIL] fichier inaccessible")

    except Exception as e:
        print(f"  ERREUR: {e}")

    return False


def main():
    print("=" * 60)
    print("A02 - Test API Geoplateforme IGN + fallback GLO-30")
    print("=" * 60)

    wfs_ok = test_wfs()
    wcs_ok = test_wcs()
    glo30_ok = test_copernicus_glo30()

    # resume
    print("\n" + "=" * 60)
    print("RESUME")
    print("=" * 60)
    print(f"  WFS (dalles Lidar HD): {'OK' if wfs_ok else 'FAIL'}")
    print(f"  WCS (GetCoverage):     {'OK' if wcs_ok else 'pas de layer Lidar HD'}")
    print(f"  Copernicus GLO-30:     {'OK' if glo30_ok else 'FAIL'}")

    if wcs_ok:
        print("\n  >> WCS disponible ! On pourrait simplifier le pipeline (1 requete = 1 crop)")
        print("  >> a evaluer: qualite du WCS vs dalle WMS-R, rate limits, etc.")
    else:
        print("\n  >> Pas de WCS Lidar HD. On reste sur le workflow WFS+WMS-R de T01.")
        print("  >> C'est pas un probleme, ca marche bien.")

    print("\nDone!")


if __name__ == "__main__":
    main()
