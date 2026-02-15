# A01 - check couverture Lidar

import json
import os
import time
import httpx

from config import BASE_DIR

WFS_URL = "https://data.geopf.fr/wfs/ows"
WFS_TYPENAME = "IGNF_MNT-LIDAR-HD:dalle"
HTTP_TIMEOUT = 30

# bbox L93 (EPSG:2154) pour les principaux massifs
MASSIFS = {
    "Alpes du Nord (Chamonix-Vanoise)": {
        "xmin": 950000, "ymin": 6480000, "xmax": 1050000, "ymax": 6580000,
    },
    "Alpes du Nord (Ecrins-Oisans)": {
        "xmin": 920000, "ymin": 6430000, "xmax": 980000, "ymax": 6480000,
    },
    "Alpes du Sud (Mercantour-Queyras)": {
        "xmin": 960000, "ymin": 6330000, "xmax": 1060000, "ymax": 6420000,
    },
    "Pyrenees Centrales": {
        "xmin": 440000, "ymin": 6160000, "xmax": 560000, "ymax": 6230000,
    },
    "Pyrenees Orientales": {
        "xmin": 560000, "ymin": 6170000, "xmax": 680000, "ymax": 6230000,
    },
    "Massif Central (Cantal-Sancy)": {
        "xmin": 620000, "ymin": 6440000, "xmax": 720000, "ymax": 6530000,
    },
    "Vosges": {
        "xmin": 980000, "ymin": 6760000, "xmax": 1060000, "ymax": 6860000,
    },
    "Jura": {
        "xmin": 890000, "ymin": 6580000, "xmax": 960000, "ymax": 6700000,
    },
    "Corse": {
        "xmin": 1180000, "ymin": 6070000, "xmax": 1250000, "ymax": 6210000,
    },
    # zoom: secteur test Chamonix
    "Chamonix (secteur test)": {
        "xmin": 1002000, "ymin": 6536000, "xmax": 1008000, "ymax": 6542000,
    },
}


def query_wfs_count(bbox, max_count=2000):
    """Requete WFS avec count limit, retourne nb features + un sample."""
    bbox_str = f"{bbox['xmin']},{bbox['ymin']},{bbox['xmax']},{bbox['ymax']}"
    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": WFS_TYPENAME,
        "OUTPUTFORMAT": "application/json",
        "BBOX": f"{bbox_str},EPSG:2154",
        "COUNT": str(max_count),
        # resulttype=hits serait mieux mais pas tous les WFS le supportent
    }

    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        resp = client.get(WFS_URL, params=params)
        resp.raise_for_status()

    data = resp.json()
    features = data.get("features", [])
    # WFS peut renvoyer numberMatched (total reel) vs numberReturned
    total = data.get("numberMatched", len(features))
    returned = data.get("numberReturned", len(features))
    return total, returned, features


def try_hits_count(bbox):
    """Essaie resultType=hits pour avoir le count sans telecharger."""
    bbox_str = f"{bbox['xmin']},{bbox['ymin']},{bbox['xmax']},{bbox['ymax']}"
    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": WFS_TYPENAME,
        "RESULTTYPE": "hits",
        "BBOX": f"{bbox_str},EPSG:2154",
    }
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            resp = client.get(WFS_URL, params=params)
            resp.raise_for_status()
        # la reponse hits est du XML avec numberMatched
        text = resp.text
        if "numberMatched" in text:
            import re
            m = re.search(r'numberMatched="(\d+)"', text)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return None


def estimate_coverage(nb_dalles, bbox):
    """Estime le % de couverture par rapport a la surface de la bbox (dalles 1km x 1km)."""
    width_km = (bbox["xmax"] - bbox["xmin"]) / 1000
    height_km = (bbox["ymax"] - bbox["ymin"]) / 1000
    total_possible = width_km * height_km
    if total_possible > 0:
        return nb_dalles / total_possible * 100
    return 0


def main():
    print("=" * 60)
    print("A01 - Couverture Lidar HD IGN par massif")
    print("=" * 60)

    results = {}

    for name, bbox in MASSIFS.items():
        print(f"\n--- {name} ---")
        width_km = (bbox["xmax"] - bbox["xmin"]) / 1000
        height_km = (bbox["ymax"] - bbox["ymin"]) / 1000
        print(f"  bbox L93: {bbox}")
        print(f"  zone: {width_km:.0f} x {height_km:.0f} km")

        # d'abord essayer hits (pas de dl)
        hits = try_hits_count(bbox)
        if hits is not None:
            print(f"  [hits] {hits} dalles (count rapide)")
            nb_dalles = hits
            sample_names = []
        else:
            # fallback: requete complete
            try:
                total, returned, features = query_wfs_count(bbox, max_count=50)
                nb_dalles = total
                sample_names = [
                    f.get("properties", {}).get("name", "?")
                    for f in features[:3]
                ]
                print(f"  [wfs] total={total}, returned={returned}")
                if sample_names:
                    print(f"  sample: {sample_names}")
            except Exception as e:
                print(f"  [erreur] {e}")
                nb_dalles = -1
                sample_names = []

        coverage = estimate_coverage(nb_dalles, bbox) if nb_dalles > 0 else 0
        covered = nb_dalles > 0
        remark = ""
        if nb_dalles == 0:
            remark = "aucune dalle, fallback GLO-30"
        elif coverage > 80:
            remark = "bien couvert"
        elif coverage > 30:
            remark = "partiellement couvert"
        elif nb_dalles > 0:
            remark = "couverture faible"
        elif nb_dalles < 0:
            remark = "erreur WFS"

        results[name] = {
            "nb_dalles": nb_dalles,
            "couvert": covered,
            "coverage_pct": round(coverage, 1),
            "bbox_km2": round(width_km * height_km),
            "remarque": remark,
        }

        print(f"  => {nb_dalles} dalles, ~{coverage:.1f}% de la bbox, {remark}")

        # rate limit
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("RECAPITULATIF")
    print("=" * 60)
    print(f"{'Massif':<40} {'Dalles':>8} {'Couv.%':>8} {'Statut':<20}")
    print("-" * 80)
    for name, r in results.items():
        status = "OUI" if r["couvert"] else "NON"
        print(f"{name:<40} {r['nb_dalles']:>8} {r['coverage_pct']:>7.1f}% {status:<5} {r['remarque']}")

    # export
    out_dir = os.path.join(BASE_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "a01_coverage_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[export] {json_path}")

    # decision
    print("\n--- Conclusion ---")
    all_covered = all(r["couvert"] for r in results.values())
    if all_covered:
        print("Tous les massifs ont au moins des dalles. Mais verifier la couverture reelle.")
    else:
        missing = [n for n, r in results.items() if not r["couvert"]]
        print(f"Massifs sans couverture Lidar HD: {missing}")
        print("  -> fallback GLO-30 necessaire pour ces zones")

    print("\nDone!")


if __name__ == "__main__":
    main()
