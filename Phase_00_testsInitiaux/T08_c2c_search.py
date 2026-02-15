# T08 - recherche routes CampToCamp API v6

import os
import json
import time
import httpx

from config import (
    MAPS_DIR,
    C2C_API_BASE, C2C_SEARCH_QUERY,
    C2C_RATE_LIMIT_S, C2C_REQUEST_TIMEOUT,
)


# =====================================================
#  Helpers HTTP
# =====================================================

def _get(url, params=None):
    """GET avec rate limit + retry basique sur 429."""
    time.sleep(C2C_RATE_LIMIT_S)
    try:
        r = httpx.get(url, params=params, timeout=C2C_REQUEST_TIMEOUT,
                      headers={"Accept-Language": "fr"})
        if r.status_code == 429:
            print("  [warn] rate limited, wait 5s...")
            time.sleep(5)
            r = httpx.get(url, params=params, timeout=C2C_REQUEST_TIMEOUT,
                          headers={"Accept-Language": "fr"})
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        print(f"  [error] {e}")
        return None


# =====================================================
#  Recherche waypoint
# =====================================================

def search_waypoint(query):
    print(f"\n--- Recherche waypoint : '{query}' ---")
    url = f"{C2C_API_BASE}/search"
    data = _get(url, params={"q": query, "t": "w", "limit": 10})
    if not data:
        return None

    waypoints = data.get("waypoints", {}).get("documents", [])
    if not waypoints:
        print("  aucun waypoint trouve")
        return None

    print(f"  {len(waypoints)} resultats")
    for wp in waypoints:
        elev = wp.get("elevation", "?")
        wtype = wp.get("waypoint_type", "?")
        locales = wp.get("locales", [])
        title = _get_title(locales)
        print(f"    id={wp['document_id']}  {title}  ({elev}m, {wtype})")

    # prendre le sommet qui match exactement le nom, sinon le plus haut
    summits = [w for w in waypoints if w.get("waypoint_type") == "summit"]
    candidates = summits if summits else waypoints

    # match exact sur le titre (FR)
    exact = [w for w in candidates
             if _get_title(w.get("locales", [])).lower() == query.lower()]
    if exact:
        best = exact[0]
    else:
        best = max(candidates, key=lambda w: w.get("elevation", 0))

    title = _get_title(best.get("locales", []))
    print(f"\n  -> choisi: {title} (id={best['document_id']}, {best.get('elevation')}m)")
    return best


def _get_title(locales):
    """Extrait le titre FR en priorite, sinon le premier dispo."""
    for loc in locales:
        if loc.get("lang") == "fr" and loc.get("title"):
            return loc["title"]
    for loc in locales:
        if loc.get("title"):
            return loc["title"]
    return "(sans titre)"


# =====================================================
#  Routes associees a un waypoint
# =====================================================

def fetch_routes(waypoint_id):
    print(f"\n--- Routes pour waypoint {waypoint_id} ---")
    url = f"{C2C_API_BASE}/routes"
    all_routes = []
    offset = 0
    limit = 50

    while True:
        data = _get(url, params={"w": waypoint_id, "limit": limit, "offset": offset})
        if not data:
            break

        docs = data.get("documents", [])
        total = data.get("total", 0)
        all_routes.extend(docs)
        print(f"  offset={offset}, got {len(docs)}, total={total}")

        if offset + limit >= total:
            break
        offset += limit

    print(f"  total routes recuperees: {len(all_routes)}")
    return all_routes


# =====================================================
#  Parsing / extraction des infos utiles
# =====================================================

def parse_route(doc):
    """Extrait les champs utiles d'un document route C2C."""
    locales = doc.get("locales", [])
    title = _get_title(locales)

    # geometry - le point representatif
    geom = doc.get("geometry", {})
    geom_geojson = geom.get("geom") if geom else None
    # c'est du GeoJSON string parfois, parfois un dict
    coords = None
    if geom_geojson:
        if isinstance(geom_geojson, str):
            try:
                g = json.loads(geom_geojson)
                coords = g.get("coordinates")
            except json.JSONDecodeError:
                pass
        elif isinstance(geom_geojson, dict):
            coords = geom_geojson.get("coordinates")

    return {
        "document_id": doc.get("document_id"),
        "title": title,
        "activities": doc.get("activities", []),
        "global_rating": doc.get("global_rating"),
        "engagement_rating": doc.get("engagement_rating"),
        "equipment_rating": doc.get("equipment_rating"),
        "elevation_min": doc.get("elevation_min"),
        "elevation_max": doc.get("elevation_max"),
        "height_diff_up": doc.get("height_diff_up"),
        "height_diff_down": doc.get("height_diff_down"),
        "orientations": doc.get("orientations"),
        "durations": doc.get("durations"),
        "coordinates": coords,
    }


# =====================================================
#  Filtrage
# =====================================================

# activites qu'on garde
KEEP_ACTIVITIES = {"mountain_climbing", "snow_ice_mixed", "skitouring"}
# activites excluantes si seules
EXCLUDE_SOLO = {"rock_climbing", "via_ferrata"}

# ordre cotation pour le tri
RATING_ORDER = [
    "F", "F+", "PD-", "PD", "PD+",
    "AD-", "AD", "AD+",
    "D-", "D", "D+",
    "TD-", "TD", "TD+",
    "ED-", "ED", "ED+",
    "ABO-", "ABO", "ABO+",
]


def filter_routes(routes):
    """Garde les routes alpi/mixte/ski, exclut escalade pure et via ferrata."""
    kept = []
    for r in routes:
        acts = set(r["activities"])
        # si que des activites excluantes, on skip
        if acts and acts.issubset(EXCLUDE_SOLO):
            continue
        # au moins une activite qu'on veut
        if acts & KEEP_ACTIVITIES:
            kept.append(r)

    # tri par cotation
    def sort_key(r):
        rat = r.get("global_rating") or ""
        try:
            return RATING_ORDER.index(rat)
        except ValueError:
            return 999

    kept.sort(key=sort_key)
    return kept


# =====================================================
#  Affichage console
# =====================================================

def print_routes_table(routes, label):
    print(f"\n--- {label} ({len(routes)} resultats) ---")
    for i, r in enumerate(routes, 1):
        rat = r["global_rating"] or "?"
        acts = "/".join(r["activities"])
        dplus = r["height_diff_up"] or 0
        elev = r["elevation_max"] or 0
        title = r["title"][:45]
        print(f"  #{i:<3} {rat:<5} {title:<47} {elev}m  D+ {dplus}m  [{acts}]")


def print_stats(routes):
    print("\n--- Stats ---")
    # par activite
    act_count = {}
    for r in routes:
        for a in r["activities"]:
            act_count[a] = act_count.get(a, 0) + 1
    print("  activites:")
    for a, c in sorted(act_count.items(), key=lambda x: -x[1]):
        print(f"    {a}: {c}")

    # distribution cotations
    rat_count = {}
    for r in routes:
        rat = r["global_rating"] or "?"
        rat_count[rat] = rat_count.get(rat, 0) + 1
    print("  cotations:")
    for rat in RATING_ORDER + ["?"]:
        if rat in rat_count:
            print(f"    {rat}: {rat_count[rat]}")


# =====================================================
#  Export GeoJSON
# =====================================================

def export_geojson(routes, waypoint):
    os.makedirs(MAPS_DIR, exist_ok=True)
    path = os.path.join(MAPS_DIR, "c2c_routes_aiguille_du_midi.geojson")

    features = []

    # feature pour le waypoint (sommet)
    wp_geom = waypoint.get("geometry", {})
    wp_coords = None
    if wp_geom:
        raw = wp_geom.get("geom")
        if isinstance(raw, str):
            try:
                wp_coords = json.loads(raw).get("coordinates")
            except json.JSONDecodeError:
                pass
        elif isinstance(raw, dict):
            wp_coords = raw.get("coordinates")

    if wp_coords:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": wp_coords},
            "properties": {
                "type": "waypoint",
                "name": _get_title(waypoint.get("locales", [])),
                "document_id": waypoint["document_id"],
                "elevation": waypoint.get("elevation"),
            },
        })

    # features pour chaque route (Point, pas LineString - c2c donne rarement le trace)
    for r in routes:
        coords = r.get("coordinates")
        if not coords:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": coords},
            "properties": {
                "type": "route",
                "name": r["title"],
                "document_id": r["document_id"],
                "activities": r["activities"],
                "global_rating": r["global_rating"],
                "engagement_rating": r["engagement_rating"],
                "elevation_max": r["elevation_max"],
                "height_diff_up": r["height_diff_up"],
            },
        })

    geojson = {"type": "FeatureCollection", "features": features}

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)

    size_kb = os.path.getsize(path) / 1024
    print(f"\n[export] {path} ({size_kb:.0f} KB, {len(features)} features)")
    return path


# =====================================================
#  Validation
# =====================================================

def validate(waypoint, all_routes, filtered_routes, geojson_path):
    print("\n--- Validation ---")
    ok = True

    # waypoint connu
    if waypoint["document_id"] == 38999:
        print("  [OK] waypoint id=38999 (Aiguille du Midi)")
    else:
        print(f"  [WARN] waypoint id={waypoint['document_id']} (attendu 38999)")

    elev = waypoint.get("elevation", 0)
    if 3840 <= elev <= 3845:
        print(f"  [OK] elevation {elev}m (~3842)")
    else:
        print(f"  [WARN] elevation {elev}m (attendu ~3842)")

    # nb routes
    if len(all_routes) > 30:
        print(f"  [OK] {len(all_routes)} routes totales (>30)")
    else:
        print(f"  [WARN] seulement {len(all_routes)} routes (attendu >30)")
        ok = False

    if len(filtered_routes) > 5:
        print(f"  [OK] {len(filtered_routes)} routes apres filtrage")
    else:
        print(f"  [WARN] seulement {len(filtered_routes)} routes apres filtrage")

    # geojson existe et non vide
    if os.path.exists(geojson_path) and os.path.getsize(geojson_path) > 100:
        print(f"  [OK] GeoJSON cree ({os.path.getsize(geojson_path)/1024:.0f} KB)")
    else:
        print(f"  [FAIL] GeoJSON manquant ou vide")
        ok = False

    return ok
