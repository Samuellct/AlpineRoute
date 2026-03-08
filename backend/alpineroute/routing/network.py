# client Valhalla -- routage reseau OSM pieton
# utilise pour le mode hybride (sentiers OSM + hors-piste raster)

import math
import httpx

from alpineroute.config import (
    VALHALLA_BASE_URL,
    VALHALLA_TIMEOUT_S,
    VALHALLA_MAX_HIKING_DIFFICULTY,
    VALHALLA_DETOUR_THRESHOLD,
    VALHALLA_MIN_DIRECT_M,
)
from alpineroute.utils import ValhallaError, setup_logging

log = setup_logging(__name__)


def decode_valhalla_shape(encoded):
    """Decode une polyline Google encodee precision 6 (specifique Valhalla).
    Retourne list de (lat, lon)."""
    coords = []
    i = 0
    lat = 0
    lng = 0
    while i < len(encoded):
        # lat
        shift = 0
        result = 0
        while True:
            b = ord(encoded[i]) - 63
            i += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        lat += (~(result >> 1) if (result & 1) else (result >> 1))

        # lng
        shift = 0
        result = 0
        while True:
            b = ord(encoded[i]) - 63
            i += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        lng += (~(result >> 1) if (result & 1) else (result >> 1))

        coords.append((lat / 1e6, lng / 1e6))

    return coords


def valhalla_available():
    """Check si Valhalla repond. Retourne True/False, jamais d'exception."""
    try:
        r = httpx.get(f"{VALHALLA_BASE_URL}/status", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def valhalla_locate(point_wgs84, max_difficulty=None):
    """Snap un point WGS84 (lat, lon) sur le reseau Valhalla.
    Retourne le JSON brut ou None si echec."""
    if max_difficulty is None:
        max_difficulty = VALHALLA_MAX_HIKING_DIFFICULTY

    lat, lon = point_wgs84
    payload = {
        "locations": [{"lat": lat, "lon": lon}],
        "costing": "pedestrian",
        "costing_options": {
            "pedestrian": {"max_hiking_difficulty": max_difficulty}
        },
    }
    try:
        r = httpx.post(
            f"{VALHALLA_BASE_URL}/locate",
            json=payload,
            timeout=VALHALLA_TIMEOUT_S,
        )
        if r.status_code == 200:
            return r.json()
        log.warning("locate %s -> HTTP %d", point_wgs84, r.status_code)
        return None
    except Exception as e:
        log.warning("locate %s echec: %s", point_wgs84, e)
        return None


def valhalla_route(start_wgs84, end_wgs84, max_difficulty=None):
    """Calcule une route pieton Valhalla entre deux points WGS84 (lat, lon).
    Retourne dict {coords, distance_km, duration_s, shape_encoded} ou None si pas de route.
    Raise ValhallaError sur erreur reseau/serveur."""
    if max_difficulty is None:
        max_difficulty = VALHALLA_MAX_HIKING_DIFFICULTY

    lat1, lon1 = start_wgs84
    lat2, lon2 = end_wgs84
    payload = {
        "locations": [
            {"lat": lat1, "lon": lon1},
            {"lat": lat2, "lon": lon2},
        ],
        "costing": "pedestrian",
        "costing_options": {
            "pedestrian": {"max_hiking_difficulty": max_difficulty}
        },
        "units": "kilometers",
    }

    try:
        r = httpx.post(
            f"{VALHALLA_BASE_URL}/route",
            json=payload,
            timeout=VALHALLA_TIMEOUT_S,
        )
    except httpx.TimeoutException:
        raise ValhallaError(f"Valhalla timeout ({VALHALLA_TIMEOUT_S}s)")
    except httpx.HTTPError as e:
        raise ValhallaError(f"Valhalla HTTP error: {e}")

    # 400 = pas de route trouvee (normal en montagne)
    if r.status_code == 400:
        log.info("Pas de route Valhalla %s -> %s", start_wgs84, end_wgs84)
        return None

    if r.status_code != 200:
        raise ValhallaError(f"Valhalla HTTP {r.status_code}: {r.text[:200]}")

    try:
        data = r.json()
        trip = data["trip"]
        leg = trip["legs"][0]
        shape_enc = leg["shape"]
        coords = decode_valhalla_shape(shape_enc)
        summary = trip["summary"]

        # parse maneuvers pour detection detours (Phase 7)
        maneuvers = []
        for m in leg.get("maneuvers", []):
            maneuvers.append({
                "begin_shape_index": m.get("begin_shape_index", 0),
                "end_shape_index": m.get("end_shape_index", 0),
                "length_km": m.get("length", 0),
            })

        return {
            "coords": coords,
            "distance_km": summary["length"],
            "duration_s": summary["time"],
            "shape_encoded": shape_enc,
            "maneuvers": maneuvers,
        }
    except (KeyError, IndexError) as e:
        raise ValhallaError(f"Reponse Valhalla inattendue: {e}")


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance en km entre deux points WGS84 (formule haversine)."""
    R = 6371.0
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def is_detour_excessive(valhalla_distance_km, start_wgs84, end_wgs84):
    """Verifie si la route Valhalla fait un detour absurde par rapport au vol d'oiseau.
    Retourne True si ratio > seuil (sauf si distance < VALHALLA_MIN_DIRECT_M)."""
    lat1, lon1 = start_wgs84
    lat2, lon2 = end_wgs84
    direct_km = haversine_km(lat1, lon1, lat2, lon2)

    # court trajet -> on laisse passer
    if direct_km < VALHALLA_MIN_DIRECT_M / 1000:
        return False

    ratio = valhalla_distance_km / direct_km
    if ratio > VALHALLA_DETOUR_THRESHOLD:
        log.info("detour excessif: %.1f km Valhalla / %.1f km direct = ratio %.1f",
                 valhalla_distance_km, direct_km, ratio)
        return True
    return False
