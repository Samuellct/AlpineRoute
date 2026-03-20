# graphe GPX overlay -- reseau topologique des traces indexees
# connecte au reseau OSM via portails Valhalla
# recherche par corridor (pas par rayon autour start/end)

import math
import logging
import os

import networkx as nx

from alpineroute.config import (
    GPX_DIR, GPX_SUBSAMPLE_M, GPX_MERGE_TOLERANCE_M,
    GPX_PORTAL_SNAP_M, GPX_CORRIDOR_RATIO, GPX_CORRIDOR_MIN_M,
    GPX_ROUTE_TRAIL_COST,
)
from alpineroute.alpine.routes import load_gpx
from alpineroute.alpine.index import load_index

logger = logging.getLogger(__name__)

# ---- cache module-level ----
_gpx_graph = None       # nx.Graph
_gpx_portals = None     # list[dict]
_node_coords = {}       # node_id -> (lat, lon, alt)
_portals_ready = False

_next_node_id = 0


# ---- helpers ----

def _haversine_m(lat1, lon1, lat2, lon2):
    """Distance haversine en metres."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _subsample_points(points, min_dist_m=None):
    """Sous-echantillonne une trace GPX, garde premier+dernier."""
    if min_dist_m is None:
        min_dist_m = GPX_SUBSAMPLE_M
    if len(points) <= 2:
        return list(points)

    result = [points[0]]
    cumul = 0.0
    for i in range(1, len(points) - 1):
        lat1, lon1, _ = points[i - 1]
        lat2, lon2, _ = points[i]
        cumul += _haversine_m(lat1, lon1, lat2, lon2)
        if cumul >= min_dist_m:
            result.append(points[i])
            cumul = 0.0
    result.append(points[-1])
    return result


def _find_nearby_node(lat, lon, node_coords, tolerance_m=None):
    """Scan lineaire: retourne node_id si un noeud est dans le rayon."""
    if tolerance_m is None:
        tolerance_m = GPX_MERGE_TOLERANCE_M
    best_id = None
    best_dist = tolerance_m
    for nid, (nlat, nlon, _) in node_coords.items():
        d = _haversine_m(lat, lon, nlat, nlon)
        if d < best_dist:
            best_dist = d
            best_id = nid
    return best_id


def _new_node_id():
    global _next_node_id
    nid = _next_node_id
    _next_node_id += 1
    return nid


def _point_to_segment_dist_m(plat, plon, lat1, lon1, lat2, lon2):
    """Distance perpendiculaire d'un point a un segment [A,B] en metres.
    Utilise une projection cartesienne locale (suffisant pour <100km)."""
    mid_lat = math.radians((lat1 + lat2) / 2)
    cos_lat = math.cos(mid_lat)
    # conversion en metres relatifs au point A
    ax, ay = 0.0, 0.0
    bx = (lon2 - lon1) * cos_lat * 111320
    by = (lat2 - lat1) * 111320
    px = (plon - lon1) * cos_lat * 111320
    py = (plat - lat1) * 111320

    # projection sur le segment
    seg_len_sq = bx * bx + by * by
    if seg_len_sq < 1e-6:
        return math.sqrt(px * px + py * py), 0.0

    t = (px * bx + py * by) / seg_len_sq
    t = max(0, min(1, t))
    proj_x = ax + t * bx
    proj_y = ay + t * by
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2), t


# ---- construction du graphe ----

def build_gpx_graph(entries, gpx_dir=None):
    """Construit le graphe GPX depuis les entries d'index.
    Return (nx.Graph, node_coords_dict)."""
    if gpx_dir is None:
        gpx_dir = GPX_DIR

    global _next_node_id
    _next_node_id = 0

    G = nx.Graph()
    node_coords = {}

    for entry in entries:
        gpx_file = entry["gpx_file"]
        gpx_path = os.path.join(gpx_dir, gpx_file)
        points = load_gpx(gpx_path)
        if points is None:
            continue

        sub = _subsample_points(points)
        if len(sub) < 2:
            continue

        # trail cost: fixe pour les routes, configurable pour les segments
        if entry["type"] == "route":
            tc = GPX_ROUTE_TRAIL_COST
        else:
            tc = entry.get("trail_cost", 0.30)

        # creer les noeuds de cette trace
        trace_nodes = []
        for i, (lat, lon, alt) in enumerate(sub):
            is_endpoint = (i == 0 or i == len(sub) - 1)
            # fusion uniquement pour les extremites (inter-traces)
            merged = None
            if is_endpoint:
                merged = _find_nearby_node(lat, lon, node_coords)

            if merged is not None:
                trace_nodes.append(merged)
            else:
                nid = _new_node_id()
                node_coords[nid] = (lat, lon, alt)
                G.add_node(nid, lat=lat, lon=lon, alt=alt)
                trace_nodes.append(nid)

        # edges entre pts consecutifs
        for i in range(len(trace_nodes) - 1):
            n1, n2 = trace_nodes[i], trace_nodes[i + 1]
            if n1 == n2:
                continue
            c1 = node_coords[n1]
            c2 = node_coords[n2]
            dist_m = _haversine_m(c1[0], c1[1], c2[0], c2[1])
            G.add_edge(n1, n2,
                       weight=dist_m,
                       trail_cost=tc,
                       gpx_source=gpx_file,
                       entry_type=entry["type"])

        logger.debug("gpx trace %s: %d nodes, %d subsampled",
                     gpx_file, len(points), len(sub))

    return G, node_coords


# ---- portails Valhalla (lazy) ----

def _ensure_portals():
    """Detection lazy des portails. Appele au premier route_via_gpx()."""
    global _portals_ready, _gpx_portals

    if _portals_ready:
        return
    if _gpx_graph is None:
        logger.debug("gpx portals: pas de graphe, skip")
        return

    from alpineroute.routing.network import valhalla_available
    try:
        if not valhalla_available():
            logger.info("gpx portals: Valhalla indisponible, retry plus tard")
            return
    except Exception:
        return

    _gpx_portals = _detect_portals(_gpx_graph, _node_coords)
    _portals_ready = True
    logger.info("gpx portals: %d detectes sur %d candidats",
                len(_gpx_portals), _count_portal_candidates(_gpx_graph))


def _count_portal_candidates(graph):
    """Compte les noeuds candidats portail (pour le log)."""
    return sum(1 for n in graph.nodes()
               if graph.degree(n) == 1 or graph.degree(n) >= 3)


def _detect_portals(graph, node_coords):
    """Detecte les portails GPX-OSM via valhalla_locate."""
    from alpineroute.routing.network import valhalla_locate, parse_locate_snap

    portals = []
    candidates = set()
    for n in graph.nodes():
        deg = graph.degree(n)
        if deg == 1 or deg >= 3:
            candidates.add(n)

    for nid in candidates:
        lat, lon, alt = node_coords[nid]
        loc = valhalla_locate((lat, lon))
        snap_pt = parse_locate_snap(loc)
        if snap_pt is None:
            logger.debug("portal candidat node=%d (%.5f,%.5f): locate echec",
                         nid, lat, lon)
            continue

        snap_m = _haversine_m(lat, lon, snap_pt[0], snap_pt[1])
        if snap_m <= GPX_PORTAL_SNAP_M:
            portals.append({
                "node_id": nid,
                "gpx_coords": (lat, lon),
                "osm_coords": snap_pt,
                "snap_m": round(snap_m, 1),
            })
            logger.debug("portal OK node=%d snap=%.0fm", nid, snap_m)
        else:
            logger.debug("portal trop loin node=%d snap=%.0fm > %.0fm",
                         nid, snap_m, GPX_PORTAL_SNAP_M)

    return portals


# ---- recherche par corridor ----

def _portals_in_corridor(start_wgs84, end_wgs84):
    """Trouve les portails dans un corridor autour de la ligne start-end.
    Largeur corridor = GPX_CORRIDOR_RATIO * distance, min GPX_CORRIDOR_MIN_M."""
    if not _gpx_portals:
        return []

    lat1, lon1 = start_wgs84
    lat2, lon2 = end_wgs84
    route_dist_m = _haversine_m(lat1, lon1, lat2, lon2)
    corridor_width = max(GPX_CORRIDOR_MIN_M, route_dist_m * GPX_CORRIDOR_RATIO)

    result = []
    for p in _gpx_portals:
        plat, plon = p["gpx_coords"]
        dist_and_t = _point_to_segment_dist_m(plat, plon, lat1, lon1, lat2, lon2)
        perp_dist, t = dist_and_t
        # accepter les portails legerement avant/apres la ligne (-0.3 a 1.3)
        if perp_dist <= corridor_width and -0.3 <= t <= 1.3:
            result.append({**p, "_perp_dist": perp_dist, "_t": t})

    logger.info("corridor search: route_dist=%.0fm, corridor=%.0fm, "
                "%d/%d portails dans le corridor",
                route_dist_m, corridor_width, len(result), len(_gpx_portals))
    return result


def _closest_gpx_node_to(target_wgs84, component_nodes, node_coords):
    """Trouve le noeud du graphe le plus proche du point cible
    dans un ensemble de noeuds."""
    lat, lon = target_wgs84
    best_nid = None
    best_dist = float("inf")
    for nid in component_nodes:
        nlat, nlon, _ = node_coords[nid]
        d = _haversine_m(lat, lon, nlat, nlon)
        if d < best_dist:
            best_dist = d
            best_nid = nid
    return best_nid, best_dist


# ---- routage via le graphe GPX ----

def route_via_gpx(start_wgs84, end_wgs84):
    """Tente de router via le graphe GPX.
    Recherche par corridor : portails le long de la ligne start-end.
    Supporte couverture partielle (un seul portail OSM).
    Return dict avec coords/portails/stats/coverage, ou None."""
    _ensure_portals()

    if _gpx_graph is None or _gpx_graph.number_of_nodes() == 0:
        logger.info("route_via_gpx: pas de graphe GPX")
        return None

    if not _gpx_portals:
        logger.info("route_via_gpx: aucun portail detecte")
        return None

    # chercher portails dans le corridor
    corridor = _portals_in_corridor(start_wgs84, end_wgs84)
    if not corridor:
        logger.info("route_via_gpx: aucun portail dans le corridor")
        return None

    # trier: entry = portal le plus proche de start, exit = le plus proche de end
    by_start = sorted(corridor, key=lambda p: _haversine_m(
        start_wgs84[0], start_wgs84[1], p["gpx_coords"][0], p["gpx_coords"][1]))
    by_end = sorted(corridor, key=lambda p: _haversine_m(
        end_wgs84[0], end_wgs84[1], p["gpx_coords"][0], p["gpx_coords"][1]))

    # --- essai 1: couverture complete (2 portails connectes) ---
    for entry_p in by_start:
        for exit_p in by_end:
            if entry_p["node_id"] == exit_p["node_id"]:
                continue
            if nx.has_path(_gpx_graph, entry_p["node_id"], exit_p["node_id"]):
                path = nx.shortest_path(
                    _gpx_graph, entry_p["node_id"], exit_p["node_id"],
                    weight="weight")
                result = _build_gpx_result(path, entry_p, exit_p, "full")
                logger.info("route_via_gpx: FULL coverage, %.2fkm via %s",
                            result["distance_km"], result["gpx_sources"])
                return result

    # --- essai 2: couverture partielle (1 portail + extension vers la dest) ---
    best_partial = None
    best_usefulness = 0  # ratio distance GPX / distance start-end

    route_dist_m = _haversine_m(
        start_wgs84[0], start_wgs84[1], end_wgs84[0], end_wgs84[1])

    for portal in corridor:
        pid = portal["node_id"]
        component = nx.node_connected_component(_gpx_graph, pid)
        if len(component) < 2:
            continue

        # trouver le noeud le plus proche de la destination dans cette composante
        far_nid, far_dist = _closest_gpx_node_to(end_wgs84, component, _node_coords)
        if far_nid is None or far_nid == pid:
            continue

        # le noeud le plus eloigne doit etre plus proche de la dest que le portail
        portal_to_end = _haversine_m(
            portal["gpx_coords"][0], portal["gpx_coords"][1],
            end_wgs84[0], end_wgs84[1])
        if far_dist >= portal_to_end:
            continue  # le GPX ne rapproche pas de la destination

        path = nx.shortest_path(_gpx_graph, pid, far_nid, weight="weight")
        gpx_dist = sum(
            _haversine_m(_node_coords[path[i]][0], _node_coords[path[i]][1],
                         _node_coords[path[i+1]][0], _node_coords[path[i+1]][1])
            for i in range(len(path) - 1)
        )
        # "utilite" = combien le GPX avance vers la destination vs la distance totale
        advance_m = portal_to_end - far_dist
        usefulness = advance_m / max(route_dist_m, 1)

        # seuil: le GPX doit couvrir au moins 5% du trajet et > 200m
        if usefulness > best_usefulness and gpx_dist > 200 and usefulness > 0.05:
            best_usefulness = usefulness
            exit_node_coords = _node_coords[far_nid]
            best_partial = {
                "portal": portal,
                "far_nid": far_nid,
                "path": path,
                "gpx_dist": gpx_dist,
                "usefulness": usefulness,
                "remaining_m": far_dist,
            }

    # --- essai 3: reverse partial (portail pres de la dest, GPX vers le depart) ---
    best_reverse = None
    best_rev_usefulness = 0

    for portal in corridor:
        pid = portal["node_id"]
        component = nx.node_connected_component(_gpx_graph, pid)
        if len(component) < 2:
            continue

        # noeud le plus proche du depart dans cette composante
        near_nid, near_dist = _closest_gpx_node_to(start_wgs84, component, _node_coords)
        if near_nid is None or near_nid == pid:
            continue

        portal_to_start = _haversine_m(
            portal["gpx_coords"][0], portal["gpx_coords"][1],
            start_wgs84[0], start_wgs84[1])
        if near_dist >= portal_to_start:
            continue  # le GPX ne rapproche pas du depart

        path = nx.shortest_path(_gpx_graph, near_nid, pid, weight="weight")
        gpx_dist = sum(
            _haversine_m(_node_coords[path[i]][0], _node_coords[path[i]][1],
                         _node_coords[path[i+1]][0], _node_coords[path[i+1]][1])
            for i in range(len(path) - 1)
        )
        advance_m = portal_to_start - near_dist
        usefulness = advance_m / max(route_dist_m, 1)

        if usefulness > best_rev_usefulness and gpx_dist > 200 and usefulness > 0.05:
            best_rev_usefulness = usefulness
            best_reverse = {
                "portal": portal,
                "near_nid": near_nid,
                "path": path,
                "gpx_dist": gpx_dist,
                "usefulness": usefulness,
                "remaining_m": near_dist,
            }

    # choisir le meilleur entre forward et reverse
    if best_partial is not None and (best_reverse is None
                                     or best_usefulness >= best_rev_usefulness):
        # forward partial (portail cote depart)
        p = best_partial
        result = _build_gpx_result(
            p["path"], p["portal"], None, "partial")
        far_coords = _node_coords[p["far_nid"]]
        result["gpx_exit_wgs84"] = (far_coords[0], far_coords[1])
        result["remaining_m"] = round(p["remaining_m"])
        result["direction"] = "forward"
        logger.info("route_via_gpx: PARTIAL forward, %.2fkm GPX, "
                    "%.0fm restant, utilite=%.0f%%, via %s",
                    result["distance_km"], p["remaining_m"],
                    p["usefulness"] * 100, result["gpx_sources"])
        return result

    if best_reverse is not None:
        # reverse partial (portail cote destination)
        p = best_reverse
        result = _build_gpx_result(
            p["path"], None, p["portal"], "partial")
        entry_coords = _node_coords[p["near_nid"]]
        result["gpx_entry_wgs84"] = (entry_coords[0], entry_coords[1])
        result["remaining_m"] = round(p["remaining_m"])
        result["direction"] = "reverse"
        logger.info("route_via_gpx: PARTIAL reverse, %.2fkm GPX, "
                    "%.0fm restant, utilite=%.0f%%, via %s",
                    result["distance_km"], p["remaining_m"],
                    p["usefulness"] * 100, result["gpx_sources"])
        return result

    logger.info("route_via_gpx: aucun trajet GPX utile trouve")
    return None


def _build_gpx_result(path_nodes, entry_portal, exit_portal, coverage):
    """Construit le dict resultat depuis un chemin dans le graphe."""
    coords = []
    dist_total = 0.0
    dplus = 0.0
    dminus = 0.0
    gpx_sources = set()

    for i, nid in enumerate(path_nodes):
        c = _node_coords[nid]
        coords.append(c)
        if i > 0:
            prev = _node_coords[path_nodes[i - 1]]
            dist_total += _haversine_m(prev[0], prev[1], c[0], c[1])
            dz = c[2] - prev[2]
            if dz > 0:
                dplus += dz
            else:
                dminus += abs(dz)
            edge_data = _gpx_graph.get_edge_data(path_nodes[i - 1], nid)
            if edge_data and "gpx_source" in edge_data:
                gpx_sources.add(edge_data["gpx_source"])

    return {
        "gpx_coords": coords,
        "entry_portal": entry_portal,
        "exit_portal": exit_portal,
        "coverage": coverage,
        "distance_km": round(dist_total / 1000, 2),
        "dplus_m": round(dplus),
        "dminus_m": round(dminus),
        "gpx_sources": sorted(gpx_sources),
    }


# ---- rebuild / cache management ----

def rebuild_gpx_graph(entries=None):
    """Reconstruit le graphe GPX. Reset les portails (lazy)."""
    global _gpx_graph, _node_coords, _gpx_portals, _portals_ready

    if entries is None:
        entries = load_index()

    graph, nc = build_gpx_graph(entries)

    # assignation atomique
    _gpx_graph = graph
    _node_coords = nc
    _gpx_portals = None
    _portals_ready = False

    n_comp = nx.number_connected_components(graph) if len(graph) > 0 else 0

    # log les traces chargees
    trace_files = set()
    for _, _, d in graph.edges(data=True):
        trace_files.add(d.get("gpx_source", "?"))
    logger.info("gpx graph rebuilt: %d nodes, %d edges, %d composantes, "
                "traces: %s", graph.number_of_nodes(), graph.number_of_edges(),
                n_comp, sorted(trace_files))

    info = {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_components": n_comp,
        "traces": sorted(trace_files),
    }
    return info


# ---- conversion GeoJSON ----

def gpx_to_geojson_feature(gpx_result, route_index=0):
    """Convertit le resultat route_via_gpx() en GeoJSON Feature."""
    coords = [
        [round(lon, 6), round(lat, 6), round(alt, 1)]
        for lat, lon, alt in gpx_result["gpx_coords"]
    ]

    time_h = _estimate_tobler_time(gpx_result["gpx_coords"])

    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {
            "route_index": route_index,
            "is_optimal": route_index == 0,
            "distance_km": gpx_result["distance_km"],
            "dplus_m": gpx_result["dplus_m"],
            "dminus_m": gpx_result["dminus_m"],
            "time_tobler_h": round(time_h, 1),
            "glacier_pct": 0,
            "cost_total": 0,
            "n_points": len(coords),
            "strategy": "gpx_graph",
            "gpx_sources": gpx_result["gpx_sources"],
        },
    }


def _estimate_tobler_time(coords_list):
    """Estimation Tobler rapide depuis les coords (lat,lon,alt)."""
    total_s = 0.0
    for i in range(1, len(coords_list)):
        lat1, lon1, alt1 = coords_list[i - 1]
        lat2, lon2, alt2 = coords_list[i]
        d = _haversine_m(lat1, lon1, lat2, lon2)
        if d < 0.1:
            continue
        slope = (alt2 - alt1) / d
        # Tobler (1993)
        speed_ms = 1.667 * math.exp(-3.5 * abs(slope + 0.05))
        if speed_ms < 0.01:
            speed_ms = 0.01
        total_s += d / speed_ms
    return total_s / 3600


# ---- debug info ----

def get_gpx_cache_info():
    """Info debug: etat du cache."""
    if _gpx_graph is None:
        return {"n_nodes": 0, "n_edges": 0, "n_portals": 0, "n_components": 0}
    n_comp = nx.number_connected_components(_gpx_graph) if len(_gpx_graph) > 0 else 0
    return {
        "n_nodes": _gpx_graph.number_of_nodes(),
        "n_edges": _gpx_graph.number_of_edges(),
        "n_portals": len(_gpx_portals) if _gpx_portals else 0,
        "n_components": n_comp,
    }
