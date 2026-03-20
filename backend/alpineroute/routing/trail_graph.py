# trail_graph.py -- graphe local depuis les sentiers OSM (Overpass)
# fallback quand Valhalla ne couvre pas la zone
# construit un NetworkX a partir du GeoDataFrame de trails.py

import math
import logging

import networkx as nx
from shapely.geometry import LineString

from alpineroute.config import (
    TRAIL_GRAPH_SUBSAMPLE_M, TRAIL_GRAPH_MERGE_M,
    TRAIL_GRAPH_MAX_SNAP_M,
)
from alpineroute.utils import l93_to_wgs84

logger = logging.getLogger(__name__)


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _subsample_line(coords_wgs84, min_dist_m):
    """Sous-echantillonne une liste de (lon, lat) WGS84."""
    if len(coords_wgs84) <= 2:
        return list(coords_wgs84)
    result = [coords_wgs84[0]]
    cumul = 0.0
    for i in range(1, len(coords_wgs84) - 1):
        lon1, lat1 = coords_wgs84[i - 1]
        lon2, lat2 = coords_wgs84[i]
        cumul += _haversine_m(lat1, lon1, lat2, lon2)
        if cumul >= min_dist_m:
            result.append(coords_wgs84[i])
            cumul = 0.0
    result.append(coords_wgs84[-1])
    return result


def _find_nearby_node(lat, lon, node_coords, tolerance_m):
    """Scan lineaire pour trouver un noeud existant dans le rayon."""
    best_id = None
    best_dist = tolerance_m
    for nid, (nlat, nlon) in node_coords.items():
        d = _haversine_m(lat, lon, nlat, nlon)
        if d < best_dist:
            best_dist = d
            best_id = nid
    return best_id


def build_trail_graph(trails_gdf):
    """Construit un graphe NetworkX depuis le GeoDataFrame de sentiers OSM.
    Le GDF est en L93 (CRS_L93) avec colonnes trail_class, trail_cost.
    Retourne (nx.Graph, node_coords_dict) avec node_coords = {id: (lat, lon)}."""
    G = nx.Graph()
    node_coords = {}  # nid -> (lat, lon)
    next_id = 0

    if trails_gdf is None or len(trails_gdf) == 0:
        return G, node_coords

    for _, row in trails_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if not isinstance(geom, LineString):
            continue

        trail_cost = row.get("trail_cost", 0.30)
        trail_class = row.get("trail_class", "trail_default")

        # coords L93 -> WGS84
        l93_pts = list(geom.coords)
        wgs_pts = []
        for x, y in l93_pts:
            lon, lat = l93_to_wgs84(x, y)
            wgs_pts.append((lon, lat))

        # sous-echantillonnage
        sub = _subsample_line(wgs_pts, TRAIL_GRAPH_SUBSAMPLE_M)
        if len(sub) < 2:
            continue

        # creer les noeuds
        trace_nodes = []
        for i, (lon, lat) in enumerate(sub):
            is_endpoint = (i == 0 or i == len(sub) - 1)
            merged = None
            if is_endpoint:
                merged = _find_nearby_node(lat, lon, node_coords, TRAIL_GRAPH_MERGE_M)

            if merged is not None:
                trace_nodes.append(merged)
            else:
                nid = next_id
                next_id += 1
                node_coords[nid] = (lat, lon)
                G.add_node(nid, lat=lat, lon=lon)
                trace_nodes.append(nid)

        # edges
        for i in range(len(trace_nodes) - 1):
            n1, n2 = trace_nodes[i], trace_nodes[i + 1]
            if n1 == n2:
                continue
            c1 = node_coords[n1]
            c2 = node_coords[n2]
            dist_m = _haversine_m(c1[0], c1[1], c2[0], c2[1])
            # poids = distance * trail_cost (plus le sentier est bon, moins il coute)
            weight = dist_m * trail_cost
            # garder le meilleur edge si doublon
            existing = G.get_edge_data(n1, n2)
            if existing is not None and existing["weight"] <= weight:
                continue
            G.add_edge(n1, n2, weight=weight, distance_m=dist_m,
                       trail_cost=trail_cost, trail_class=trail_class)

    n_comp = nx.number_connected_components(G) if len(G) > 0 else 0
    logger.info("trail graph: %d nodes, %d edges, %d composantes",
                G.number_of_nodes(), G.number_of_edges(), n_comp)
    return G, node_coords


def _snap_to_graph(lat, lon, node_coords, max_dist_m=None):
    """Snap un point WGS84 sur le graphe. Retourne (node_id, dist_m) ou (None, inf)."""
    if max_dist_m is None:
        max_dist_m = TRAIL_GRAPH_MAX_SNAP_M
    best_id = None
    best_dist = float("inf")
    for nid, (nlat, nlon) in node_coords.items():
        d = _haversine_m(lat, lon, nlat, nlon)
        if d < best_dist:
            best_dist = d
            best_id = nid
    if best_dist <= max_dist_m:
        return best_id, best_dist
    return None, float("inf")


def route_via_trail_graph(trails_gdf, start_wgs84, end_wgs84):
    """Route entre 2 points via le graphe local OSM.
    start/end_wgs84: (lat, lon).
    Retourne dict {coords, distance_km, trail_sources} ou None."""
    G, node_coords = build_trail_graph(trails_gdf)
    if G.number_of_nodes() < 2:
        logger.info("trail_graph: trop peu de noeuds (%d)", G.number_of_nodes())
        return None

    start_nid, start_snap = _snap_to_graph(
        start_wgs84[0], start_wgs84[1], node_coords)
    end_nid, end_snap = _snap_to_graph(
        end_wgs84[0], end_wgs84[1], node_coords)

    if start_nid is None:
        logger.info("trail_graph: start trop loin du graphe (%.0fm)", start_snap)
        return None
    if end_nid is None:
        logger.info("trail_graph: end trop loin du graphe (%.0fm)", end_snap)
        return None
    if start_nid == end_nid:
        logger.info("trail_graph: start == end (meme noeud)")
        return None

    if not nx.has_path(G, start_nid, end_nid):
        logger.info("trail_graph: pas de chemin entre start (node %d) et end (node %d)",
                     start_nid, end_nid)
        return None

    path = nx.shortest_path(G, start_nid, end_nid, weight="weight")

    # extraire coords + stats
    coords = []
    dist_total = 0.0
    trail_classes = set()
    for i, nid in enumerate(path):
        lat, lon = node_coords[nid]
        coords.append((lat, lon))
        if i > 0:
            edge = G.get_edge_data(path[i - 1], nid)
            if edge:
                dist_total += edge["distance_m"]
                trail_classes.add(edge.get("trail_class", "?"))

    logger.info("trail_graph: route OK, %.2fkm, %d pts, snap_start=%.0fm, "
                "snap_end=%.0fm, classes=%s",
                dist_total / 1000, len(coords), start_snap, end_snap,
                sorted(trail_classes))

    return {
        "coords": coords,  # [(lat, lon), ...]
        "distance_km": round(dist_total / 1000, 3),
        "snap_start_m": round(start_snap),
        "snap_end_m": round(end_snap),
        "trail_classes": sorted(trail_classes),
        "n_points": len(coords),
    }
