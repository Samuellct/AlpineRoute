# T09 - FastAPI

import os
import sys
import time
import json
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from pyproj import Transformer
from skimage.graph import route_through_array
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import (
    DEM_DIR, DERIVED_DIR,
    DEM_RESOLUTION, NODATA_VALUE,
    CRS_L93, CRS_WGS84,
    API_HOST, API_PORT,
    BBOX_WGS84,
)


# =======================================================
#  Modeles Pydantic
# =======================================================

class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float


# =======================================================
#  Chargement
# =======================================================

def _load_all():
    """Charge DEM, cost surface, glacier mask en memoire."""
    data = {}

    # cost surface
    cost_path = os.path.join(DERIVED_DIR, f"cost_surface_{DEM_RESOLUTION}m.tif")
    if not os.path.exists(cost_path):
        print(f"[error] cost surface introuvable: {cost_path}")
        print("  -> lancer T04 d'abord")
        return None

    with rasterio.open(cost_path) as ds:
        cost = ds.read(1).astype(np.float64)
        transform = ds.transform
        data["crs"] = str(ds.crs)
        data["bounds"] = ds.bounds

    # prep nodata -> inf
    nodata_mask = (cost == NODATA_VALUE)
    cost[nodata_mask] = np.inf
    cost = np.clip(cost, 0, 1e6)
    cost[nodata_mask] = np.inf
    data["cost"] = cost
    data["transform"] = transform

    # DEM
    dem_path = os.path.join(DEM_DIR, f"dem_aiguille_du_midi_{DEM_RESOLUTION}m.tif")
    if not os.path.exists(dem_path):
        print(f"[error] DEM introuvable: {dem_path}")
        return None
    with rasterio.open(dem_path) as ds:
        data["dem"] = ds.read(1).astype(np.float32)

    # glacier mask
    glacier_path = os.path.join(DERIVED_DIR, f"glacier_mask_{DEM_RESOLUTION}m.tif")
    if os.path.exists(glacier_path):
        with rasterio.open(glacier_path) as ds:
            data["glacier_mask"] = ds.read(1).astype(bool)
    else:
        data["glacier_mask"] = None
        print("[warn] pas de masque glacier")

    print(f"[startup] charge: cost={cost.shape}, dem loaded, "
          f"glacier={'oui' if data['glacier_mask'] is not None else 'non'}")
    return data


# =======================================================
#  Lifespan
# =======================================================

@asynccontextmanager
async def lifespan(app):
    print("--- Chargement des donnees ---")
    t0 = time.time()
    data = _load_all()
    if data is None:
        print("[FATAL] impossible de charger les donnees")
        app.state.data = None
    else:
        app.state.data = data
        dt = time.time() - t0
        print(f"--- Pret en {dt:.1f}s ---")
    yield
    print("--- Shutdown ---")


# =======================================================
#  FastAPI
# =======================================================

app = FastAPI(title="AlpineRoute Optimizer", version="0.1-test", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=2)


# =======================================================
#  Helpers
# =======================================================

def wgs84_to_pixel(lat, lon, transform, shape):
    """WGS84 lat/lon -> row/col pixel."""
    proj = Transformer.from_crs(CRS_WGS84, CRS_L93, always_xy=True)
    x_l93, y_l93 = proj.transform(lon, lat)
    col, row = ~transform * (x_l93, y_l93)
    row, col = int(round(row)), int(round(col))
    h, w = shape
    if not (0 <= row < h and 0 <= col < w):
        return None
    return row, col, x_l93, y_l93


def pixel_to_l93(rows, cols, transform):
    """rows/cols -> coords L93."""
    coords = np.array([transform * (c, r) for r, c in zip(rows, cols)])
    return coords[:, 0], coords[:, 1]


def reproject_to_wgs84(x_l93, y_l93):
    """L93 -> WGS84 (lon, lat)."""
    proj = Transformer.from_crs(CRS_L93, CRS_WGS84, always_xy=True)
    lons, lats = proj.transform(x_l93, y_l93)
    return lons, lats


def compute_route(data, start_lat, start_lon, end_lat, end_lon):
    """Calcul complet: conversion coords + pathfinding + stats. Bloquant."""
    cost = data["cost"]
    dem = data["dem"]
    transform = data["transform"]
    glacier_mask = data["glacier_mask"]

    # conversion coords
    start_px = wgs84_to_pixel(start_lat, start_lon, transform, cost.shape)
    if start_px is None:
        raise ValueError(f"Point de depart hors grille: ({start_lat}, {start_lon})")
    end_px = wgs84_to_pixel(end_lat, end_lon, transform, cost.shape)
    if end_px is None:
        raise ValueError(f"Point d'arrivee hors grille: ({end_lat}, {end_lon})")

    start_rc = (start_px[0], start_px[1])
    end_rc = (end_px[0], end_px[1])

    # dijkstra
    t0 = time.time()
    path_coords, path_cost = route_through_array(
        cost, start=start_rc, end=end_rc,
        fully_connected=True, geometric=True,
    )
    dt_pathfind = time.time() - t0
    path_coords = np.array(path_coords)

    if len(path_coords) == 0:
        raise RuntimeError("Pathfinding a retourne un chemin vide")

    # extraction stats
    rows = path_coords[:, 0]
    cols = path_coords[:, 1]
    elevations = dem[rows, cols]

    x_l93, y_l93 = pixel_to_l93(rows, cols, transform)

    dx = np.diff(x_l93)
    dy = np.diff(y_l93)
    dz = np.diff(elevations)
    seg_dist = np.sqrt(dx**2 + dy**2)
    dist_2d = float(np.sum(seg_dist))

    dplus = float(np.sum(dz[dz > 0]))
    dminus = float(abs(np.sum(dz[dz < 0])))

    # temps Tobler
    gradient = np.where(seg_dist > 0, dz / seg_dist, 0)
    v_tobler = 6.0 * np.exp(-3.5 * np.abs(gradient + 0.05)) * 0.6
    v_tobler = np.maximum(v_tobler, 0.01)
    seg_time_h = (seg_dist / 1000.0) / v_tobler
    total_time_h = float(np.sum(seg_time_h))

    # glacier
    glacier_pct = 0.0
    if glacier_mask is not None:
        on_glacier = glacier_mask[rows, cols]
        glacier_pct = float(on_glacier.sum() / len(rows) * 100)

    # reprojection WGS84 pour le GeoJSON de sortie
    lons, lats = reproject_to_wgs84(x_l93, y_l93)

    # sous-echantillonnage si trop de points (> 5000)
    n = len(lons)
    if n > 5000:
        step = max(1, n // 5000)
        idx = np.arange(0, n, step)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)
        lons = lons[idx]
        lats = lats[idx]
        elevations = elevations[idx]

    coordinates = [
        [round(float(lo), 6), round(float(la), 6), round(float(el), 1)]
        for lo, la, el in zip(lons, lats, elevations)
    ]

    return {
        "status": "ok",
        "route": {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates,
            },
            "properties": {
                "distance_km": round(dist_2d / 1000, 2),
                "dplus_m": round(dplus),
                "dminus_m": round(dminus),
                "time_tobler_h": round(total_time_h, 1),
                "glacier_pct": round(glacier_pct, 1),
                "n_points": len(coordinates),
                "resolution_m": DEM_RESOLUTION,
            },
        },
        "computation_time_s": round(dt_pathfind, 1),
    }

# =======================================================
#  Endpoints
# =======================================================
@app.get("/health")
async def health():
    loaded = app.state.data is not None
    return {"status": "ok" if loaded else "error", "dem_loaded": loaded}


@app.get("/info")
async def info():
    data = app.state.data
    if data is None:
        raise HTTPException(500, "Donnees non chargees")

    bounds = data["bounds"]
    cost = data["cost"]

    # bbox approx
    return {
        "crs": data["crs"],
        "shape": list(cost.shape),
        "resolution_m": DEM_RESOLUTION,
        "bounds_l93": {
            "left": bounds.left,
            "bottom": bounds.bottom,
            "right": bounds.right,
            "top": bounds.top,
        },
        "bbox_wgs84": BBOX_WGS84,
        "glacier_mask": data["glacier_mask"] is not None,
    }


@app.post("/calculate")
async def calculate(req: RouteRequest):
    data = app.state.data
    if data is None:
        raise HTTPException(500, "Donnees non chargees")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            executor,
            compute_route,
            data,
            req.start_lat, req.start_lon,
            req.end_lat, req.end_lon,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    return result


# =======================================================

if __name__ == "__main__":
    import uvicorn
    print(f"Demarrage sur http://{API_HOST}:{API_PORT}")
    print("  POST /calculate  - calcul de route")
    print("  GET  /info       - metadonnees DEM")
    print("  GET  /health     - etat du serveur")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
