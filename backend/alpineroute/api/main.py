# API FastAPI -- v2.0 pipeline adaptatif + CRUD routes/zones + GPX
# plus de chargement au startup, chaque requete lance le pipeline complet

import os
import json
import uuid
import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import gpxpy
import gpxpy.gpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, FileResponse

from alpineroute.config import DEM_CACHE_DIR, COST_CACHE_DIR, DB_PATH, CORS_ORIGINS, CRS_WGS84, NODATA_VALUE, GPX_DIR
from alpineroute.api.models import RouteRequest, ZoneCreate, ZoneUpdate
from alpineroute import pipeline as _pipeline
from alpineroute.pipeline import run_pipeline
from alpineroute.utils import PointOutOfBoundsError, DownloadError
from alpineroute.cost.glacier import find_rgi_shapefiles, load_and_clip_rgi
from alpineroute.db.schema import init_db
from alpineroute.db.crud import (
    get_route, list_routes, delete_route, save_route,
    save_zone, list_zones, get_zone, update_zone, delete_zone,
    list_alpine_routes, get_alpine_route, list_summits, get_alpine_routes_geojson,
    list_terrain_segments, get_terrain_segments_geojson,
)
from alpineroute.alpine.index import reload_index

logger = logging.getLogger(__name__)

# stockage des jobs SSE (thread-safe via dict python)
jobs: dict = {}


# --- App ---

@asynccontextmanager
async def lifespan(app):
    # config logging applicatif (uvicorn ne configure que ses propres loggers)
    log_level = os.environ.get("ALPINEROUTE_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO),
                        format="%(levelname)s %(name)s: %(message)s")

    os.makedirs(DEM_CACHE_DIR, exist_ok=True)
    os.makedirs(COST_CACHE_DIR, exist_ok=True)
    init_db()
    # sync traces GPX au demarrage
    try:
        result = reload_index()
        logger.info("alpine index: %s", result)
    except Exception as e:
        logger.warning("alpine index load: %s", e)
    logger.info("AlpineRoute API v2.0 ready")
    yield
    jobs.clear()


app = FastAPI(
    title="AlpineRoute Optimizer",
    version="2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=2)


# ---- Calculate ----

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/calculate")
async def calculate(req: RouteRequest):
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(executor, run_pipeline, req)
    except PointOutOfBoundsError as e:
        raise HTTPException(400, str(e))
    except DownloadError as e:
        raise HTTPException(502, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    return result


# -- SSE async --

@app.post("/calculate-async")
async def calculate_async(req: RouteRequest):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "running", "progress": 0,
        "step": "init", "message": "Demarrage...",
    }

    def _progress_cb(pct, step):
        jobs[job_id].update(progress=pct, step=step,
                            message=f"{step} ({pct}%)")

    def _worker():
        try:
            result = run_pipeline(req, progress_callback=_progress_cb)
            jobs[job_id].update(
                progress=100, step="done", message="Termine",
                status="completed", result=result,
            )
        except Exception as e:
            logger.exception("pipeline error job %s", job_id)
            jobs[job_id].update(status="error", message=str(e))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return {"job_id": job_id}


@app.get("/progress/{job_id}")
async def progress_sse(job_id: str):
    async def gen():
        last_pct = -1
        last_status = ""
        while True:
            job = jobs.get(job_id)
            if job is None:
                yield f"event: error\ndata: {json.dumps({'error': 'job not found'})}\n\n"
                break
            pct = job["progress"]
            status = job["status"]
            # envoyer si progress OU status a change
            if pct != last_pct or status != last_status:
                payload = {
                    "progress": pct, "step": job["step"],
                    "message": job["message"], "status": status,
                }
                if status == "completed" and "result" in job:
                    payload["result"] = job["result"]
                yield f"data: {json.dumps(payload)}\n\n"
                last_pct = pct
                last_status = status
                if status in ("completed", "error"):
                    break
            await asyncio.sleep(0.2)

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Routes CRUD

@app.get("/routes")
async def api_list_routes(
    lon_min: Optional[float] = Query(None),
    lat_min: Optional[float] = Query(None),
    lon_max: Optional[float] = Query(None),
    lat_max: Optional[float] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    min_distance_m: Optional[float] = Query(None),
    max_distance_m: Optional[float] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    bbox = None
    if all(v is not None for v in [lon_min, lat_min, lon_max, lat_max]):
        bbox = {"lon_min": lon_min, "lat_min": lat_min,
                "lon_max": lon_max, "lat_max": lat_max}

    rows = list_routes(DB_PATH, bbox=bbox, date_from=date_from,
                       date_to=date_to, min_distance=min_distance_m,
                       max_distance=max_distance_m,
                       limit=limit, offset=offset)
    return {"routes": rows, "count": len(rows)}


@app.get("/routes/{route_id}")
async def api_get_route(route_id: int):
    row = get_route(DB_PATH, route_id)
    if row is None:
        raise HTTPException(404, f"route {route_id} not found")
    # parse le geojson stocke
    if row.get("geojson") and isinstance(row["geojson"], str):
        try:
            row["geojson"] = json.loads(row["geojson"])
        except json.JSONDecodeError:
            pass
    return row


@app.delete("/routes/{route_id}")
async def api_delete_route(route_id: int):
    ok = delete_route(DB_PATH, route_id)
    if not ok:
        raise HTTPException(404, f"route {route_id} not found")
    return {"status": "deleted", "id": route_id}


@app.get("/routes/{route_id}/gpx")
async def api_route_gpx(route_id: int):
    row = get_route(DB_PATH, route_id)
    if row is None:
        raise HTTPException(404, f"route {route_id} not found")

    geojson_str = row.get("geojson")
    if not geojson_str:
        raise HTTPException(404, "pas de geojson pour cette route")

    if isinstance(geojson_str, str):
        geojson_data = json.loads(geojson_str)
    else:
        geojson_data = geojson_str

    # extraire les coordonnees du Feature
    coords = geojson_data.get("geometry", {}).get("coordinates", [])
    if not coords:
        raise HTTPException(404, "geojson sans coordonnees")

    # construire le GPX en memoire
    gpx = gpxpy.gpx.GPX()
    route_name = row.get("name") or f"Route #{route_id}"
    gpx.name = route_name
    gpx.creator = "AlpineRoute Optimizer"

    dist_km = (row.get("distance_m") or 0) / 1000
    gpx.description = (
        f"Route {dist_km:.1f}km D+{row.get('dplus_m', 0):.0f}m "
        f"~{row.get('time_tobler_h', 0):.1f}h"
    )

    track = gpxpy.gpx.GPXTrack()
    track.name = route_name
    gpx.tracks.append(track)

    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)

    for pt in coords:
        lon, lat = pt[0], pt[1]
        elev = pt[2] if len(pt) > 2 else None
        segment.points.append(
            gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon, elevation=elev)
        )

    xml = gpx.to_xml()
    filename = f"route_{route_id}.gpx"

    return Response(
        content=xml,
        media_type="application/gpx+xml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# --- Calques (overlays) ---

@app.get("/glaciers")
async def api_glaciers(bbox: str = Query(..., description="xmin,ymin,xmax,ymax en WGS84")):
    """Retourne les glaciers RGI dans la bbox en GeoJSON."""
    try:
        parts = [float(x) for x in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError
        lon_min, lat_min, lon_max, lat_max = parts
    except ValueError:
        raise HTTPException(400, "bbox invalide, format: xmin,ymin,xmax,ymax")

    try:
        import geopandas as gpd
        from pyproj import Transformer
        from alpineroute.config import CRS_L93

        # reprojeter bbox WGS84 -> L93
        transformer = Transformer.from_crs("EPSG:4326", CRS_L93, always_xy=True)
        x1, y1 = transformer.transform(lon_min, lat_min)
        x2, y2 = transformer.transform(lon_max, lat_max)
        bbox_l93 = {
            "xmin": min(x1, x2), "ymin": min(y1, y2),
            "xmax": max(x1, x2), "ymax": max(y1, y2),
        }

        shp_files = find_rgi_shapefiles()
        if not shp_files:
            return {"type": "FeatureCollection", "features": []}

        gdf = load_and_clip_rgi(shp_files[0], bbox_l93)
        if len(gdf) == 0:
            return {"type": "FeatureCollection", "features": []}

        # reprojeter en WGS84 + simplifier pour le front
        gdf = gdf.to_crs(CRS_WGS84)
        gdf["geometry"] = gdf.geometry.simplify(0.0005)  # ~50m

        # garder quelques colonnes utiles
        keep = [c for c in ["glims_id", "glac_name", "area_km2", "geometry"]
                if c in gdf.columns]
        geojson = json.loads(gdf[keep].to_json())
        return geojson

    except HTTPException:
        raise
    except Exception as e:
        logger.warning("glaciers endpoint error: %s", e)
        raise HTTPException(500, f"erreur glaciers: {e}")


@app.get("/cost-surface")
async def api_cost_surface():
    """Retourne la surface de cout du dernier calcul en PNG + bounds."""
    if not _pipeline._last_cost_surface:
        raise HTTPException(404, "aucun calcul en cours, lancer /calculate d'abord")

    try:
        import io
        from PIL import Image

        cost = _pipeline._last_cost_surface["cost"]
        bbox = _pipeline._last_cost_surface["bbox_wgs84"]

        # masquer nodata, log-scale, normaliser 0-255
        valid = np.isfinite(cost) & (cost > 0) & (cost != NODATA_VALUE)
        if not np.any(valid):
            raise HTTPException(404, "surface de cout vide")

        # separer pixels infranchissables (>= 1e6) du terrain normal
        impassable = valid & (cost >= 1e6)
        traversable = valid & (cost < 1e6)

        arr = np.where(traversable, cost, np.nan)
        log_cost = np.where(traversable, np.log1p(arr), np.nan)

        vmin = np.nanpercentile(log_cost, 2)
        vmax = np.nanpercentile(log_cost, 98)
        if vmax - vmin < 0.01:
            vmax = vmin + 1

        norm = np.clip((log_cost - vmin) / (vmax - vmin), 0, 1)

        # colormap vert -> jaune -> rouge (RGBA)
        H, W = norm.shape
        rgba = np.zeros((H, W, 4), dtype=np.uint8)

        # vert (0) -> jaune (0.5) -> rouge (1) pour le terrain traversable
        r = np.where(norm < 0.5, norm * 2 * 255, 255).astype(np.uint8)
        g = np.where(norm < 0.5, 255, (1 - (norm - 0.5) * 2) * 255).astype(np.uint8)
        b = np.zeros_like(r)
        a = np.where(traversable, 180, 0).astype(np.uint8)

        # pixels infranchissables en noir semi-transparent
        r[impassable] = 30
        g[impassable] = 30
        b[impassable] = 30
        a[impassable] = 200

        rgba[:, :, 0] = r
        rgba[:, :, 1] = g
        rgba[:, :, 2] = b
        rgba[:, :, 3] = a

        # sous-echantillonner si trop grand
        max_dim = 500
        if H > max_dim or W > max_dim:
            scale = max_dim / max(H, W)
            new_h, new_w = int(H * scale), int(W * scale)
            img = Image.fromarray(rgba, 'RGBA')
            img = img.resize((new_w, new_h), Image.NEAREST)
        else:
            img = Image.fromarray(rgba, 'RGBA')

        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)

        # bounds en WGS84
        bounds = {
            "south": bbox["lat_min"], "north": bbox["lat_max"],
            "west": bbox["lon_min"], "east": bbox["lon_max"],
        }

        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={
                "X-Bounds-South": str(bounds["south"]),
                "X-Bounds-North": str(bounds["north"]),
                "X-Bounds-West": str(bounds["west"]),
                "X-Bounds-East": str(bounds["east"]),
                "Cache-Control": "no-cache",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("cost-surface error")
        raise HTTPException(500, f"erreur cost surface: {e}")


# Zones CRUD

@app.get("/zones")
async def api_list_zones(
    zone_type: Optional[str] = Query(None),
    active_only: bool = Query(False),
):
    rows = list_zones(DB_PATH, zone_type=zone_type, active_only=active_only)
    return {"zones": rows, "count": len(rows)}


@app.post("/zones", status_code=201)
async def api_create_zone(body: ZoneCreate):
    zone_data = body.model_dump()
    zone_id = save_zone(DB_PATH, zone_data)
    return {"status": "created", "id": zone_id}


@app.get("/zones/{zone_id}")
async def api_get_zone(zone_id: int):
    row = get_zone(DB_PATH, zone_id)
    if row is None:
        raise HTTPException(404, f"zone {zone_id} not found")
    return row


@app.put("/zones/{zone_id}")
async def api_update_zone(zone_id: int, body: ZoneUpdate):
    updates = body.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(400, "rien a mettre a jour")
    ok = update_zone(DB_PATH, zone_id, updates)
    if not ok:
        raise HTTPException(404, f"zone {zone_id} not found")
    return {"status": "updated", "id": zone_id}


@app.delete("/zones/{zone_id}")
async def api_delete_zone(zone_id: int):
    ok = delete_zone(DB_PATH, zone_id)
    if not ok:
        raise HTTPException(404, f"zone {zone_id} not found")
    return {"status": "deleted", "id": zone_id}


# --- Terrain segments ---

@app.get("/terrain-segments")
async def api_list_terrain_segments(
    segment_type: Optional[str] = Query(None),
):
    rows = list_terrain_segments(DB_PATH, segment_type=segment_type)
    return {"segments": rows, "count": len(rows)}


@app.get("/terrain-segments/geojson")
async def api_terrain_segments_geojson(
    segment_type: Optional[str] = Query(None),
):
    fc = get_terrain_segments_geojson(DB_PATH, segment_type=segment_type)
    return fc


# --- Alpine routes (traces GPX indexees) ---

@app.get("/alpine-routes")
async def api_list_alpine_routes(
    massif: Optional[str] = Query(None),
    summit: Optional[str] = Query(None),
):
    rows = list_alpine_routes(DB_PATH, massif=massif, summit=summit)
    return {"routes": rows, "count": len(rows)}


# routes fixes AVANT la parametrique /{route_id}
@app.get("/alpine-routes/summits")
async def api_list_summits():
    rows = list_summits(DB_PATH)
    return {"summits": rows, "count": len(rows)}


@app.get("/alpine-routes/geojson")
async def api_alpine_routes_geojson(
    massif: Optional[str] = Query(None),
    summit: Optional[str] = Query(None),
):
    fc = get_alpine_routes_geojson(DB_PATH, massif=massif, summit=summit)
    return fc


@app.get("/alpine-routes/{route_id}")
async def api_get_alpine_route(route_id: int):
    row = get_alpine_route(DB_PATH, route_id)
    if row is None:
        raise HTTPException(404, f"alpine route {route_id} not found")
    return row


@app.get("/alpine-routes/{route_id}/gpx")
async def api_alpine_route_gpx(route_id: int):
    row = get_alpine_route(DB_PATH, route_id)
    if row is None:
        raise HTTPException(404, f"alpine route {route_id} not found")
    gpx_path = os.path.join(GPX_DIR, row["gpx_path"])
    if not os.path.isfile(gpx_path):
        raise HTTPException(404, f"fichier gpx introuvable: {row['gpx_path']}")
    return FileResponse(
        gpx_path,
        media_type="application/gpx+xml",
        filename=os.path.basename(gpx_path),
    )


@app.post("/admin/reload-index")
async def api_reload_index():
    try:
        result = reload_index()
        return {"status": "ok", **result}
    except Exception as e:
        logger.exception("reload-index error")
        raise HTTPException(500, f"reload error: {e}")


# --- Cache admin ---

@app.post("/admin/invalidate-cache")
async def api_invalidate_cache(bbox: Optional[dict] = None):
    from alpineroute.cost.cache import invalidate_cache
    bbox_l93 = None
    if bbox:
        bbox_l93 = {
            "xmin": bbox.get("xmin", 0), "ymin": bbox.get("ymin", 0),
            "xmax": bbox.get("xmax", 0), "ymax": bbox.get("ymax", 0),
        }
    n = invalidate_cache(bbox_l93)
    return {"status": "ok", "invalidated": n}


@app.get("/admin/cache-stats")
async def api_cache_stats():
    from alpineroute.cost.cache import cache_stats
    return cache_stats()
