// appels API backend
import type {
  RouteResult, RouteParams, SSEMessage, MarkerPoint,
  RouteSummary, RouteDetail, UserZone, ZoneCreatePayload,
} from './types'

function buildBody(start: MarkerPoint, end: MarkerPoint, params: RouteParams) {
  return {
    start_lat: start.lat,
    start_lon: start.lng,
    end_lat: end.lat,
    end_lon: end.lng,
    resolution: params.resolution,
    month: params.month,
    acclimatized: params.acclimatized,
    n_alternatives: params.n_alternatives,
    anisotropic: params.anisotropic,
  }
}

// lance le calcul async
export async function calculateRouteAsync(
  start: MarkerPoint, end: MarkerPoint, params: RouteParams,
): Promise<string> {
  const resp = await fetch('/api/calculate-async', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(buildBody(start, end, params)),
  })
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }))
    throw new Error(err.detail || `Erreur ${resp.status}`)
  }
  const data = await resp.json()
  return data.job_id
}

// SSE progress
export function subscribeProgress(
  jobId: string,
  onMessage: (msg: SSEMessage) => void,
  onError: (err: string) => void,
): () => void {
  const es = new EventSource(`/api/progress/${jobId}`)

  es.onmessage = (ev) => {
    try {
      const msg: SSEMessage = JSON.parse(ev.data)
      onMessage(msg)
      // fermer quand termine ou erreur
      if (msg.status === 'completed' || msg.status === 'error') {
        es.close()
      }
    } catch {
      console.warn('SSE parse error', ev.data)
    }
  }

  // event: error (job not found)
  es.addEventListener('error', (ev) => {
    const raw = (ev as MessageEvent).data as string | undefined
    if (raw) {
      try {
        const parsed = JSON.parse(raw)
        onError(parsed.error || 'Erreur SSE')
      } catch {
        onError('Connexion SSE perdue')
      }
    } else {
      onError('Connexion SSE perdue')
    }
    es.close()
  })

  return () => es.close()
}

// fallback sync (pas de progress)
export async function calculateRoute(
  start: MarkerPoint, end: MarkerPoint, params: RouteParams,
): Promise<RouteResult> {
  const resp = await fetch('/api/calculate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(buildBody(start, end, params)),
  })
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }))
    throw new Error(err.detail || `Erreur ${resp.status}`)
  }
  return resp.json()
}


// =====================================================
//  Routes CRUD
// =====================================================

export async function fetchRouteHistory(
  limit = 50, offset = 0,
): Promise<RouteSummary[]> {
  const resp = await fetch(`/api/routes?limit=${limit}&offset=${offset}`)
  if (!resp.ok) throw new Error(`Erreur ${resp.status}`)
  const data = await resp.json()
  return data.routes
}

export async function fetchRouteDetail(id: number): Promise<RouteDetail> {
  const resp = await fetch(`/api/routes/${id}`)
  if (!resp.ok) throw new Error(`Erreur ${resp.status}`)
  return resp.json()
}

export async function deleteRoute(id: number): Promise<void> {
  const resp = await fetch(`/api/routes/${id}`, { method: 'DELETE' })
  if (!resp.ok) throw new Error(`Erreur ${resp.status}`)
}


// =====================================================
//  Zones CRUD
// =====================================================

export async function fetchZones(): Promise<UserZone[]> {
  const resp = await fetch('/api/zones')
  if (!resp.ok) throw new Error(`Erreur ${resp.status}`)
  const data = await resp.json()
  return data.zones
}

export async function createZone(payload: ZoneCreatePayload): Promise<{ id: number }> {
  const resp = await fetch('/api/zones', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!resp.ok) throw new Error(`Erreur ${resp.status}`)
  return resp.json()
}

export async function deleteZone(id: number): Promise<void> {
  const resp = await fetch(`/api/zones/${id}`, { method: 'DELETE' })
  if (!resp.ok) throw new Error(`Erreur ${resp.status}`)
}


// =====================================================
//  Alpine routes + segments
// =====================================================

export async function fetchAlpineRoutesGeoJSON(
  massif?: string,
): Promise<GeoJSON.FeatureCollection> {
  const params = massif ? `?massif=${encodeURIComponent(massif)}` : ''
  const resp = await fetch(`/api/alpine-routes/geojson${params}`)
  if (!resp.ok) throw new Error(`Erreur ${resp.status}`)
  return resp.json()
}

export async function fetchSegmentsGeoJSON(
  segmentType?: string,
): Promise<GeoJSON.FeatureCollection> {
  const params = segmentType ? `?segment_type=${encodeURIComponent(segmentType)}` : ''
  const resp = await fetch(`/api/terrain-segments/geojson${params}`)
  if (!resp.ok) throw new Error(`Erreur ${resp.status}`)
  return resp.json()
}


// =====================================================
//  Overlays (glaciers, cost surface)
// =====================================================

export async function fetchGlaciers(bbox: string): Promise<GeoJSON.FeatureCollection> {
  const resp = await fetch(`/api/glaciers?bbox=${bbox}`)
  if (!resp.ok) throw new Error(`Erreur ${resp.status}`)
  return resp.json()
}

export interface CostSurfaceData {
  imageUrl: string
  bounds: [[number, number], [number, number]]  // [[sw_lng, sw_lat], [ne_lng, ne_lat]]
}

export async function fetchCostSurface(): Promise<CostSurfaceData> {
  const resp = await fetch('/api/cost-surface')
  if (!resp.ok) throw new Error(`Erreur ${resp.status}`)

  const south = parseFloat(resp.headers.get('X-Bounds-South') || '0')
  const north = parseFloat(resp.headers.get('X-Bounds-North') || '0')
  const west = parseFloat(resp.headers.get('X-Bounds-West') || '0')
  const east = parseFloat(resp.headers.get('X-Bounds-East') || '0')

  const blob = await resp.blob()
  const imageUrl = URL.createObjectURL(blob)

  return {
    imageUrl,
    bounds: [[west, south], [east, north]],
  }
}
