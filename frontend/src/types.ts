// types partages frontend

export type Coord3D = [number, number, number] // [lon, lat, ele]

export interface RouteProperties {
  route_index: number
  is_optimal: boolean
  distance_km: number
  dplus_m: number
  dminus_m: number
  time_tobler_h: number
  glacier_pct: number
  cost_total: number
  n_points: number
  resolution_m: number
}

export interface RouteFeature {
  type: 'Feature'
  geometry: {
    type: 'LineString'
    coordinates: Coord3D[]
  }
  properties: RouteProperties
}

export interface RouteResult {
  status: string
  route: RouteFeature
  computation_time_s: number
  routes?: RouteFeature[]
  n_routes?: number
  saved_route_id?: number
  warnings?: string[]
}

export interface SSEMessage {
  progress: number
  step: string
  message: string
  status: string
  result?: RouteResult
}

export interface RouteParams {
  resolution: number
  month: number
  acclimatized: boolean
  n_alternatives: number
  anisotropic: boolean
}

export interface MarkerPoint {
  lng: number
  lat: number
}

export type BasemapId = 'plan' | 'satellite' | 'topo-global' | 'satellite-global'

export type OverlayId = 'slopes' | 'cost' | 'glaciers'

// -- historique --

export interface RouteSummary {
  id: number
  created_at: string
  name: string | null
  start_lat: number
  start_lon: number
  end_lat: number
  end_lon: number
  distance_m: number
  dplus_m: number
  dminus_m: number
  time_tobler_h: number
  glacier_pct: number
  cost_total: number | null
  computation_time_s: number
}

export interface RouteDetail extends RouteSummary {
  geojson: RouteFeature
}

// -- zones --

export type ZoneType = 'crevasse' | 'serac' | 'cornice' | 'rockfall' | 'forbidden' | 'custom'

export interface UserZone {
  id: number
  name: string
  zone_type: ZoneType
  cost_multiplier: number
  geojson: object
  active: boolean
}

export interface ZoneCreatePayload {
  name: string
  zone_type: ZoneType
  cost_multiplier: number
  geojson: object
  active: boolean
}

export type SidebarTab = 'calcul' | 'historique'

// helper route affichée
export function getSelectedRoute(
  routeResult: RouteResult | null,
  idx: number,
): RouteFeature | null {
  if (!routeResult) return null
  if (routeResult.routes && routeResult.routes[idx]) return routeResult.routes[idx]
  return routeResult.route
}
