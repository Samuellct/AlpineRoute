import { useApp } from '../context'
import type { RouteFeature } from '../types'

// helper: la route actuellement selectionnee
function getSelectedRoute(
  routeResult: { route: RouteFeature; routes?: RouteFeature[] } | null,
  idx: number,
): RouteFeature | null {
  if (!routeResult) return null
  if (routeResult.routes && routeResult.routes[idx]) return routeResult.routes[idx]
  return routeResult.route
}

export default function RouteStats() {
  const { state } = useApp()
  if (!state.routeResult) return null

  const route = getSelectedRoute(state.routeResult, state.selectedRouteIndex)
  if (!route) return null

  const p = route.properties
  const ct = state.routeResult.computation_time_s
  const nRoutes = state.routeResult.routes?.length || 1

  const stats = [
    { label: 'Distance', value: `${p.distance_km} km` },
    { label: 'D+', value: `${p.dplus_m} m` },
    { label: 'D-', value: `${p.dminus_m} m` },
    { label: 'Temps Tobler', value: `${p.time_tobler_h} h` },
    { label: 'Glacier', value: `${p.glacier_pct}%` },
    { label: 'Resolution', value: `${p.resolution_m} m` },
    { label: 'Calcul', value: `${ct} s` },
  ]

  return (
    <div>
      <h3 className="text-xs text-gray-400 mb-2 uppercase tracking-wider">
        Resultats
        {nRoutes > 1 && (
          <span className="ml-2 normal-case text-gray-500">
            Route {state.selectedRouteIndex + 1}/{nRoutes}
            {state.selectedRouteIndex === 0 ? ' (optimale)' : ''}
          </span>
        )}
      </h3>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        {stats.map(s => (
          <div key={s.label} className="flex justify-between">
            <span className="text-gray-400">{s.label}</span>
            <span className="font-medium">{s.value}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
