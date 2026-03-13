import { useApp } from '../context'
import { getSelectedRoute } from '../types'

const STRATEGY_LABELS: Record<string, { label: string; color: string }> = {
  network: { label: 'Reseau OSM', color: 'bg-green-600' },
  hybrid: { label: 'Hybride', color: 'bg-blue-600' },
  raster: { label: 'Raster', color: 'bg-amber-600' },
  hybrid_bridge: { label: 'Pont raster', color: 'bg-blue-500' },
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

  const strategyKey = state.routeResult.strategy
  const strategyInfo = strategyKey ? STRATEGY_LABELS[strategyKey] : null

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

      {/* strategie */}
      {strategyInfo && (
        <div className="flex items-center gap-2 mb-2">
          <span className={`text-xs px-2 py-0.5 rounded ${strategyInfo.color} text-white`}>
            {strategyInfo.label}
          </span>
        </div>
      )}

      {/* layers */}
      {state.routeResult.layers_used && state.routeResult.layers_used.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-2">
          {state.routeResult.layers_used.map(l => (
            <span key={l} className="text-[10px] px-1.5 py-0.5 rounded bg-gray-700 text-gray-300">
              {l}
            </span>
          ))}
        </div>
      )}

      {/* valhalla dispo */}
      {state.routeResult.valhalla_available === false && (
        <div className="text-[10px] text-gray-500 mb-1">
          Valhalla indisponible
        </div>
      )}

      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        {stats.map(s => (
          <div key={s.label} className="flex justify-between">
            <span className="text-gray-400">{s.label}</span>
            <span className="font-medium">{s.value}</span>
          </div>
        ))}
      </div>
      {state.routeResult?.warnings?.map((w, i) => (
        <div key={i} className="bg-amber-50 border border-amber-300 text-amber-800 text-xs p-2 rounded mt-2">
          {w}
        </div>
      ))}
    </div>
  )
}
