// profil altimetrique -- Recharts AreaChart, hover synchro carte
import { useMemo } from 'react'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer } from 'recharts'
import { useApp } from '../context'
import type { RouteFeature } from '../types'

// haversine simplifiee (m) entre deux points WGS84
function haversine(lon1: number, lat1: number, lon2: number, lat2: number): number {
  const R = 6371000
  const toRad = Math.PI / 180
  const dLat = (lat2 - lat1) * toRad
  const dLon = (lon2 - lon1) * toRad
  const a = Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1 * toRad) * Math.cos(lat2 * toRad) * Math.sin(dLon / 2) ** 2
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
}

interface DataPoint {
  dist: number   // km cumule
  alt: number
  index: number
}

function getSelectedRoute(
  routeResult: { route: RouteFeature; routes?: RouteFeature[] } | null,
  idx: number,
): RouteFeature | null {
  if (!routeResult) return null
  if (routeResult.routes && routeResult.routes[idx]) return routeResult.routes[idx]
  return routeResult.route
}

// chevron SVG inline (pas de dep externe)
function ChevronUp() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M4 10L8 6L12 10" />
    </svg>
  )
}

function ChevronDown() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M4 6L8 10L12 6" />
    </svg>
  )
}

export default function ElevationProfile() {
  const { state, dispatch } = useApp()

  const route = getSelectedRoute(state.routeResult, state.selectedRouteIndex)

  const data: DataPoint[] = useMemo(() => {
    if (!route) return []
    const coords = route.geometry.coordinates
    let cumDist = 0
    return coords.map((c, i) => {
      if (i > 0) {
        cumDist += haversine(coords[i - 1][0], coords[i - 1][1], c[0], c[1])
      }
      return { dist: Math.round(cumDist) / 1000, alt: c[2], index: i }
    })
  }, [route])

  if (!route || data.length === 0) return null

  // profil masque -> juste un petit bouton pour re-ouvrir
  if (!state.profileVisible) {
    return (
      <button
        onClick={() => dispatch({ type: 'TOGGLE_PROFILE' })}
        className="absolute bottom-2 left-1/2 -translate-x-1/2 z-10
          bg-gray-900/80 hover:bg-gray-800 text-white/80 hover:text-white
          rounded-t px-4 py-1 text-xs backdrop-blur-sm cursor-pointer
          border border-gray-700 border-b-0 transition-colors"
        title="Afficher le profil"
      >
        <ChevronUp />
      </button>
    )
  }

  // trouver la distance du hover venant de la carte
  const hoverDist = state.hoveredIndex != null && data[state.hoveredIndex]
    ? data[state.hoveredIndex].dist
    : null

  return (
    <div className="absolute bottom-0 left-0 right-0 h-48 z-10
      bg-gray-900/90 backdrop-blur-sm border-t border-gray-700">
      {/* bouton fermer */}
      <button
        onClick={() => dispatch({ type: 'TOGGLE_PROFILE' })}
        className="absolute top-1 right-2 z-20 text-white/50 hover:text-white
          cursor-pointer p-1 transition-colors"
        title="Masquer le profil"
      >
        <ChevronDown />
      </button>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={data}
          margin={{ top: 10, right: 20, left: 10, bottom: 5 }}
          onMouseMove={(e: any) => {
            if (e?.activePayload?.[0]) {
              const idx = e.activePayload[0].payload.index
              dispatch({ type: 'SET_HOVERED_INDEX', index: idx })
            }
          }}
          onMouseLeave={() => dispatch({ type: 'SET_HOVERED_INDEX', index: null })}
        >
          <defs>
            <linearGradient id="altGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#22c55e" stopOpacity={0.6} />
              <stop offset="95%" stopColor="#22c55e" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="dist"
            type="number"
            domain={['dataMin', 'dataMax']}
            tickFormatter={v => `${v.toFixed(1)}`}
            tick={{ fill: '#9ca3af', fontSize: 11 }}
            stroke="#4b5563"
            label={{ value: 'km', position: 'insideBottomRight', fill: '#9ca3af', fontSize: 11 }}
          />
          <YAxis
            domain={['auto', 'auto']}
            tick={{ fill: '#9ca3af', fontSize: 11 }}
            stroke="#4b5563"
            label={{ value: 'm', position: 'insideTopLeft', fill: '#9ca3af', fontSize: 11 }}
            width={50}
          />
          <Tooltip
            contentStyle={{
              background: 'rgba(17,24,39,0.95)', border: '1px solid #374151',
              borderRadius: 6, fontSize: 12, color: '#fff',
            }}
            formatter={(value: any) => [`${Math.round(value)} m`, 'Altitude']}
            labelFormatter={(v: any) => `${Number(v).toFixed(2)} km`}
          />
          {hoverDist != null && (
            <ReferenceLine x={hoverDist} stroke="#facc15" strokeDasharray="3 3" />
          )}
          <Area
            type="monotone"
            dataKey="alt"
            stroke="#22c55e"
            fill="url(#altGrad)"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 3, fill: '#facc15' }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
