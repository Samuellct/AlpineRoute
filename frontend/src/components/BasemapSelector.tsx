// basemap + overlays
import { useApp } from '../context'
import type { BasemapId, OverlayId } from '../types'

const basemaps: { id: BasemapId; label: string }[] = [
  { id: 'plan', label: 'Plan IGN' },
  { id: 'satellite', label: 'Satellite IGN' },
  { id: 'topo-global', label: 'Topo' },
  { id: 'satellite-global', label: 'Satellite' },
]

const overlays: { id: OverlayId; label: string }[] = [
  { id: 'slopes', label: 'Pentes' },
  { id: 'glaciers', label: 'Glaciers' },
  { id: 'cost', label: 'Cout' },
]

export default function BasemapSelector() {
  const { state, dispatch } = useApp()

  return (
    <div className="absolute bottom-6 right-3 flex flex-col gap-2 z-10">
      {/* calques */}
      <div className="flex flex-col gap-0.5">
        <span className="text-[10px] text-white/60 uppercase tracking-wider px-1">Calques</span>
        {overlays.map(o => {
          const active = state.activeOverlays.includes(o.id)
          return (
            <button
              key={o.id}
              onClick={() => dispatch({ type: 'TOGGLE_OVERLAY', overlay: o.id })}
              className={`px-3 py-1 text-xs rounded shadow cursor-pointer
                ${active
                  ? 'bg-blue-600 text-white font-semibold'
                  : 'bg-gray-800/80 text-white/70 hover:bg-gray-700/90'
                }`}
            >
              {o.label}
            </button>
          )
        })}
      </div>

      {/* separator */}
      <div className="border-t border-white/20" />

      {/* basemaps (exclusif) */}
      <div className="flex flex-col gap-0.5">
        <span className="text-[10px] text-white/60 uppercase tracking-wider px-1">Fond</span>
        {basemaps.map(b => (
          <button
            key={b.id}
            onClick={() => dispatch({ type: 'SET_BASEMAP', basemap: b.id })}
            className={`px-3 py-1.5 text-xs rounded shadow cursor-pointer
              ${state.basemap === b.id
                ? 'bg-white text-gray-900 font-semibold'
                : 'bg-gray-800/80 text-white hover:bg-gray-700/90'
              }`}
          >
            {b.label}
          </button>
        ))}
      </div>
    </div>
  )
}
