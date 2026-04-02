import { useState } from 'react'
import { useApp } from '../context'
import { useNominatim } from '../hooks/useNominatim'
import type { MarkerPoint } from '../types'

const MONTHS = [
  'Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin',
  'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre',
]

const RESOLUTIONS = [0.5, 1.0, 2.0, 5.0, 10.0]

// champ de recherche Nominatim + affichage coords
function SearchField({ label, point, onSelect }: {
  label: string
  point: MarkerPoint | null
  onSelect: (pt: MarkerPoint) => void
}) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const { results, loading } = useNominatim(query, open)

  // coords formatees si point defini par clic carte
  const displayValue = point && !open
    ? `${point.lat.toFixed(5)}, ${point.lng.toFixed(5)}`
    : query

  return (
    <div className="relative">
      <label className="text-gray-400 text-xs">{label}</label>
      <input
        value={displayValue}
        onChange={e => { setQuery(e.target.value); setOpen(true) }}
        onFocus={() => { if (!point) setOpen(true) }}
        onBlur={() => setTimeout(() => setOpen(false), 200)}
        placeholder="Rechercher un lieu..."
        className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white text-sm"
      />
      {open && results.length > 0 && (
        <div className="absolute z-30 w-full bg-gray-800 border border-gray-700 rounded mt-1 max-h-48 overflow-y-auto shadow-lg">
          {results.map((r, i) => (
            <div key={i}
              onMouseDown={() => {
                onSelect({ lat: parseFloat(r.lat), lng: parseFloat(r.lon) })
                setQuery(r.display_name.split(',')[0])
                setOpen(false)
              }}
              className="px-2 py-1.5 text-sm text-white hover:bg-gray-700 cursor-pointer truncate"
            >
              {r.display_name}
            </div>
          ))}
        </div>
      )}
      {loading && <div className="text-[10px] text-gray-500 mt-0.5">Recherche...</div>}
    </div>
  )
}

export default function ParamsForm() {
  const { state, dispatch } = useApp()

  const reset = () => dispatch({ type: 'RESET' })

  return (
    <div className="flex flex-col gap-3 text-sm">
      {/* coords / recherche */}
      <SearchField
        label="Départ"
        point={state.startPoint}
        onSelect={pt => dispatch({ type: 'SET_START_POINT', point: pt })}
      />
      <SearchField
        label="Arrivée"
        point={state.endPoint}
        onSelect={pt => dispatch({ type: 'SET_END_POINT', point: pt })}
      />

      {/* resolution */}
      <div title="Taille du pixel MNT en metres. Plus petit = plus precis mais plus lent.">
        <label className="text-gray-400 text-xs">Résolution Lidar MNT (m)</label>
        <select
          value={state.params.resolution}
          onChange={e => dispatch({
            type: 'SET_PARAMS', params: { resolution: parseFloat(e.target.value) },
          })}
          className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white"
        >
          {RESOLUTIONS.map(r => (
            <option key={r} value={r}>{r}m</option>
          ))}
        </select>
      </div>

      {/* mois */}
      <div title="Influence le calcul : ensoleillement, neige, conditions glaciaires.">
        <label className="text-gray-400 text-xs">Mois prévu pour la course</label>
        <select
          value={state.params.month}
          onChange={e => dispatch({
            type: 'SET_PARAMS', params: { month: parseInt(e.target.value) },
          })}
          className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white"
        >
          {MONTHS.map((m, i) => (
            <option key={i + 1} value={i + 1}>{m}</option>
          ))}
        </select>
      </div>

      {/* routes alternatives */}
      <div title="Nombre de variantes calculees par penalisation du chemin optimal.">
        <label className="text-gray-400 text-xs">Routes alternatives</label>
        <select
          value={state.params.n_alternatives}
          onChange={e => dispatch({
            type: 'SET_PARAMS', params: { n_alternatives: parseInt(e.target.value) },
          })}
          className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white"
        >
          {[0, 1, 2, 3, 4, 5].map(n => (
            <option key={n} value={n}>{n === 0 ? 'Aucune' : n}</option>
          ))}
        </select>
      </div>

      {/* mode precis (anisotrope) */}
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={state.params.anisotropic}
          onChange={e => dispatch({
            type: 'SET_PARAMS', params: { anisotropic: e.target.checked },
          })}
          className="accent-green-500"
        />
        <span className="flex flex-col">
          <span title="Prend en compte la direction de marche (montee/descente). Plus lent mais plus realiste.">Mode précis</span>
          <span className="text-[10px] text-gray-500">Dijkstra anisotrope</span>
        </span>
      </label>

      {/* acclimatation */}
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={state.params.acclimatized}
          onChange={e => dispatch({
            type: 'SET_PARAMS', params: { acclimatized: e.target.checked },
          })}
          className="accent-green-500"
        />
        <span title="Reduit la penalite d'altitude au-dessus de 2500m si vous etes acclimatise.">Acclimatation altitude</span>
      </label>

      {/* 3D toggle */}
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={state.is3D}
          onChange={() => dispatch({ type: 'TOGGLE_3D' })}
          className="accent-green-500"
        />
        <span>Relief 3D</span>
      </label>

      {/* reset */}
      <button
        onClick={reset}
        className="text-xs text-gray-400 hover:text-white underline cursor-pointer self-start"
      >
        Reset
      </button>
    </div>
  )
}
