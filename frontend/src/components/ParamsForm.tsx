// formulaire parametres route
import { useApp } from '../context'

const MONTHS = [
  'Janvier', 'Fevrier', 'Mars', 'Avril', 'Mai', 'Juin',
  'Juillet', 'Aout', 'Septembre', 'Octobre', 'Novembre', 'Decembre',
]

const RESOLUTIONS = [0.5, 1.0, 2.0, 5.0, 10.0]

function formatCoord(lng: number, lat: number): string {
  return `${lat.toFixed(5)}, ${lng.toFixed(5)}`
}

export default function ParamsForm() {
  const { state, dispatch } = useApp()

  const reset = () => dispatch({ type: 'RESET' })

  return (
    <div className="flex flex-col gap-3 text-sm">
      {/* coords */}
      <div>
        <label className="text-gray-400 text-xs">Départ</label>
        <div className="text-white">
          {state.startPoint
            ? formatCoord(state.startPoint.lng, state.startPoint.lat)
            : 'Cliquer sur la carte'}
        </div>
      </div>
      <div>
        <label className="text-gray-400 text-xs">Arrivée</label>
        <div className="text-white">
          {state.endPoint
            ? formatCoord(state.endPoint.lng, state.endPoint.lat)
            : 'Cliquer sur la carte'}
        </div>
      </div>

      {/* resolution */}
      <div>
        <label className="text-gray-400 text-xs">Resolution Lidar MNT (m)</label>
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
      <div>
        <label className="text-gray-400 text-xs">Mois</label>
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
      <div>
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

      {/* mode précis (anisotrope) */}
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
          <span>Mode precis</span>
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
        <span>Acclimatation altitude</span>
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
