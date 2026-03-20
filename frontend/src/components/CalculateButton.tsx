// bouton calcul + progress bar
import { useRef } from 'react'
import { useApp } from '../context'
import { calculateRouteAsync, subscribeProgress } from '../api'

// noms d'etapes lisibles
const STEP_LABELS: Record<string, string> = {
  init: 'Initialisation...',
  network: 'Reseau routier...',
  gpx_graph: 'Traces GPX...',
  bbox: 'Calcul emprise...',
  cache: 'Verification cache...',
  dem: 'Telechargement MNT...',
  terrain: 'Analyse terrain...',
  worldcover: 'Occupation du sol...',
  glacier: 'Detection glaciers...',
  radiation: 'Radiation solaire...',
  osm: 'Sentiers OSM...',
  cost: 'Surface de cout...',
  zones: 'Zones utilisateur...',
  pathfinding: 'Recherche de chemin...',
  result: 'Export resultats...',
  done: 'Termine',
}

export default function CalculateButton() {
  const { state, dispatch } = useApp()
  const cleanupRef = useRef<(() => void) | null>(null)

  const canCalculate = state.startPoint && state.endPoint && state.calcStatus !== 'running'

  async function handleClick() {
    if (!state.startPoint || !state.endPoint) return

    // cleanup eventSource precedent
    if (cleanupRef.current) {
      cleanupRef.current()
      cleanupRef.current = null
    }

    dispatch({ type: 'CALC_START' })

    try {
      const jobId = await calculateRouteAsync(
        state.startPoint, state.endPoint, state.params,
      )

      const cleanup = subscribeProgress(
        jobId,
        (msg) => {
          if (msg.status === 'completed' && msg.result) {
            dispatch({ type: 'CALC_DONE', result: msg.result })
          } else if (msg.status === 'error') {
            dispatch({ type: 'CALC_ERROR', message: msg.message })
          } else {
            dispatch({ type: 'CALC_PROGRESS', progress: msg.progress, step: msg.step })
          }
        },
        (err) => {
          dispatch({ type: 'CALC_ERROR', message: err })
        },
      )
      cleanupRef.current = cleanup
    } catch (e) {
      dispatch({ type: 'CALC_ERROR', message: (e as Error).message })
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <button
        onClick={handleClick}
        disabled={!canCalculate}
        className={`w-full py-2 rounded font-medium text-sm cursor-pointer
          ${canCalculate
            ? 'bg-green-600 hover:bg-green-500 text-white'
            : 'bg-gray-700 text-gray-500 cursor-not-allowed'
          }`}
      >
        {state.calcStatus === 'running' ? 'Calcul en cours...' : 'Calculer'}
      </button>

      {/* progress bar */}
      {state.calcStatus === 'running' && (
        <div className="flex flex-col gap-1">
          <div className="w-full h-2 bg-gray-700 rounded overflow-hidden">
            <div
              className="h-full bg-green-500 transition-all duration-300"
              style={{ width: `${state.progress}%` }}
            />
          </div>
          <span className="text-xs text-gray-400">
            {STEP_LABELS[state.progressStep] || state.progressStep}
          </span>
        </div>
      )}

      {/* erreur */}
      {state.calcStatus === 'error' && state.errorMessage && (
        <div className="text-xs text-red-400 bg-red-900/30 rounded p-2">
          {state.errorMessage}
        </div>
      )}
    </div>
  )
}
