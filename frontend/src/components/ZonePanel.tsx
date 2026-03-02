/* zones de danger dans l'onglet calcul*/
import { useEffect } from 'react'
import { useApp } from '../context'
import { fetchZones, deleteZone } from '../api'
import type { ZoneType } from '../types'

// labels FR pour les types
const ZONE_LABELS: Record<ZoneType, string> = {
  crevasse: 'Crevasse',
  serac: 'Serac',
  cornice: 'Corniche',
  rockfall: 'Chute de pierres',
  forbidden: 'Interdit',
  custom: 'Personnalise',
}

export default function ZonePanel() {
  const { state, dispatch } = useApp()

  async function loadZones() {
    try {
      const zones = await fetchZones()
      dispatch({ type: 'SET_ZONES', zones })
    } catch (e) {
      console.warn('zones fetch failed', e)
    }
  }

  // fetch au mount
  useEffect(() => {
    loadZones()
  }, [loadZones])

  async function handleDelete(id: number) {
    try {
      await deleteZone(id)
      dispatch({ type: 'REMOVE_ZONE', id })
    } catch (e) {
      console.warn('delete zone failed', e)
    }
  }

  function toggleDraw() {
    dispatch({ type: 'SET_DRAWING_MODE', active: !state.drawingMode })
  }

  return (
    <div>
      <h3 className="text-xs text-gray-400 mb-2 uppercase tracking-wider">
        Zones de danger
      </h3>

      <button
        onClick={toggleDraw}
        className={`w-full py-1.5 text-xs rounded cursor-pointer mb-2
          ${state.drawingMode
            ? 'bg-red-600/80 hover:bg-red-500 text-white'
            : 'bg-gray-800 hover:bg-gray-700 text-white'
          }`}
      >
        {state.drawingMode ? 'Annuler dessin' : 'Dessiner zone'}
      </button>

      {state.drawingMode && (
        <p className="text-xs text-yellow-400/80 mb-2">
          Cliquer pour placer les sommets, double-clic pour terminer
        </p>
      )}

      {state.zones.length === 0 && (
        <p className="text-xs text-gray-500">Aucune zone</p>
      )}

      {state.zones.map(z => (
        <div
          key={z.id}
          className="flex justify-between items-center py-1 text-sm"
        >
          <div className="flex-1 truncate">
            <span className="text-white">{z.name}</span>
            <span className="text-gray-500 ml-1.5 text-xs">
              {ZONE_LABELS[z.zone_type] || z.zone_type}
            </span>
          </div>
          <button
            onClick={() => handleDelete(z.id)}
            className="text-xs text-gray-500 hover:text-red-400 cursor-pointer ml-2"
          >
            suppr.
          </button>
        </div>
      ))}
    </div>
  )
}
