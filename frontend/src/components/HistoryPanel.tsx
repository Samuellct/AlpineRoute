// historique -liste des routes calc
import { useEffect, useCallback } from 'react'
import { useApp } from '../context'
import { fetchRouteHistory, fetchRouteDetail, deleteRoute } from '../api'

export default function HistoryPanel() {
  const { state, dispatch } = useApp()

  const loadHistory = useCallback(async () => {
    dispatch({ type: 'HISTORY_LOADING' })
    try {
      const routes = await fetchRouteHistory(50, 0)
      dispatch({ type: 'HISTORY_LOADED', routes })
    } catch (e) {
      console.warn('history fetch failed', e)
      dispatch({ type: 'HISTORY_LOADED', routes: [] })
    }
  }, [dispatch])

  // charger la liste au mount et a chaque switch
  useEffect(() => {
    if (state.activeTab !== 'historique') return
    loadHistory()
  }, [state.activeTab, loadHistory])

  async function handleLoad(id: number) {
    try {
      const detail = await fetchRouteDetail(id)
      dispatch({ type: 'LOAD_HISTORY_ROUTE', detail })
    } catch (e) {
      console.warn('route detail fetch failed', e)
    }
  }

  async function handleDelete(id: number) {
    try {
      await deleteRoute(id)
      // refresh la liste
      loadHistory()
    } catch (e) {
      console.warn('delete route failed', e)
    }
  }

  if (state.historyLoading) {
    return <div className="text-gray-400 text-sm py-4">Chargement...</div>
  }

  if (state.historyRoutes.length === 0) {
    return <div className="text-gray-500 text-sm py-4">Aucune route sauvegardee</div>
  }

  return (
    <div className="flex flex-col gap-2">
      {state.historyRoutes.map(r => {
        const dist = r.distance_m ? (r.distance_m / 1000).toFixed(1) : '?'
        const dplus = r.dplus_m ? Math.round(r.dplus_m) : '?'
        const date = r.created_at?.split('T')[0] || ''
        const name = r.name || `Route #${r.id}`

        return (
          <div
            key={r.id}
            className="bg-gray-800/80 rounded p-2.5 hover:bg-gray-700/80 transition-colors"
          >
            <div className="flex justify-between items-start mb-1">
              <button
                onClick={() => handleLoad(r.id)}
                className="text-sm font-medium text-white hover:text-green-400
                  cursor-pointer text-left flex-1 truncate"
              >
                {name}
              </button>
              <button
                onClick={() => handleDelete(r.id)}
                className="text-xs text-gray-500 hover:text-red-400 cursor-pointer ml-2 shrink-0"
              >
                suppr.
              </button>
            </div>
            <div className="flex gap-3 text-xs text-gray-400">
              <span>{dist} km</span>
              <span>D+{dplus}m</span>
              <span>{date}</span>
            </div>
          </div>
        )
      })}
    </div>
  )
}
