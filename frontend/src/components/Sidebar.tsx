// sidebar retractable -- onglets calcul / historique
import { useApp } from '../context'
import ParamsForm from './ParamsForm'
import CalculateButton from './CalculateButton'
import RouteStats from './RouteStats'
import ExportButtons from './ExportButtons'
import ZonePanel from './ZonePanel'
import HistoryPanel from './HistoryPanel'

export default function Sidebar() {
  const { state, dispatch } = useApp()

  return (
    <>
      {/* toggle button -- toujours visible */}
      <button
        onClick={() => dispatch({ type: 'TOGGLE_SIDEBAR' })}
        className="absolute top-3 left-3 z-20 bg-gray-900/90 text-white
          w-9 h-9 rounded flex items-center justify-center
          hover:bg-gray-700 cursor-pointer text-lg shadow"
        style={{ left: state.sidebarOpen ? '330px' : '12px' }}
      >
        {state.sidebarOpen ? '\u2039' : '\u203A'}
      </button>

      {/* panneau */}
      <div
        className={`absolute top-0 left-0 h-full w-80 z-10
          bg-gray-900/95 backdrop-blur-sm text-white
          transition-transform duration-300 overflow-y-auto
          ${state.sidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}
      >
        <div className="p-4 flex flex-col gap-4">
          <h1 className="text-lg font-bold tracking-tight">AlpineRoute</h1>

          {/* tabs */}
          <div className="flex border-b border-gray-700">
            <button
              onClick={() => dispatch({ type: 'SET_TAB', tab: 'calcul' })}
              className={`flex-1 py-1.5 text-xs font-medium cursor-pointer
                ${state.activeTab === 'calcul'
                  ? 'text-white border-b-2 border-green-500'
                  : 'text-gray-500 hover:text-gray-300'
                }`}
            >
              Calcul
            </button>
            <button
              onClick={() => dispatch({ type: 'SET_TAB', tab: 'historique' })}
              className={`flex-1 py-1.5 text-xs font-medium cursor-pointer
                ${state.activeTab === 'historique'
                  ? 'text-white border-b-2 border-green-500'
                  : 'text-gray-500 hover:text-gray-300'
                }`}
            >
              Historique
            </button>
          </div>

          {/* contenu tab */}
          {state.activeTab === 'calcul' ? (
            <>
              <ParamsForm />
              <CalculateButton />
              {state.routeResult && <RouteStats />}
              {state.routeResult && <ExportButtons />}
              <ZonePanel />
            </>
          ) : (
            <HistoryPanel />
          )}
        </div>
      </div>
    </>
  )
}
