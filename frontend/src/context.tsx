// state global - context + useReducer
import { createContext, useContext, useReducer, type ReactNode } from 'react'
import type {
  MarkerPoint, RouteParams, RouteResult,
  BasemapId, OverlayId, SidebarTab, RouteSummary, RouteDetail, UserZone,
} from './types'

// -- state --
export interface AppState {
  startPoint: MarkerPoint | null
  endPoint: MarkerPoint | null
  params: RouteParams
  calcStatus: 'idle' | 'running' | 'done' | 'error'
  progress: number
  progressStep: string
  errorMessage: string | null
  routeResult: RouteResult | null
  hoveredIndex: number | null
  basemap: BasemapId
  sidebarOpen: boolean
  is3D: boolean
  selectedRouteIndex: number // routes alternatives
  activeTab: SidebarTab
  historyRoutes: RouteSummary[]
  historyLoading: boolean
  loadedHistoryRoute: RouteDetail | null
  zones: UserZone[] //fonctionnemnt bof bof a delete pour la v2
  drawingMode: boolean
  pendingZoneGeojson: object | null
  zoneFormOpen: boolean
  activeOverlays: OverlayId[]
}

const defaultParams: RouteParams = {
  resolution: 1.0,
  month: 7,
  acclimatized: true,
  n_alternatives: 0,
  anisotropic: false,
}

const initialState: AppState = {
  startPoint: null,
  endPoint: null,
  params: defaultParams,
  calcStatus: 'idle',
  progress: 0,
  progressStep: '',
  errorMessage: null,
  routeResult: null,
  hoveredIndex: null,
  basemap: 'plan',
  sidebarOpen: true,
  is3D: true,
  selectedRouteIndex: 0,
  activeTab: 'calcul',
  historyRoutes: [],
  historyLoading: false,
  loadedHistoryRoute: null,
  zones: [],
  drawingMode: false,
  pendingZoneGeojson: null,
  zoneFormOpen: false,
  activeOverlays: [],
}

// -- actions --
type Action =
  | { type: 'SET_START_POINT'; point: MarkerPoint | null }
  | { type: 'SET_END_POINT'; point: MarkerPoint | null }
  | { type: 'SET_PARAMS'; params: Partial<RouteParams> }
  | { type: 'CALC_START' }
  | { type: 'CALC_PROGRESS'; progress: number; step: string }
  | { type: 'CALC_DONE'; result: RouteResult }
  | { type: 'CALC_ERROR'; message: string }
  | { type: 'SET_HOVERED_INDEX'; index: number | null }
  | { type: 'SET_BASEMAP'; basemap: BasemapId }
  | { type: 'TOGGLE_SIDEBAR' }
  | { type: 'TOGGLE_3D' }
  | { type: 'SELECT_ROUTE'; index: number }
  | { type: 'SET_TAB'; tab: SidebarTab }
  | { type: 'HISTORY_LOADING' }
  | { type: 'HISTORY_LOADED'; routes: RouteSummary[] }
  | { type: 'LOAD_HISTORY_ROUTE'; detail: RouteDetail }
  | { type: 'CLEAR_HISTORY_ROUTE' }
  | { type: 'SET_ZONES'; zones: UserZone[] }
  | { type: 'ADD_ZONE'; zone: UserZone }
  | { type: 'REMOVE_ZONE'; id: number }
  | { type: 'SET_DRAWING_MODE'; active: boolean }
  | { type: 'SET_PENDING_ZONE'; geojson: object | null }
  | { type: 'SET_ZONE_FORM_OPEN'; open: boolean }
  | { type: 'TOGGLE_OVERLAY'; overlay: OverlayId }
  | { type: 'RESET' }

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SET_START_POINT':
      return { ...state, startPoint: action.point }
    case 'SET_END_POINT':
      return { ...state, endPoint: action.point }
    case 'SET_PARAMS':
      return { ...state, params: { ...state.params, ...action.params } }
    case 'CALC_START':
      return {
        ...state, calcStatus: 'running', progress: 0,
        progressStep: 'init', errorMessage: null, routeResult: null,
        selectedRouteIndex: 0, loadedHistoryRoute: null,
      }
    case 'CALC_PROGRESS':
      return { ...state, progress: action.progress, progressStep: action.step }
    case 'CALC_DONE':
      return {
        ...state, calcStatus: 'done', progress: 100,
        progressStep: 'done', routeResult: action.result,
        selectedRouteIndex: 0,
      }
    case 'CALC_ERROR':
      return { ...state, calcStatus: 'error', errorMessage: action.message }
    case 'SET_HOVERED_INDEX':
      return { ...state, hoveredIndex: action.index }
    case 'SET_BASEMAP':
      return { ...state, basemap: action.basemap }
    case 'TOGGLE_SIDEBAR':
      return { ...state, sidebarOpen: !state.sidebarOpen }
    case 'TOGGLE_3D':
      return { ...state, is3D: !state.is3D }
    case 'SELECT_ROUTE':
      return { ...state, selectedRouteIndex: action.index, hoveredIndex: null }
    case 'SET_TAB':
      return { ...state, activeTab: action.tab }
    case 'HISTORY_LOADING':
      return { ...state, historyLoading: true }
    case 'HISTORY_LOADED':
      return { ...state, historyLoading: false, historyRoutes: action.routes }
    case 'LOAD_HISTORY_ROUTE': {
      // on peuple routeResult avec le geojson de la route historique
      const feat = action.detail.geojson
      const fakeResult: RouteResult = {
        status: 'ok',
        route: feat,
        computation_time_s: action.detail.computation_time_s,
      }
      return {
        ...state, loadedHistoryRoute: action.detail,
        routeResult: fakeResult, calcStatus: 'done',
        selectedRouteIndex: 0, activeTab: 'calcul',
      }
    }
    case 'CLEAR_HISTORY_ROUTE':
      return { ...state, loadedHistoryRoute: null }
    case 'SET_ZONES':
      return { ...state, zones: action.zones }
    case 'ADD_ZONE':
      return { ...state, zones: [...state.zones, action.zone] }
    case 'REMOVE_ZONE':
      return { ...state, zones: state.zones.filter(z => z.id !== action.id) }
    case 'SET_DRAWING_MODE':
      return { ...state, drawingMode: action.active }
    case 'SET_PENDING_ZONE':
      return { ...state, pendingZoneGeojson: action.geojson }
    case 'SET_ZONE_FORM_OPEN':
      return { ...state, zoneFormOpen: action.open }
    case 'TOGGLE_OVERLAY': {
      const has = state.activeOverlays.includes(action.overlay)
      return {
        ...state,
        activeOverlays: has
          ? state.activeOverlays.filter(o => o !== action.overlay)
          : [...state.activeOverlays, action.overlay],
      }
    }
    case 'RESET':
      return {
        ...initialState, basemap: state.basemap,
        is3D: state.is3D, sidebarOpen: state.sidebarOpen,
        zones: state.zones, activeOverlays: state.activeOverlays,
      }
    default:
      return state
  }
}

// -- context --
interface AppContextType {
  state: AppState
  dispatch: React.Dispatch<Action>
}

const AppContext = createContext<AppContextType | null>(null)

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState)
  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  )
}

export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be inside AppProvider')
  return ctx
}
