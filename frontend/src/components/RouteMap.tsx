// carte principale
import { useEffect, useRef, useCallback } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { TerraDraw, TerraDrawPolygonMode } from 'terra-draw'
import { TerraDrawMapLibreGLAdapter } from 'terra-draw-maplibre-gl-adapter'
import { useApp } from '../context'
import { getSelectedRoute } from '../types'
import type { BasemapId, Coord3D, ZoneType } from '../types'
import { useSlopesOverlay, useGlaciersOverlay, useCostOverlay, useAlpineRoutesOverlay, useSegmentsOverlay, useAltitudeOverlay } from '../hooks/useOverlays'
import BasemapSelector from './BasemapSelector'
import Legend from './Legend'

// position par defaut
const CENTER: [number, number] = [6.87, 45.88]
const ZOOM = 13

const MT_KEY = import.meta.env.VITE_MAPTILER_KEY || ''

// URLs basemaps -- IGN WMTS (France) + MapTiler (global)
const BASEMAP_URLS: Record<BasemapId, string> = {
  plan: 'https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=GEOGRAPHICALGRIDSYSTEMS.PLANIGNV2&STYLE=normal&FORMAT=image/png&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}',
  satellite: 'https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=ORTHOIMAGERY.ORTHOPHOTOS&STYLE=normal&FORMAT=image/jpeg&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}',
  'topo-global': `https://api.maptiler.com/maps/topo-v2/256/{z}/{x}/{y}.png?key=${MT_KEY}`,
  'satellite-global': `https://api.maptiler.com/maps/satellite/256/{z}/{x}/{y}.jpg?key=${MT_KEY}`,
}

// couleurs zones par type
const ZONE_COLORS: Record<ZoneType, string> = {
  crevasse: '#3b82f6',
  serac: '#8b5cf6',
  cornice: '#06b6d4',
  rockfall: '#f59e0b',
  forbidden: '#ef4444',
  custom: '#6b7280',
}

// couleur altitude: vert bas -> jaune -> orange -> rouge haut
function altitudeColorStops(coords: Coord3D[]): [number, string][] {
  const eles = coords.map(c => c[2])
  const minE = Math.min(...eles)
  const maxE = Math.max(...eles)
  if (maxE - minE < 1) return [[0, '#22c55e'], [1, '#22c55e']]

  let totalDist = 0
  const dists = [0]
  for (let i = 1; i < coords.length; i++) {
    const dx = coords[i][0] - coords[i - 1][0]
    const dy = coords[i][1] - coords[i - 1][1]
    totalDist += Math.sqrt(dx * dx + dy * dy)
    dists.push(totalDist)
  }
  if (totalDist === 0) return [[0, '#22c55e'], [1, '#22c55e']]

  // sous-echantillonner a 100 stops max pour perf
  const maxStops = 100
  const step = Math.max(1, Math.floor(coords.length / maxStops))

  const stops: [number, string][] = []
  for (let i = 0; i < coords.length; i += step) {
    const progress = dists[i] / totalDist
    const t = (eles[i] - minE) / (maxE - minE)
    stops.push([progress, altColor(t)])
  }
  // dernier point
  const lastProgress = 1
  const lastT = (eles[eles.length - 1] - minE) / (maxE - minE)
  if (stops[stops.length - 1][0] < 0.999) {
    stops.push([lastProgress, altColor(lastT)])
  }
  return stops
}

// vert -> jaune -> orange -> rouge
function altColor(t: number): string {
  if (t < 0.33) {
    const r = Math.round(34 + t / 0.33 * (234 - 34))
    const g = Math.round(197 - t / 0.33 * (197 - 179))
    return `rgb(${r}, ${g}, 94)`
  } else if (t < 0.66) {
    const s = (t - 0.33) / 0.33
    const r = Math.round(234 + s * (249 - 234))
    const g = Math.round(179 - s * (179 - 115))
    return `rgb(${r}, ${g}, ${Math.round(94 - s * 72)})`
  } else {
    const s = (t - 0.66) / 0.34
    const r = Math.round(249 - s * (249 - 220))
    const g = Math.round(115 - s * (115 - 38))
    return `rgb(${r}, ${g}, ${Math.round(22 - s * 22)})`
  }
}

export default function RouteMap() {
  const { state, dispatch } = useApp()
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const startMarkerRef = useRef<maplibregl.Marker | null>(null)
  const endMarkerRef = useRef<maplibregl.Marker | null>(null)
  const hoverMarkerRef = useRef<maplibregl.Marker | null>(null)
  const drawRef = useRef<TerraDraw | null>(null)

  // ref stable pour lire le state dans les callbacks map
  const stateRef = useRef(state)
  stateRef.current = state

  // -- init map --
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return

    // attribution selon la source active
    const isIGN = !state.basemap.includes('global')
    const attr = isIGN ? '&copy; IGN' : `&copy; <a href="https://www.maptiler.com/copyright/">MapTiler</a>`

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        name: 'AlpineRoute',
        sources: {
          'basemap': {
            type: 'raster',
            tiles: [BASEMAP_URLS[state.basemap]],
            tileSize: 256,
            attribution: attr,
            maxzoom: 18,
          },
          'terrain-dem': {
            type: 'raster-dem',
            tiles: [`https://api.maptiler.com/tiles/terrain-rgb-v2/{z}/{x}/{y}.webp?key=${MT_KEY}`],
            tileSize: 256,
            maxzoom: 14,
          },
        },
        layers: [{
          id: 'basemap-layer',
          type: 'raster',
          source: 'basemap',
          minzoom: 0,
          maxzoom: 18,
        }],
        terrain: { source: 'terrain-dem', exaggeration: 1.2 },
      },
      center: CENTER,
      zoom: ZOOM,
      pitch: 60,
      bearing: -20,
      maxPitch: 75,
    })

    map.addControl(new maplibregl.NavigationControl({
      visualizePitch: true,
    }), 'top-right')
    map.addControl(new maplibregl.ScaleControl(), 'bottom-left')

    // click -> place markers (guard: pas pendant le dessin)
    map.on('click', (e) => {
      if (stateRef.current.drawingMode) return

      const { lng, lat } = e.lngLat
      const s = stateRef.current
      if (!s.startPoint) {
        dispatch({ type: 'SET_START_POINT', point: { lng, lat } })
      } else if (!s.endPoint) {
        dispatch({ type: 'SET_END_POINT', point: { lng, lat } })
      }
    })

    // sources et layers au load
    map.on('load', () => {
      // -- routes alternatives (en dessous) --
      map.addSource('alt-routes', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      })
      map.addLayer({
        id: 'alt-routes-halo',
        type: 'line',
        source: 'alt-routes',
        paint: {
          'line-color': '#000',
          'line-width': 6,
          'line-opacity': 0.15,
        },
      })
      map.addLayer({
        id: 'alt-routes-line',
        type: 'line',
        source: 'alt-routes',
        paint: {
          'line-color': '#9ca3af',
          'line-width': 3.5,
          'line-opacity': 0.5,
        },
      })

      // -- route principale --
      map.addSource('route', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
        lineMetrics: true,
      })
      map.addLayer({
        id: 'route-halo',
        type: 'line',
        source: 'route',
        paint: {
          'line-color': '#000',
          'line-width': 7,
          'line-opacity': 0.35,
        },
      })
      map.addLayer({
        id: 'route-line',
        type: 'line',
        source: 'route',
        paint: {
          'line-color': '#22c55e',
          'line-width': 4,
          'line-opacity': 0.9,
        },
      })

      // -- zones utilisateur --
      map.addSource('user-zones', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      })
      map.addLayer({
        id: 'zones-fill',
        type: 'fill',
        source: 'user-zones',
        paint: {
          'fill-color': ['get', 'color'],
          'fill-opacity': 0.25,
        },
      })
      map.addLayer({
        id: 'zones-outline',
        type: 'line',
        source: 'user-zones',
        paint: {
          'line-color': ['get', 'color'],
          'line-width': 2,
          'line-dasharray': [3, 2],
          'line-opacity': 0.7,
        },
      })

      // clic sur route alternative
      map.on('click', 'alt-routes-line', (e) => {
        if (!e.features?.length) return
        const props = e.features[0].properties
        if (props && props.route_index != null) {
          dispatch({ type: 'SELECT_ROUTE', index: props.route_index })
        }
      })

      // curseur main sur les alt routes
      map.on('mouseenter', 'alt-routes-line', () => {
        map.getCanvas().style.cursor = 'pointer'
      })
      map.on('mouseleave', 'alt-routes-line', () => {
        map.getCanvas().style.cursor = ''
      })
    })

    mapRef.current = map

    return () => {
      map.remove()
      mapRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // -- basemap swap --
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    const src = map.getSource('basemap') as maplibregl.RasterTileSource | undefined
    if (src) {
      src.setTiles([BASEMAP_URLS[state.basemap]])
    }
  }, [state.basemap])

  // -- terrain 3D --
  useEffect(() => {
    const map = mapRef.current
    if (!map || !map.isStyleLoaded()) return
    if (state.is3D) {
      map.setTerrain({ source: 'terrain-dem', exaggeration: 1.2 })
      map.easeTo({ pitch: 60, duration: 500 })
    } else {
      map.setTerrain(null)
      map.easeTo({ pitch: 0, duration: 500 })
    }
  }, [state.is3D])

  // -- calques --
  useSlopesOverlay(mapRef.current, state.activeOverlays)
  useGlaciersOverlay(mapRef.current, state.activeOverlays)
  useCostOverlay(mapRef.current, state.activeOverlays, state.routeResult)
  useAlpineRoutesOverlay(mapRef.current, state.activeOverlays)
  useSegmentsOverlay(mapRef.current, state.activeOverlays)
  useAltitudeOverlay(mapRef.current, state.activeOverlays, state.routeResult)

  // helper pour creer/deplacer un marker
  const upsertMarker = useCallback((
    ref: React.RefObject<maplibregl.Marker | null>,
    point: { lng: number; lat: number } | null,
    color: string,
    onDragEnd: (lngLat: maplibregl.LngLat) => void,
  ) => {
    const map = mapRef.current
    if (!map) return

    if (!point) {
      if (ref.current) {
        ref.current.remove()
        ref.current = null
      }
      return
    }

    if (ref.current) {
      ref.current.setLngLat([point.lng, point.lat])
    } else {
      const m = new maplibregl.Marker({ color, draggable: true })
        .setLngLat([point.lng, point.lat])
        .addTo(map)
      m.on('dragend', () => {
        const pos = m.getLngLat()
        onDragEnd(pos)
      })
      ref.current = m
    }
  }, [])

  // -- start marker --
  useEffect(() => {
    upsertMarker(startMarkerRef, state.startPoint, '#22c55e', (pos) => {
      dispatch({ type: 'SET_START_POINT', point: { lng: pos.lng, lat: pos.lat } })
    })
  }, [state.startPoint, upsertMarker, dispatch])

  // -- end marker --
  useEffect(() => {
    upsertMarker(endMarkerRef, state.endPoint, '#ef4444', (pos) => {
      dispatch({ type: 'SET_END_POINT', point: { lng: pos.lng, lat: pos.lat } })
    })
  }, [state.endPoint, upsertMarker, dispatch])

  // -- affiche la route + alternatives --
  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    const routeSrc = map.getSource('route') as maplibregl.GeoJSONSource | undefined
    const altSrc = map.getSource('alt-routes') as maplibregl.GeoJSONSource | undefined
    if (!routeSrc || !altSrc) return

    if (!state.routeResult) {
      routeSrc.setData({ type: 'FeatureCollection', features: [] })
      altSrc.setData({ type: 'FeatureCollection', features: [] })
      return
    }

    const result = state.routeResult
    const allRoutes = result.routes || [result.route]
    const selected = getSelectedRoute(state.routeResult, state.selectedRouteIndex)!

    // route selectionnee -> source 'route' (avec gradient)
    routeSrc.setData({ type: 'FeatureCollection', features: [selected] })

    // routes alternatives -> source 'alt-routes' (gris)
    const altFeatures = allRoutes.filter((_, i) => i !== state.selectedRouteIndex)
    altSrc.setData({ type: 'FeatureCollection', features: altFeatures })

    // line-gradient par altitude sur la route selectionnee
    const coords = selected.geometry.coordinates as Coord3D[]
    const stops = altitudeColorStops(coords)
    const gradientExpr: maplibregl.ExpressionSpecification = [
      'interpolate', ['linear'], ['line-progress'],
      ...stops.flatMap(([p, c]) => [p, c]),
    ]
    map.setPaintProperty('route-line', 'line-gradient', gradientExpr)

    // zoom sur toutes les routes
    const allCoords = allRoutes.flatMap(r => r.geometry.coordinates as Coord3D[])
    if (allCoords.length > 0) {
      const bounds = allCoords.reduce(
        (b, c) => b.extend([c[0], c[1]]),
        new maplibregl.LngLatBounds([allCoords[0][0], allCoords[0][1]], [allCoords[0][0], allCoords[0][1]]),
      )
      map.fitBounds(bounds, { padding: 80, pitch: state.is3D ? 60 : 0 })
    }
  }, [state.routeResult, state.selectedRouteIndex, state.is3D])

  // -- hover marker (synchro profil) --
  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    if (state.hoveredIndex == null || !state.routeResult) {
      if (hoverMarkerRef.current) {
        hoverMarkerRef.current.remove()
        hoverMarkerRef.current = null
      }
      return
    }

    const selected = getSelectedRoute(state.routeResult, state.selectedRouteIndex)!
    const coords = selected.geometry.coordinates
    const idx = Math.min(state.hoveredIndex, coords.length - 1)
    const [lng, lat] = coords[idx]

    if (hoverMarkerRef.current) {
      hoverMarkerRef.current.setLngLat([lng, lat])
    } else {
      hoverMarkerRef.current = new maplibregl.Marker({ color: '#facc15' })
        .setLngLat([lng, lat])
        .addTo(map)
    }
  }, [state.hoveredIndex, state.routeResult, state.selectedRouteIndex])

  // -- zones utilisateur sur la carte --
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    const src = map.getSource('user-zones') as maplibregl.GeoJSONSource | undefined
    if (!src) return

    const features = state.zones
      .filter(z => z.active && z.geojson)
      .map(z => {
        const geo = z.geojson as Record<string, unknown>
        return {
          type: 'Feature' as const,
          geometry: (geo.geometry || geo) as GeoJSON.Geometry,
          properties: {
            id: z.id,
            name: z.name,
            zone_type: z.zone_type,
            color: ZONE_COLORS[z.zone_type] || '#6b7280',
          },
        }
      })

    src.setData({ type: 'FeatureCollection', features })
  }, [state.zones])

  // -- terra-draw pour dessiner des zones --
  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    if (state.drawingMode) {
      // creer l'instance terra-draw
      const adapter = new TerraDrawMapLibreGLAdapter({ map })
      const draw = new TerraDraw({
        adapter,
        modes: [new TerraDrawPolygonMode()],
      })
      draw.start()
      draw.setMode('polygon')

      draw.on('finish', (id) => {
        const snapshot = draw.getSnapshot()
        const feat = snapshot.find(f => f.id === id)
        if (feat) {
          dispatch({ type: 'SET_PENDING_ZONE', geojson: feat })
          dispatch({ type: 'SET_ZONE_FORM_OPEN', open: true })
        }
        dispatch({ type: 'SET_DRAWING_MODE', active: false })
      })

      drawRef.current = draw
    } else {
      // cleanup
      if (drawRef.current) {
        try { drawRef.current.stop() } catch { /* deja stoppe */ }
        drawRef.current = null
      }
    }

    return () => {
      if (drawRef.current) {
        try { drawRef.current.stop() } catch { /* cleanup */ }
        drawRef.current = null
      }
    }
  }, [state.drawingMode, dispatch])

  return (
    <div className="w-full h-full relative">
      <div ref={containerRef} className="w-full h-full" />
      <BasemapSelector />
      <Legend />
    </div>
  )
}
