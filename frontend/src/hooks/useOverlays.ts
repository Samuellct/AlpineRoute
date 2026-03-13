// hooks overlay pour RouteMap
import { useEffect, useRef } from 'react'
import maplibregl from 'maplibre-gl'
import { fetchGlaciers, fetchCostSurface, fetchAlpineRoutesGeoJSON, fetchSegmentsGeoJSON } from '../api'
import type { OverlayId, RouteResult } from '../types'

const SLOPES_OVERLAY_URL = 'https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=GEOGRAPHICALGRIDSYSTEMS.SLOPES.MOUNTAIN&STYLE=normal&FORMAT=image/png&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}'

// -- pentes IGN raster --
export function useSlopesOverlay(
  map: maplibregl.Map | null,
  overlays: OverlayId[],
) {
  useEffect(() => {
    if (!map || !map.isStyleLoaded()) return

    const srcId = 'overlay-slopes'
    const layerId = 'overlay-slopes-layer'
    const active = overlays.includes('slopes')

    if (active) {
      if (!map.getSource(srcId)) {
        map.addSource(srcId, {
          type: 'raster',
          tiles: [SLOPES_OVERLAY_URL],
          tileSize: 256,
          maxzoom: 18,
        })
      }
      if (!map.getLayer(layerId)) {
        const beforeLayer = map.getLayer('alt-routes-halo') ? 'alt-routes-halo' : undefined
        map.addLayer({
          id: layerId,
          type: 'raster',
          source: srcId,
          paint: { 'raster-opacity': 0.5 },
        }, beforeLayer)
      }
    } else {
      if (map.getLayer(layerId)) map.removeLayer(layerId)
      if (map.getSource(srcId)) map.removeSource(srcId)
    }
  }, [map, overlays])
}

// -- glaciers GeoJSON avec debounce moveend --
export function useGlaciersOverlay(
  map: maplibregl.Map | null,
  overlays: OverlayId[],
) {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (!map || !map.isStyleLoaded()) return

    const active = overlays.includes('glaciers')
    const srcId = 'overlay-glaciers'
    const fillId = 'overlay-glaciers-fill'
    const lineId = 'overlay-glaciers-line'

    if (!active) {
      if (map.getLayer(fillId)) map.removeLayer(fillId)
      if (map.getLayer(lineId)) map.removeLayer(lineId)
      if (map.getSource(srcId)) map.removeSource(srcId)
      return
    }

    // init source vide
    if (!map.getSource(srcId)) {
      map.addSource(srcId, {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      })
    }
    if (!map.getLayer(fillId)) {
      const before = map.getLayer('alt-routes-halo') ? 'alt-routes-halo' : undefined
      map.addLayer({
        id: fillId,
        type: 'fill',
        source: srcId,
        paint: { 'fill-color': '#60a5fa', 'fill-opacity': 0.3 },
      }, before)
    }
    if (!map.getLayer(lineId)) {
      map.addLayer({
        id: lineId,
        type: 'line',
        source: srcId,
        paint: { 'line-color': '#3b82f6', 'line-width': 1.5, 'line-opacity': 0.7 },
      })
    }

    const loadGlaciers = () => {
      const bounds = map.getBounds()
      const bbox = `${bounds.getWest()},${bounds.getSouth()},${bounds.getEast()},${bounds.getNorth()}`
      fetchGlaciers(bbox)
        .then(geojson => {
          const src = map.getSource(srcId) as maplibregl.GeoJSONSource | undefined
          if (src) src.setData(geojson)
        })
        .catch(err => console.warn('glaciers fetch fail:', err))
    }

    loadGlaciers()

    // refresh sur moveend (debounce)
    const onMove = () => {
      if (timerRef.current) clearTimeout(timerRef.current)
      timerRef.current = setTimeout(loadGlaciers, 500)
    }
    map.on('moveend', onMove)

    return () => {
      map.off('moveend', onMove)
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [map, overlays])
}

// -- surface de cout png --
export function useCostOverlay(
  map: maplibregl.Map | null,
  overlays: OverlayId[],
  routeResult: RouteResult | null,
) {
  const urlRef = useRef<string | null>(null)

  useEffect(() => {
    if (!map || !map.isStyleLoaded()) return

    const active = overlays.includes('cost')
    const srcId = 'overlay-cost'
    const layerId = 'overlay-cost-layer'

    if (!active || !routeResult) {
      if (map.getLayer(layerId)) map.removeLayer(layerId)
      if (map.getSource(srcId)) map.removeSource(srcId)
      if (urlRef.current) {
        URL.revokeObjectURL(urlRef.current)
        urlRef.current = null
      }
      return
    }

    // fetch le PNG
    fetchCostSurface()
      .then(data => {
        if (urlRef.current) URL.revokeObjectURL(urlRef.current)
        urlRef.current = data.imageUrl

        // remove ancien layer/source si existe
        if (map.getLayer(layerId)) map.removeLayer(layerId)
        if (map.getSource(srcId)) map.removeSource(srcId)

        const [sw, ne] = data.bounds
        map.addSource(srcId, {
          type: 'image',
          url: data.imageUrl,
          coordinates: [
            [sw[0], ne[1]],  // top-left
            [ne[0], ne[1]],  // top-right
            [ne[0], sw[1]],  // bottom-right
            [sw[0], sw[1]],  // bottom-left
          ],
        })

        const before = map.getLayer('alt-routes-halo') ? 'alt-routes-halo' : undefined
        map.addLayer({
          id: layerId,
          type: 'raster',
          source: srcId,
          paint: { 'raster-opacity': 0.6 },
        }, before)
      })
      .catch(err => console.warn('cost surface fetch fail:', err))

    return () => {
      if (urlRef.current) {
        URL.revokeObjectURL(urlRef.current)
        urlRef.current = null
      }
    }
  }, [map, overlays, routeResult])
}


// -- traces alpinisme (alpine-routes) --
export function useAlpineRoutesOverlay(
  map: maplibregl.Map | null,
  overlays: OverlayId[],
) {
  const loadedRef = useRef(false)
  const popupRef = useRef<maplibregl.Popup | null>(null)

  useEffect(() => {
    if (!map || !map.isStyleLoaded()) return

    const active = overlays.includes('alpine-routes')
    const srcId = 'overlay-alpine-routes'
    const layerId = 'overlay-alpine-routes-line'

    if (!active) {
      if (map.getLayer(layerId)) map.removeLayer(layerId)
      if (map.getSource(srcId)) map.removeSource(srcId)
      loadedRef.current = false
      if (popupRef.current) { popupRef.current.remove(); popupRef.current = null }
      return
    }

    // source vide, on charge les donnees une seule fois
    if (!map.getSource(srcId)) {
      map.addSource(srcId, {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      })
    }

    if (!map.getLayer(layerId)) {
      const before = map.getLayer('alt-routes-halo') ? 'alt-routes-halo' : undefined
      map.addLayer({
        id: layerId,
        type: 'line',
        source: srcId,
        paint: {
          'line-width': 2.5,
          'line-opacity': 0.8,
          // couleur par cotation
          'line-color': [
            'match', ['get', 'grade'],
            'F', '#22c55e',
            'F+', '#4ade80',
            'PD-', '#38bdf8',
            'PD', '#3b82f6',
            'PD+', '#2563eb',
            'AD-', '#f59e0b',
            'AD', '#f97316',
            'AD+', '#ea580c',
            'D-', '#ef4444',
            'D', '#dc2626',
            'D+', '#b91c1c',
            'TD-', '#a855f7',
            'TD', '#9333ea',
            'TD+', '#7c3aed',
            'ED', '#6d28d9',
            '#6b7280', // fallback gris
          ],
        },
      }, before)
    }

    // charger les donnees (une seule fois)
    if (!loadedRef.current) {
      loadedRef.current = true
      fetchAlpineRoutesGeoJSON()
        .then(geojson => {
          const src = map.getSource(srcId) as maplibregl.GeoJSONSource | undefined
          if (src) src.setData(geojson)
        })
        .catch(err => console.warn('alpine-routes fetch fail:', err))
    }

    // popup au hover
    const onEnter = (e: maplibregl.MapMouseEvent & { features?: maplibregl.MapGeoJSONFeature[] }) => {
      if (!e.features?.length) return
      map.getCanvas().style.cursor = 'pointer'
      const props = e.features[0].properties
      if (!props) return

      const html = [
        props.summit ? `<b>${props.summit}</b>` : '',
        props.voie ? props.voie : '',
        props.grade ? `Cotation: ${props.grade}` : '',
        props.dplus_m ? `D+: ${Math.round(props.dplus_m)}m` : '',
        props.distance_m ? `Dist: ${(props.distance_m / 1000).toFixed(1)}km` : '',
      ].filter(Boolean).join('<br/>')

      if (popupRef.current) popupRef.current.remove()
      popupRef.current = new maplibregl.Popup({ closeButton: false, offset: 10 })
        .setLngLat(e.lngLat)
        .setHTML(`<div style="font-size:12px">${html}</div>`)
        .addTo(map)
    }

    const onLeave = () => {
      map.getCanvas().style.cursor = ''
      if (popupRef.current) { popupRef.current.remove(); popupRef.current = null }
    }

    map.on('mouseenter', layerId, onEnter)
    map.on('mouseleave', layerId, onLeave)

    return () => {
      map.off('mouseenter', layerId, onEnter)
      map.off('mouseleave', layerId, onLeave)
      if (popupRef.current) { popupRef.current.remove(); popupRef.current = null }
    }
  }, [map, overlays])
}


// -- segments terrain --
export function useSegmentsOverlay(
  map: maplibregl.Map | null,
  overlays: OverlayId[],
) {
  const loadedRef = useRef(false)
  const popupRef = useRef<maplibregl.Popup | null>(null)

  useEffect(() => {
    if (!map || !map.isStyleLoaded()) return

    const active = overlays.includes('segments')
    const srcId = 'overlay-segments'
    const layerId = 'overlay-segments-line'

    if (!active) {
      if (map.getLayer(layerId)) map.removeLayer(layerId)
      if (map.getSource(srcId)) map.removeSource(srcId)
      loadedRef.current = false
      if (popupRef.current) { popupRef.current.remove(); popupRef.current = null }
      return
    }

    if (!map.getSource(srcId)) {
      map.addSource(srcId, {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      })
    }

    if (!map.getLayer(layerId)) {
      const before = map.getLayer('alt-routes-halo') ? 'alt-routes-halo' : undefined
      map.addLayer({
        id: layerId,
        type: 'line',
        source: srcId,
        paint: {
          'line-color': '#eab308',
          'line-width': 2,
          'line-opacity': 0.7,
          'line-dasharray': [4, 2],
        },
      }, before)
    }

    if (!loadedRef.current) {
      loadedRef.current = true
      fetchSegmentsGeoJSON()
        .then(geojson => {
          const src = map.getSource(srcId) as maplibregl.GeoJSONSource | undefined
          if (src) src.setData(geojson)
        })
        .catch(err => console.warn('segments fetch fail:', err))
    }

    // popup au hover
    const onEnter = (e: maplibregl.MapMouseEvent & { features?: maplibregl.MapGeoJSONFeature[] }) => {
      if (!e.features?.length) return
      map.getCanvas().style.cursor = 'pointer'
      const props = e.features[0].properties
      if (!props) return

      const html = [
        props.start_name && props.end_name ? `<b>${props.start_name} > ${props.end_name}</b>` : '',
        props.segment_type ? `Type: ${props.segment_type}` : '',
        props.distance_m ? `Dist: ${(props.distance_m / 1000).toFixed(1)}km` : '',
        props.dplus_m ? `D+: ${Math.round(props.dplus_m)}m` : '',
        props.notes || '',
      ].filter(Boolean).join('<br/>')

      if (popupRef.current) popupRef.current.remove()
      popupRef.current = new maplibregl.Popup({ closeButton: false, offset: 10 })
        .setLngLat(e.lngLat)
        .setHTML(`<div style="font-size:12px">${html}</div>`)
        .addTo(map)
    }

    const onLeave = () => {
      map.getCanvas().style.cursor = ''
      if (popupRef.current) { popupRef.current.remove(); popupRef.current = null }
    }

    map.on('mouseenter', layerId, onEnter)
    map.on('mouseleave', layerId, onLeave)

    return () => {
      map.off('mouseenter', layerId, onEnter)
      map.off('mouseleave', layerId, onLeave)
      if (popupRef.current) { popupRef.current.remove(); popupRef.current = null }
    }
  }, [map, overlays])
}
