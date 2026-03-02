// hooks overlay pour RouteMap
import { useEffect, useRef } from 'react'
import maplibregl from 'maplibre-gl'
import { fetchGlaciers, fetchCostSurface } from '../api'
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
