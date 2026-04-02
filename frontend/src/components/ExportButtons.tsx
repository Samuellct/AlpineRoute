// export GPX + GeoJSON
import { useApp } from '../context'
import { getSelectedRoute } from '../types'

function downloadFile(content: string, filename: string, mime: string) {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

function toGPX(coords: [number, number, number][], trackName: string): string {
  const trkpts = coords.map(([lon, lat, ele]) =>
    `      <trkpt lat="${lat}" lon="${lon}"><ele>${ele}</ele></trkpt>`
  ).join('\n')

  return `<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="AlpineRoute">
  <trk>
    <name>${trackName}</name>
    <trkseg>
${trkpts}
    </trkseg>
  </trk>
</gpx>`
}

export default function ExportButtons() {
  const { state } = useApp()
  if (!state.routeResult) return null

  const route = getSelectedRoute(state.routeResult, state.selectedRouteIndex)
  if (!route) return null

  const p = route.properties
  const suffix = p.route_index > 0 ? `_alt${p.route_index}` : ''
  const baseName = `AlpineRoute_${p.distance_km}km_D+${p.dplus_m}m${suffix}`

  function exportGPX() {
    const gpx = toGPX(route!.geometry.coordinates, baseName)
    downloadFile(gpx, `${baseName}.gpx`, 'application/gpx+xml')
  }

  function exportGeoJSON() {
    const fc = {
      type: 'FeatureCollection',
      features: [route],
    }
    downloadFile(JSON.stringify(fc, null, 2), `${baseName}.geojson`, 'application/geo+json')
  }

  return (
    <div className="flex gap-2">
      <button
        onClick={exportGPX}
        className="flex-1 py-1.5 text-xs bg-gray-800 hover:bg-gray-700 rounded cursor-pointer"
      >
        Export GPX
      </button>
      <button
        onClick={exportGeoJSON}
        className="flex-1 py-1.5 text-xs bg-gray-800 hover:bg-gray-700 rounded cursor-pointer"
      >
        Export Geojson
      </button>
    </div>
  )
}
