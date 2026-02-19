import { useState } from 'react'
import { useApp } from '../context'
import { createZone } from '../api'
import type { ZoneType, ZoneCreatePayload } from '../types'

const ZONE_TYPE_OPTIONS: { value: ZoneType; label: string }[] = [
  { value: 'crevasse', label: 'Crevasse' },
  { value: 'serac', label: 'Serac' },
  { value: 'cornice', label: 'Corniche' },
  { value: 'rockfall', label: 'Chute de pierres' },
  { value: 'forbidden', label: 'Interdit' },
  { value: 'custom', label: 'Personnalise' },
]

export default function ZoneForm() {
  const { state, dispatch } = useApp()
  const [name, setName] = useState('')
  const [zoneType, setZoneType] = useState<ZoneType>('forbidden')
  const [multiplier, setMultiplier] = useState(100)
  const [saving, setSaving] = useState(false)

  if (!state.zoneFormOpen) return null

  async function handleSave() {
    if (!name.trim() || !state.pendingZoneGeojson) return
    setSaving(true)
    try {
      const payload: ZoneCreatePayload = {
        name: name.trim(),
        zone_type: zoneType,
        cost_multiplier: multiplier,
        geojson: state.pendingZoneGeojson,
        active: true,
      }
      const resp = await createZone(payload)
      dispatch({
        type: 'ADD_ZONE',
        zone: { ...payload, id: resp.id },
      })
      handleClose()
    } catch (e) {
      console.warn('create zone failed', e)
    } finally {
      setSaving(false)
    }
  }

  function handleClose() {
    dispatch({ type: 'SET_ZONE_FORM_OPEN', open: false })
    dispatch({ type: 'SET_PENDING_ZONE', geojson: null })
    setName('')
    setZoneType('forbidden')
    setMultiplier(100)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-5 w-80 shadow-xl">
        <h3 className="text-white font-medium mb-3">Nouvelle zone</h3>

        <div className="flex flex-col gap-3 text-sm">
          <div>
            <label className="text-gray-400 text-xs">Nom</label>
            <input
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="ex: Crevasses Geant"
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white"
            />
          </div>

          <div>
            <label className="text-gray-400 text-xs">Type</label>
            <select
              value={zoneType}
              onChange={e => setZoneType(e.target.value as ZoneType)}
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white"
            >
              {ZONE_TYPE_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-gray-400 text-xs">Multiplicateur cout</label>
            <input
              type="number"
              value={multiplier}
              onChange={e => setMultiplier(parseFloat(e.target.value) || 100)}
              min={1}
              max={10000}
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white"
            />
          </div>

          <div className="flex gap-2 mt-1">
            <button
              onClick={handleSave}
              disabled={!name.trim() || saving}
              className="flex-1 py-1.5 bg-green-600 hover:bg-green-500 text-white
                rounded text-xs cursor-pointer disabled:bg-gray-700 disabled:text-gray-500"
            >
              {saving ? 'Sauvegarde...' : 'Sauvegarder'}
            </button>
            <button
              onClick={handleClose}
              className="flex-1 py-1.5 bg-gray-800 hover:bg-gray-700 text-white
                rounded text-xs cursor-pointer"
            >
              Annuler
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
