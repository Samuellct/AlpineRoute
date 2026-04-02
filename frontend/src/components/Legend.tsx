// legende des calques overlays actifs
import { useApp } from '../context'
import type { OverlayId } from '../types'

// gradient css vert -> jaune -> rouge
const GRADIENT_GYR = 'linear-gradient(to right, #22c55e, #eab308, #ef4444)'

// cotations alpines et couleurs (match useOverlays)
const GRADES = [
  { label: 'F', color: '#22c55e' },
  { label: 'PD', color: '#3b82f6' },
  { label: 'AD', color: '#f97316' },
  { label: 'D', color: '#dc2626' },
  { label: 'TD', color: '#9333ea' },
  { label: 'ED', color: '#6d28d9' },
]

function LegendBlock({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[10px] text-white/60 uppercase tracking-wider">{title}</span>
      {children}
    </div>
  )
}

function GradientBar({ left, right }: { left: string; right: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <div className="h-2 w-full rounded" style={{ background: GRADIENT_GYR }} />
      <div className="flex justify-between text-[9px] text-gray-400">
        <span>{left}</span>
        <span>{right}</span>
      </div>
    </div>
  )
}

// legende specifique par calque
function OverlayLegend({ id }: { id: OverlayId }) {
  switch (id) {
    case 'slopes':
      return (
        <LegendBlock title="Pentes">
          <div className="flex flex-col gap-0.5">
            <div className="h-2 w-full rounded" style={{
              background: 'linear-gradient(to right, #22c55e, #eab308, #f97316, #ef4444, #7c2d12)',
            }} />
            <div className="flex justify-between text-[9px] text-gray-400">
              <span>0</span>
              <span>30</span>
              <span>45+</span>
            </div>
          </div>
        </LegendBlock>
      )
    case 'glaciers':
      return (
        <LegendBlock title="Glaciers">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-sm bg-blue-400/70" />
            <span className="text-[10px] text-gray-300">Glacier RGI</span>
          </div>
        </LegendBlock>
      )
    case 'cost':
      return (
        <LegendBlock title="Surface de coût">
          <GradientBar left="Faible" right="Élevé" />
        </LegendBlock>
      )
    case 'alpine-routes':
      return (
        <LegendBlock title="Traces alpines">
          <div className="flex flex-wrap gap-x-2 gap-y-0.5">
            {GRADES.map(g => (
              <div key={g.label} className="flex items-center gap-1">
                <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: g.color }} />
                <span className="text-[10px] text-gray-300">{g.label}</span>
              </div>
            ))}
          </div>
        </LegendBlock>
      )
    case 'segments':
      return (
        <LegendBlock title="Segments terrain">
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-0.5 border-t-2 border-dashed border-yellow-500" />
            <span className="text-[10px] text-gray-300">Segment</span>
          </div>
        </LegendBlock>
      )
    case 'altitude':
      return (
        <LegendBlock title="Altitude MNT">
          <GradientBar left="Bas" right="Haut" />
        </LegendBlock>
      )
    default:
      return null
  }
}

export default function Legend() {
  const { state } = useApp()
  const active = state.activeOverlays

  if (active.length === 0) return null

  return (
    <div className="absolute bottom-6 left-3 z-10 flex flex-col gap-2 bg-gray-900/90 backdrop-blur-sm rounded-lg px-3 py-2 max-w-[180px] shadow-lg">
      {active.map(id => (
        <OverlayLegend key={id} id={id} />
      ))}
    </div>
  )
}
