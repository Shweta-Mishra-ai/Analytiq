import { lazy, Suspense } from 'react'
import type { Figure } from '../types'

const Plot = lazy(() => import('react-plotly.js'))

interface Props {
  figure: Figure
  onClickPoint?: (point: Record<string, unknown>) => void
}

export default function PlotlyChart({ figure, onClickPoint }: Props) {
  const layout = {
    ...figure.layout,
    autosize: true,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { l: 45, r: 15, t: 40, b: 40 },
    font: { ...(figure.layout?.font ?? {}), color: '#8b97a8', size: 11 },
  }
  return (
    <Suspense
      fallback={<div className="p-6 text-center text-xs text-mute">Loading chart…</div>}
    >
      <Plot
        data={figure.data}
        layout={layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
        onClick={(e: { points?: Record<string, unknown>[] }) => {
          if (onClickPoint && e.points?.length) onClickPoint(e.points[0])
        }}
      />
    </Suspense>
  )
}
