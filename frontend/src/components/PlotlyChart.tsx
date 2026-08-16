import { lazy, Suspense } from 'react'
import type { Figure } from '../types'

const Plot = lazy(() => import('react-plotly.js'))

interface Props {
  figure: Figure
  onClickPoint?: (point: Record<string, unknown>) => void
  /** Hide the figure's own title — for tiles that print it in their header. */
  bare?: boolean
}

/**
 * The server builds a fully styled figure. This component used to
 * overwrite the parts of it that matter most: a hardcoded
 * `margin: { t: 40 }` clipped any title taller than one line, and
 * `font.color: '#8b97a8'` — a colour that is not in the design tokens —
 * replaced whatever the figure asked for. So the finding-led titles were
 * cut in half and every chart drew its text in a fifth grey.
 *
 * Layout coming from the server now wins. Only genuinely client-side
 * concerns are set here: autosize, and transparency so the chart sits on
 * the panel rather than painting its own background.
 */
export default function PlotlyChart({ figure, onClickPoint, bare }: Props) {
  const layout = {
    ...figure.layout,
    autosize: true,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    ...(bare
      ? {
          title: undefined,
          margin: { ...(figure.layout?.margin ?? {}), t: 8 },
        }
      : {}),
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
