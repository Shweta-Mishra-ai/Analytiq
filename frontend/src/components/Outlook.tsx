import { useEffect, useState } from 'react'
import { apiGet } from '../api/client'
import PlotlyChart from './PlotlyChart'
import { Panel } from './Ui'

interface Segment {
  segment: string
  prior: number
  current: number
  delta: number
  share: number
}

interface Why {
  measure: string
  period: string
  prior_period: string
  pct: number
  column: string | null
  segments: Segment[]
  sentences: string[]
}

interface Point {
  month: string
  value: number
  low: number
  high: number
}

interface Forecast {
  measure: string
  method: string
  aggregate: string
  error_pct: number
  naive_error_pct: number
  n_months: number
  points: Point[]
  history: { month: string; value: number }[]
  sentences: string[]
}

interface Priority {
  rank: number
  finding: string
  action: string
  value: number
  value_basis: string
  evidence: string
  level: string
}

export interface OutlookData {
  why: Why | null
  forecast: Forecast | null
  priorities: Priority[]
  notes: string[]
}

function compactMoney(v: number): string {
  const a = Math.abs(v)
  const sign = v < 0 ? '−' : ''
  if (a >= 1e9) return `${sign}${(a / 1e9).toFixed(1)}B`
  if (a >= 1e6) return `${sign}${(a / 1e6).toFixed(1)}M`
  if (a >= 1e3) return `${sign}${(a / 1e3).toFixed(0)}K`
  return `${sign}${a.toFixed(0)}`
}

const levelStyle: Record<string, string> = {
  High: 'bg-rose/15 text-rose',
  Medium: 'bg-amber/15 text-amber',
  Low: 'bg-accent/15 text-accent',
}

function forecastFigure(f: Forecast) {
  const hx = f.history.map((h) => h.month)
  const fx = f.points.map((p) => p.month)
  // The band joins the last actual month so the eye follows on from it.
  const last = f.history[f.history.length - 1]
  const bx = [last.month, ...fx]
  return {
    data: [
      { x: hx, y: f.history.map((h) => h.value), type: 'scatter', mode: 'lines+markers',
        name: 'Actual', line: { color: '#5b8def', width: 2 }, marker: { size: 4 } },
      { x: bx, y: [last.value, ...f.points.map((p) => p.high)], type: 'scatter', mode: 'lines',
        line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
      { x: bx, y: [last.value, ...f.points.map((p) => p.low)], type: 'scatter', mode: 'lines',
        fill: 'tonexty', fillcolor: 'rgba(91,141,239,0.30)', line: { width: 0 },
        name: '8 in 10 range', hoverinfo: 'skip' },
      { x: bx, y: [last.value, ...f.points.map((p) => p.value)], type: 'scatter',
        mode: 'lines+markers', name: 'Forecast',
        line: { color: '#5b8def', width: 2, dash: 'dot' }, marker: { size: 5 } },
    ],
    // A total starts from zero, as on the report's trend chart: a 5% dip
    // drawn from 65K looked like a collapse.
    layout: {
      showlegend: true,
      legend: { orientation: 'h', y: -0.2 },
      height: 260,
      yaxis: { rangemode: f.aggregate === 'total' ? 'tozero' : 'normal' },
    },
  }
}

export default function Outlook({ ds }: { ds: string }) {
  const [data, setData] = useState<OutlookData | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    setData(null)
    setError('')
    apiGet<OutlookData>(`/api/analytics/${ds}/outlook`)
      .then(setData)
      .catch((e) => setError(e.message))
  }, [ds])

  if (error) return <p className="text-xs text-rose">Outlook unavailable: {error}</p>
  if (!data) return <p className="text-xs text-mute">Working out what changed and what comes next…</p>
  if (!data.why && !data.forecast && !data.priorities.length) return null

  return (
    <div className="space-y-5" data-testid="outlook">
      {data.priorities.length > 0 && (
        <Panel
          title="What to act on first"
          subtitle="Ranked by the money involved a year, weighted by how firm the evidence is"
        >
          <ol className="space-y-3">
            {data.priorities.map((p) => (
              <li key={p.rank} className="flex gap-3" data-testid="priority">
                <span className="mt-0.5 text-sm font-bold text-mute">{p.rank}</span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold uppercase ${levelStyle[p.level] ?? ''}`}>
                      {p.level}
                    </span>
                    <span className="text-sm font-semibold text-ink">{p.finding}</span>
                  </div>
                  <p className="mt-1 text-xs text-ink">→ {p.action}</p>
                  <p className="mt-1 text-[11px] text-mute">
                    <span className="font-semibold">{compactMoney(p.value)}</span> — {p.value_basis}.{' '}
                    {p.evidence}.
                  </p>
                </div>
              </li>
            ))}
          </ol>
        </Panel>
      )}

      <div className="grid gap-5 lg:grid-cols-2">
        {data.why && (
          <Panel title="What changed and why" subtitle={`${data.why.period} against the ${data.why.prior_period}`}>
            <div className={`text-2xl font-bold ${data.why.pct >= 0 ? 'text-teal' : 'text-rose'}`}>
              {data.why.pct >= 0 ? '+' : '−'}
              {Math.abs(data.why.pct).toFixed(1)}%
            </div>
            {data.why.column && data.why.segments.length > 0 && (
              <div className="mt-3 space-y-1.5">
                <div className="text-[11px] uppercase text-mute">By {data.why.column}</div>
                {data.why.segments.map((s) => (
                  <div key={s.segment} className="flex items-center gap-2 text-xs">
                    <span className="w-28 truncate text-ink">{s.segment}</span>
                    <div className="h-2 flex-1 rounded bg-white/5">
                      <div
                        className={`h-2 rounded ${s.delta < 0 ? 'bg-rose/70' : 'bg-teal/70'}`}
                        style={{ width: `${Math.max(4, Math.round(s.share * 100))}%` }}
                      />
                    </div>
                    <span className={`w-16 text-right ${s.delta < 0 ? 'text-rose' : 'text-teal'}`}>
                      {s.delta > 0 ? '+' : ''}
                      {compactMoney(s.delta)}
                    </span>
                  </div>
                ))}
              </div>
            )}
            <ul className="mt-3 space-y-1 text-xs text-mute">
              {data.why.sentences.slice(1).map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ul>
          </Panel>
        )}

        {data.forecast && (
          <Panel title="Next three months" subtitle={`Method: ${data.forecast.method}`}>
            <div className="h-64">
              <PlotlyChart figure={forecastFigure(data.forecast)} />
            </div>
            <p className="mt-2 text-xs text-mute" data-testid="forecast-accuracy">
              Tested on the last {data.forecast.n_months} months as if they were the future: off by{' '}
              <span className="font-semibold text-ink">{data.forecast.error_pct.toFixed(1)}%</span> on
              average, against {data.forecast.naive_error_pct.toFixed(1)}% for holding last month&apos;s
              level.
            </p>
            <p className="mt-1 text-[11px] text-mute">{data.forecast.sentences[2]}</p>
          </Panel>
        )}
      </div>

      {data.notes.length > 0 && (
        <ul className="space-y-1 text-[11px] text-mute">
          {data.notes.map((n, i) => (
            <li key={i}>ⓘ {n}</li>
          ))}
        </ul>
      )}
    </div>
  )
}
