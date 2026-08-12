import { useEffect, useMemo, useState } from 'react'
import { Activity } from 'lucide-react'
import { apiGet, apiPost } from '../api/client'
import { useApp } from '../store/app'
import PlotlyChart from '../components/PlotlyChart'
import type { Figure } from '../types'
import { Btn, ErrorBox, NeedData, PageHeader, Panel, Spinner } from '../components/Ui'

interface Fields {
  duration_columns: string[]
  event_columns: string[]
  group_columns: string[]
}

interface SurvivalPoint {
  time: number
  at_risk: number
  events: number
  survival_prob: number
  ci_lower: number
  ci_upper: number
}

interface SurvivalCurve {
  label: string
  n_total: number
  n_events: number
  n_censored: number
  median_survival: number | null
  points: SurvivalPoint[]
  milestone_probs: Record<string, number>
}

interface Comparison {
  group_a: string
  group_b: string
  logrank_stat: number
  p_value: number
  is_significant: boolean
  verdict: string
}

interface SurvivalReport {
  duration_col: string
  event_col: string
  group_col: string | null
  overall_curve: SurvivalCurve
  group_curves: SurvivalCurve[]
  pairwise_comparisons: Comparison[]
  summary: string
  warnings: string[]
}

const PALETTE = ['#4f8ef7', '#22d3a5', '#f7934f', '#a78bfa', '#f77070']

export default function SurvivalPage() {
  const dataset = useApp((s) => s.dataset)
  const ds = dataset?.dataset_id

  const [fields, setFields] = useState<Fields | null>(null)
  const [duration, setDuration] = useState('')
  const [event, setEvent] = useState('')
  const [group, setGroup] = useState('')
  const [report, setReport] = useState<SurvivalReport | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!ds) return
    setReport(null)
    setError('')
    apiGet<Fields>(`/api/analytics/${ds}/survival/fields`)
      .then((f) => {
        setFields(f)
        setDuration(f.duration_columns[0] ?? '')
        setEvent(f.event_columns[0] ?? '')
        setGroup('')
      })
      .catch((e) => setError(e.message))
  }, [ds])

  const figure: Figure | null = useMemo(() => {
    if (!report) return null
    const curves =
      report.group_curves.length > 0 ? report.group_curves : [report.overall_curve]
    return {
      data: curves.map((c, i) => ({
        x: [0, ...c.points.map((p) => p.time)],
        y: [1, ...c.points.map((p) => p.survival_prob)],
        type: 'scatter',
        mode: 'lines',
        line: { shape: 'hv', color: PALETTE[i % PALETTE.length], width: 2 },
        name: c.label,
      })),
      layout: {
        showlegend: curves.length > 1,
        legend: { orientation: 'h', y: -0.2 },
        xaxis: { title: report.duration_col, gridcolor: '#232d3f' },
        yaxis: {
          title: 'Still active',
          tickformat: '.0%',
          range: [0, 1.02],
          gridcolor: '#232d3f',
        },
      },
    }
  }, [report])

  if (!dataset) return <NeedData />

  const run = async () => {
    setBusy(true)
    setError('')
    try {
      setReport(
        await apiPost<SurvivalReport>(`/api/analytics/${ds}/survival`, {
          duration_col: duration,
          event_col: event,
          group_col: group || null,
        }),
      )
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setReport(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="p-8">
      <PageHeader
        title="Survival Analysis"
        subtitle="How long until people leave, churn or cancel — and which groups last longer"
      />

      {error && (
        <div className="mb-4">
          <ErrorBox message={error} />
        </div>
      )}

      <Panel title="Choose the time and the event">
        <div className="flex flex-wrap items-end gap-3">
          <Select
            label="Time elapsed"
            value={duration}
            onChange={setDuration}
            options={fields?.duration_columns ?? []}
          />
          <Select
            label="Event happened"
            value={event}
            onChange={setEvent}
            options={fields?.event_columns ?? []}
          />
          <Select
            label="Split by (optional)"
            value={group}
            onChange={setGroup}
            options={['', ...(fields?.group_columns ?? [])]}
          />
          <Btn onClick={run} disabled={busy || !duration || !event}>
            <span className="flex items-center gap-1.5">
              <Activity className="h-4 w-4" /> Run analysis
            </span>
          </Btn>
        </div>
        <p className="mt-2 text-xs text-mute">
          &quot;Event happened&quot; is a yes/no column such as attrition, churn or
          cancelled. Rows where it hasn&apos;t happened yet are censored, not
          ignored — that is what makes this more honest than a simple average.
        </p>
      </Panel>

      {busy && <Spinner label="Estimating survival curves…" />}

      {report && !busy && (
        <div className="mt-5 space-y-5">
          <Panel>
            <p className="text-sm text-ink">{report.summary}</p>
          </Panel>

          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat
              label="Records"
              value={report.overall_curve.n_total.toLocaleString()}
            />
            <Stat
              label="Event occurred"
              value={report.overall_curve.n_events.toLocaleString()}
              tone="rose"
            />
            <Stat
              label="Still active"
              value={report.overall_curve.n_censored.toLocaleString()}
              tone="teal"
            />
            <Stat
              label="Median lifetime"
              value={
                report.overall_curve.median_survival == null
                  ? 'Not reached'
                  : report.overall_curve.median_survival.toFixed(1)
              }
            />
          </div>

          {figure && (
            <Panel title="Survival curve">
              <div className="h-80">
                <PlotlyChart figure={figure} />
              </div>
            </Panel>
          )}

          {Object.keys(report.overall_curve.milestone_probs).length > 0 && (
            <Panel title="Still active after…">
              <div className="flex flex-wrap gap-6">
                {Object.entries(report.overall_curve.milestone_probs).map(
                  ([t, p]) => (
                    <div key={t}>
                      <div className="text-[11px] text-mute uppercase">
                        {Number(t).toFixed(0)} {report.duration_col}
                      </div>
                      <div className="text-lg font-bold text-accent">
                        {(p * 100).toFixed(1)}%
                      </div>
                    </div>
                  ),
                )}
              </div>
            </Panel>
          )}

          {report.group_curves.length > 0 && (
            <Panel title="By group">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="text-mute">
                    {['Group', 'n', 'Events', 'Still active', 'Median lifetime'].map(
                      (h) => (
                        <th key={h} className="px-2 py-2">{h}</th>
                      ),
                    )}
                  </tr>
                </thead>
                <tbody>
                  {report.group_curves.map((c, i) => (
                    <tr key={c.label} className="border-t border-edge/60">
                      <td className="px-2 py-2 font-semibold text-ink">
                        <span className="flex items-center gap-2">
                          <span
                            className="inline-block h-2.5 w-2.5 rounded-full"
                            style={{ backgroundColor: PALETTE[i % PALETTE.length] }}
                          />
                          {c.label}
                        </span>
                      </td>
                      <td className="px-2 py-2 text-mute">{c.n_total}</td>
                      <td className="px-2 py-2 text-mute">{c.n_events}</td>
                      <td className="px-2 py-2 text-mute">{c.n_censored}</td>
                      <td className="px-2 py-2 text-mute">
                        {c.median_survival == null
                          ? 'Not reached'
                          : c.median_survival.toFixed(1)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Panel>
          )}

          {report.pairwise_comparisons.length > 0 && (
            <Panel title="Do the groups really differ? (log-rank test)">
              <ul className="space-y-2 text-sm">
                {report.pairwise_comparisons.map((c, i) => (
                  <li key={i} className="flex gap-2">
                    <span
                      className={c.is_significant ? 'text-teal' : 'text-mute'}
                    >
                      {c.is_significant ? '✓' : '·'}
                    </span>
                    <span className="text-mute">{c.verdict}</span>
                  </li>
                ))}
              </ul>
            </Panel>
          )}

          {report.warnings.length > 0 && (
            <Panel title="Notes">
              <ul className="space-y-1 text-xs text-amber">
                {report.warnings.map((w, i) => (
                  <li key={i}>⚠ {w}</li>
                ))}
              </ul>
            </Panel>
          )}
        </div>
      )}
    </div>
  )
}

function Select({
  label,
  value,
  onChange,
  options,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  options: string[]
}) {
  return (
    <div className="min-w-44">
      <label className="mb-1 block text-[11px] text-mute uppercase">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o === '' ? '— none —' : o}
          </option>
        ))}
      </select>
    </div>
  )
}

function Stat({
  label,
  value,
  tone = 'ink',
}: {
  label: string
  value: string
  tone?: 'ink' | 'teal' | 'rose'
}) {
  const tones = { ink: 'text-ink', teal: 'text-teal', rose: 'text-rose' }
  return (
    <Panel>
      <div className="text-xs text-mute uppercase">{label}</div>
      <div className={`text-xl font-bold ${tones[tone]}`}>{value}</div>
    </Panel>
  )
}
