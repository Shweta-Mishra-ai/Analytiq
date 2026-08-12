import { useEffect, useState } from 'react'
import { apiGet } from '../api/client'
import { useApp } from '../store/app'
import { ErrorBox, NeedData, PageHeader, Panel, Spinner } from '../components/Ui'

interface RootCause {
  target_col: string
  n_low_performers: number
  low_pct: number
  drivers: { factor: string; impact: string; direction: string; detail: string }[]
  top_driver: string
  interpretation: string
  recommendations: string[]
}

interface Cohort {
  cohort_col: string
  metric_col: string
  cohorts: { name: string; n: number; mean: number; rank: number; vs_avg_pct: number }[]
  best_cohort: string
  worst_cohort: string
  gap_pct: number
  is_significant: boolean
  interpretation: string
}

interface Pareto {
  group_col: string
  value_col: string
  top_groups_share: number
  pareto_holds: boolean
  interpretation: string
  groups: { name: string; value: number; cumulative_pct: number; in_top_20: boolean }[]
}

interface Segment {
  segment_name: string
  segment_col: string
  n: number
  health_score: number
  strengths: string[]
  weaknesses: string[]
  opportunity: string
}

interface BiReport {
  root_causes: RootCause[]
  cohorts: Cohort[]
  pareto: Pareto[]
  segments: Segment[]
  key_insights: string[]
  executive_brief: string
}

export default function BiPage() {
  const dataset = useApp((s) => s.dataset)
  const [report, setReport] = useState<BiReport | null>(null)
  const [error, setError] = useState('')
  const ds = dataset?.dataset_id

  useEffect(() => {
    if (!ds) return
    setReport(null)
    setError('')
    apiGet<BiReport>(`/api/analytics/${ds}/bi`)
      .then(setReport)
      .catch((e) => setError(e.message))
  }, [ds])

  if (!dataset) return <NeedData />

  return (
    <div className="p-8">
      <PageHeader
        title="Business Intelligence"
        subtitle="Root cause, cohorts, Pareto and segment health"
      />
      {error && <ErrorBox message={error} />}
      {!report && !error && <Spinner label="Running BI analysis…" />}

      {report && (
        <div className="space-y-5">
          {report.executive_brief && (
            <Panel title="Executive brief">
              <p className="text-sm leading-relaxed text-mute">{report.executive_brief}</p>
            </Panel>
          )}

          {report.key_insights.length > 0 && (
            <Panel title="Key insights">
              <ul className="space-y-1.5 text-sm text-mute">
                {report.key_insights.map((k, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="text-accent">▸</span> {k}
                  </li>
                ))}
              </ul>
            </Panel>
          )}

          {report.root_causes.map((rc, i) => (
            <Panel key={i} title={`Root cause — low ${rc.target_col}`}>
              <p className="mb-3 text-xs text-mute">
                {rc.n_low_performers.toLocaleString()} low performers ({rc.low_pct.toFixed(1)}%).{' '}
                {rc.interpretation}
              </p>
              <div className="space-y-1.5">
                {rc.drivers.map((d, j) => (
                  <div key={j} className="flex items-start gap-3 rounded-lg bg-panel2 px-3 py-2 text-xs">
                    <span className={`mt-0.5 font-bold ${d.direction === 'higher' ? 'text-rose' : 'text-teal'}`}>
                      {d.impact}
                    </span>
                    <div>
                      <div className="font-semibold text-ink">{d.factor}</div>
                      <div className="text-mute">{d.detail}</div>
                    </div>
                  </div>
                ))}
              </div>
              {rc.recommendations.length > 0 && (
                <ul className="mt-3 space-y-1 text-xs text-mute">
                  {rc.recommendations.map((r, j) => (
                    <li key={j}>→ {r}</li>
                  ))}
                </ul>
              )}
            </Panel>
          ))}

          {report.cohorts.map((c, i) => (
            <Panel key={i} title={`Cohorts — ${c.metric_col} by ${c.cohort_col}`}>
              <p className="mb-3 text-xs text-mute">{c.interpretation}</p>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="text-mute">
                      <th className="px-2 py-1.5">Cohort</th>
                      <th className="px-2 py-1.5">n</th>
                      <th className="px-2 py-1.5">Mean</th>
                      <th className="px-2 py-1.5">vs avg</th>
                    </tr>
                  </thead>
                  <tbody>
                    {c.cohorts.map((row) => (
                      <tr key={row.name} className="border-t border-edge/60">
                        <td className="px-2 py-1.5 font-semibold text-ink">
                          {row.name}
                          {row.name === c.best_cohort && <span className="ml-1 text-teal">★</span>}
                          {row.name === c.worst_cohort && <span className="ml-1 text-rose">▼</span>}
                        </td>
                        <td className="px-2 py-1.5 text-mute">{row.n.toLocaleString()}</td>
                        <td className="px-2 py-1.5 text-mute">
                          {row.mean.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                        </td>
                        <td className={`px-2 py-1.5 ${row.vs_avg_pct >= 0 ? 'text-teal' : 'text-rose'}`}>
                          {row.vs_avg_pct >= 0 ? '+' : ''}
                          {row.vs_avg_pct.toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          ))}

          {report.pareto.map((p, i) => (
            <Panel key={i} title={`Pareto — ${p.value_col} by ${p.group_col}`}>
              <p className="mb-3 text-xs text-mute">
                Top 20% of groups drive <b className="text-ink">{p.top_groups_share.toFixed(0)}%</b> of total.{' '}
                {p.interpretation}
              </p>
              <div className="space-y-1">
                {p.groups.slice(0, 10).map((g) => (
                  <div key={g.name} className="flex items-center gap-2 text-xs">
                    <span className="w-32 truncate text-mute">{g.name}</span>
                    <div className="h-2 flex-1 overflow-hidden rounded-full bg-panel2">
                      <div
                        className={`h-full ${g.in_top_20 ? 'bg-accent' : 'bg-edge'}`}
                        style={{ width: `${Math.min(g.cumulative_pct, 100)}%` }}
                      />
                    </div>
                    <span className="w-12 text-right text-mute">
                      {g.cumulative_pct.toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </Panel>
          ))}

          {report.segments.length > 0 && (
            <Panel title="Segment health">
              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                {report.segments.map((s) => (
                  <div key={s.segment_name} className="rounded-lg bg-panel2 p-3 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-ink">{s.segment_name}</span>
                      <span
                        className={`font-bold ${
                          s.health_score >= 70 ? 'text-teal'
                            : s.health_score >= 45 ? 'text-amber' : 'text-rose'
                        }`}
                      >
                        {Math.round(s.health_score)}
                      </span>
                    </div>
                    <div className="mt-1 text-mute">n = {s.n.toLocaleString()}</div>
                    {s.strengths.length > 0 && (
                      <div className="mt-1.5 text-teal">✓ {s.strengths.join(' · ')}</div>
                    )}
                    {s.weaknesses.length > 0 && (
                      <div className="mt-0.5 text-rose">✗ {s.weaknesses.join(' · ')}</div>
                    )}
                    {s.opportunity && (
                      <div className="mt-1.5 text-mute">
                        <span className="text-[10px] font-semibold tracking-wide text-mute/80 uppercase">
                          Opportunity:
                        </span>{' '}
                        {s.opportunity}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </Panel>
          )}
        </div>
      )}
    </div>
  )
}
