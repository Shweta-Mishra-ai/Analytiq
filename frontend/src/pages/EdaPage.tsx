import { useEffect, useState } from 'react'
import { apiGet } from '../api/client'
import { useApp } from '../store/app'
import { ErrorBox, NeedData, PageHeader, Panel, Spinner } from '../components/Ui'
import * as fmt from '../lib/format'

interface Univariate {
  column: string
  n: number
  missing_pct: number
  mean?: number
  median?: number
  std?: number
  skewness?: number
  skew_label?: string
  normality_verdict?: string
  outliers_iqr: number
  outlier_pct: number
  best_fit_dist?: string
  unique_count: number
  top_value?: string
  interpretation: string
  plain?: string
}

interface Bivariate {
  col_a: string
  col_b: string
  test_name: string
  statistic: number
  p_value: number
  is_significant: boolean
  effect_label?: string
  interpretation: string
  plain?: string
}

interface GroupComparison {
  numeric_col: string
  group_col: string
  n_groups: number
  test_used: string
  p_value: number
  is_significant: boolean
  effect_label?: string
  interpretation: string
  plain?: string
  post_hoc: string[]
}

interface Vif {
  feature: string
  vif: number
  verdict: string
  interpretation: string
  plain?: string
}

interface TimeSeries {
  column: string
  date_col: string
  is_stationary?: boolean
  trend?: string
  seasonality?: string
  interpretation: string
  plain?: string
}

interface EdaReport {
  n_rows: number
  n_cols: number
  univariate: Record<string, Univariate>
  correlations: Bivariate[]
  group_comparisons: GroupComparison[]
  multicollinearity: Vif[]
  time_series: TimeSeries[]
  key_findings: string[]
  plain_findings: string[]
  warnings: string[]
}

const sig = (p: number) => (p < 0.001 ? 'p < 0.001' : `p = ${p.toFixed(3)}`)

export default function EdaPage() {
  const dataset = useApp((s) => s.dataset)
  const [report, setReport] = useState<EdaReport | null>(null)
  const [error, setError] = useState('')
  // Deep EDA is a technical page and stays one — every statistic it
  // computed is still on it. What changed is that the headline findings
  // can be read without knowing what a variance inflation factor is,
  // which is the difference between a page a director can use and a page
  // they forward to someone else.
  const [plainWording, setPlainWording] = useState(true)
  const ds = dataset?.dataset_id

  useEffect(() => {
    if (!ds) return
    setReport(null)
    setError('')
    apiGet<EdaReport>(`/api/analytics/${ds}/eda`)
      .then(setReport)
      .catch((e) => setError(e.message))
  }, [ds])

  if (!dataset) return <NeedData />

  return (
    <div className="p-8">
      {/* The first line a reader sees. "Statistical tests, distributions,
          VIF, group comparisons and time series" describes the page to
          someone who already knows what is on it; the plain subtitle
          describes what they can learn from it. The analysis behind both
          is identical. */}
      <PageHeader
        title="Deep EDA"
        subtitle={
          plainWording
            ? 'How each column behaves, which ones move together, and what that means for the numbers you quote'
            : 'Statistical tests, distributions, VIF, group comparisons and time series'
        }
      />
      {error && <ErrorBox message={error} />}
      {!report && !error && (
        <Spinner label="Running statistical analysis… (first run can take a moment)" />
      )}

      {report && (
        <div className="space-y-5">
          {(report.plain_findings?.length || report.key_findings.length) > 0 && (
            <Panel
              title="Key findings"
              subtitle={
                plainWording
                  ? 'What the analysis found, and what follows from it'
                  : 'The tests behind those findings, with their statistics'
              }
              right={
                <div className="flex rounded-md border border-edge text-[11px]">
                  {[
                    ['In plain terms', true],
                    ['Statistical', false],
                  ].map(([label, plain]) => (
                    <button
                      key={String(label)}
                      type="button"
                      onClick={() => setPlainWording(plain as boolean)}
                      className={`px-2.5 py-1 ${
                        plainWording === plain
                          ? 'bg-accent/15 text-accent'
                          : 'text-mute hover:text-ink'
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              }
            >
              <ul className="space-y-2 text-sm text-mute">
                {(plainWording && report.plain_findings?.length
                  ? report.plain_findings
                  : report.key_findings
                ).map((f, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="text-accent">▸</span> {f}
                  </li>
                ))}
              </ul>
            </Panel>
          )}

          <Panel title="Univariate analysis">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="text-mute">
                    {['Column', 'Mean', 'Median', 'Std', 'Skew', 'Normality', 'Outliers', 'Best fit', 'Interpretation'].map(
                      (h) => (
                        <th key={h} className="px-2 py-2 whitespace-nowrap">{h}</th>
                      ),
                    )}
                  </tr>
                </thead>
                <tbody>
                  {Object.values(report.univariate).map((u) => (
                    <tr key={u.column} className="border-t border-edge/60 text-mute">
                      <td className="px-2 py-2 font-semibold text-ink" title={u.column}>
                        {fmt.label(u.column)}
                      </td>
                      <td className="px-2 py-2 font-data text-right">{fmt.num(u.mean)}</td>
                      <td className="px-2 py-2 font-data text-right">{fmt.num(u.median)}</td>
                      <td className="px-2 py-2 font-data text-right">{fmt.num(u.std)}</td>
                      <td className="px-2 py-2">{u.skew_label ?? '—'}</td>
                      <td className="px-2 py-2">{u.normality_verdict ?? '—'}</td>
                      <td className="px-2 py-2">
                        {u.outliers_iqr > 0 ? (
                          <span className="text-amber">{u.outliers_iqr} ({u.outlier_pct.toFixed(1)}%)</span>
                        ) : ('0')}
                      </td>
                      <td className="px-2 py-2">{u.best_fit_dist ?? '—'}</td>
                      <td className="max-w-xs px-2 py-2">
                        {(plainWording && u.plain) || u.interpretation}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>

          {report.correlations.length > 0 && (
            <Panel title="Significant relationships">
              <div className="space-y-2">
                {report.correlations.map((c, i) => (
                  <div key={i} className="rounded-lg bg-panel2 px-3 py-2 text-xs">
                    <div className="font-semibold text-ink">
                      {fmt.label(c.col_a)} ↔ {fmt.label(c.col_b)}
                      <span className="ml-2 font-normal text-mute">
                        {c.test_name} · r = {c.statistic.toFixed(3)} · {sig(c.p_value)}
                        {c.effect_label && ` · ${c.effect_label} effect`}
                      </span>
                    </div>
                    <div className="mt-0.5 text-mute">
                      {(plainWording && c.plain) || c.interpretation}
                    </div>
                  </div>
                ))}
              </div>
            </Panel>
          )}

          {report.group_comparisons.length > 0 && (
            <Panel title="Group comparisons">
              <div className="space-y-2">
                {report.group_comparisons.map((g, i) => (
                  <div key={i} className="rounded-lg bg-panel2 px-3 py-2 text-xs">
                    <div className="font-semibold text-ink">
                      {fmt.label(g.numeric_col)} across {fmt.label(g.group_col)}
                      <span className="ml-2 font-normal text-mute">
                        {g.test_used} · {sig(g.p_value)} ·{' '}
                        {g.is_significant ? (
                          <span className="text-teal">significant</span>
                        ) : ('not significant')}
                      </span>
                    </div>
                    <div className="mt-0.5 text-mute">
                      {(plainWording && g.plain) || g.interpretation}
                    </div>
                    {g.post_hoc.length > 0 && (
                      <div className="mt-1 text-mute">{g.post_hoc.join(' · ')}</div>
                    )}
                  </div>
                ))}
              </div>
            </Panel>
          )}

          <div className="grid gap-5 lg:grid-cols-2">
            {report.multicollinearity.length > 0 && (
              <Panel title="Multicollinearity (VIF)">
                <div className="space-y-1.5 text-xs">
                  {report.multicollinearity.map((v) => (
                    <div key={v.feature} className="rounded-lg bg-panel2 px-3 py-2">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-ink" title={v.feature}>
                          {fmt.label(v.feature)}
                        </span>
                        <span className={
                          v.verdict === 'OK' ? 'text-teal'
                            : v.verdict === 'Moderate' ? 'text-amber' : 'text-rose'
                        }>
                          VIF {v.vif.toFixed(1)} — {v.verdict}
                        </span>
                      </div>
                      {/* "VIF 14.0 — High" was the whole of this panel. It
                          is the most jargon-dense thing on the page and it
                          carried no explanation at all. */}
                      {((plainWording && v.plain) || v.interpretation) && (
                        <p className="mt-1 text-mute">
                          {(plainWording && v.plain) || v.interpretation}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </Panel>
            )}
            {report.time_series.length > 0 && (
              <Panel title="Time series">
                <div className="space-y-2 text-xs">
                  {report.time_series.map((t, i) => (
                    <div key={i} className="rounded-lg bg-panel2 px-3 py-2">
                      <div className="font-semibold text-ink">
                        {fmt.label(t.column)}{' '}
                        <span className="font-normal text-mute">
                          over {fmt.label(t.date_col)}
                        </span>
                      </div>
                      <div className="mt-0.5 text-mute">
                        Trend: {t.trend ?? '—'} · Stationary: {t.is_stationary === null ? '—' : t.is_stationary ? 'yes' : 'no'}
                        {t.seasonality && ` · ${t.seasonality}`}
                      </div>
                      <div className="mt-0.5 text-mute">
                        {(plainWording && t.plain) || t.interpretation}
                      </div>
                    </div>
                  ))}
                </div>
              </Panel>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
