import { useEffect, useState } from 'react'
import { apiGet } from '../api/client'
import { useApp } from '../store/app'
import { ErrorBox, NeedData, PageHeader, Panel, Spinner } from '../components/Ui'

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
  post_hoc: string[]
}

interface Vif {
  feature: string
  vif: number
  verdict: string
  interpretation: string
}

interface TimeSeries {
  column: string
  date_col: string
  is_stationary?: boolean
  trend?: string
  seasonality?: string
  interpretation: string
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
  warnings: string[]
}

const sig = (p: number) => (p < 0.001 ? 'p < 0.001' : `p = ${p.toFixed(3)}`)

export default function EdaPage() {
  const dataset = useApp((s) => s.dataset)
  const [report, setReport] = useState<EdaReport | null>(null)
  const [error, setError] = useState('')
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
      <PageHeader
        title="Deep EDA"
        subtitle="Statistical tests, distributions, VIF, group comparisons and time series"
      />
      {error && <ErrorBox message={error} />}
      {!report && !error && (
        <Spinner label="Running statistical analysis… (first run can take a moment)" />
      )}

      {report && (
        <div className="space-y-5">
          {report.key_findings.length > 0 && (
            <Panel title="Key findings">
              <ul className="space-y-1.5 text-sm text-mute">
                {report.key_findings.map((f, i) => (
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
                      <td className="px-2 py-2 font-semibold text-ink">{u.column}</td>
                      <td className="px-2 py-2">{u.mean?.toLocaleString(undefined, { maximumFractionDigits: 2 }) ?? '—'}</td>
                      <td className="px-2 py-2">{u.median?.toLocaleString(undefined, { maximumFractionDigits: 2 }) ?? '—'}</td>
                      <td className="px-2 py-2">{u.std?.toLocaleString(undefined, { maximumFractionDigits: 2 }) ?? '—'}</td>
                      <td className="px-2 py-2">{u.skew_label ?? '—'}</td>
                      <td className="px-2 py-2">{u.normality_verdict ?? '—'}</td>
                      <td className="px-2 py-2">
                        {u.outliers_iqr > 0 ? (
                          <span className="text-amber">{u.outliers_iqr} ({u.outlier_pct.toFixed(1)}%)</span>
                        ) : ('0')}
                      </td>
                      <td className="px-2 py-2">{u.best_fit_dist ?? '—'}</td>
                      <td className="max-w-xs px-2 py-2">{u.interpretation}</td>
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
                      {c.col_a} ↔ {c.col_b}
                      <span className="ml-2 font-normal text-mute">
                        {c.test_name} · r = {c.statistic.toFixed(3)} · {sig(c.p_value)}
                        {c.effect_label && ` · ${c.effect_label} effect`}
                      </span>
                    </div>
                    <div className="mt-0.5 text-mute">{c.interpretation}</div>
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
                      {g.numeric_col} across {g.group_col}
                      <span className="ml-2 font-normal text-mute">
                        {g.test_used} · {sig(g.p_value)} ·{' '}
                        {g.is_significant ? (
                          <span className="text-teal">significant</span>
                        ) : ('not significant')}
                      </span>
                    </div>
                    <div className="mt-0.5 text-mute">{g.interpretation}</div>
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
                    <div key={v.feature} className="flex items-center justify-between rounded-lg bg-panel2 px-3 py-2">
                      <span className="font-semibold text-ink">{v.feature}</span>
                      <span className={
                        v.verdict === 'OK' ? 'text-teal'
                          : v.verdict === 'Moderate' ? 'text-amber' : 'text-rose'
                      }>
                        VIF {v.vif.toFixed(1)} — {v.verdict}
                      </span>
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
                        {t.column} <span className="font-normal text-mute">over {t.date_col}</span>
                      </div>
                      <div className="mt-0.5 text-mute">
                        Trend: {t.trend ?? '—'} · Stationary: {t.is_stationary === null ? '—' : t.is_stationary ? 'yes' : 'no'}
                        {t.seasonality && ` · ${t.seasonality}`}
                      </div>
                      <div className="mt-0.5 text-mute">{t.interpretation}</div>
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
