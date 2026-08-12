import { useEffect, useState } from 'react'
import { GitCompare } from 'lucide-react'
import { apiGet, apiPost, type DatasetMeta } from '../api/client'
import { useApp } from '../store/app'
import { Btn, ErrorBox, NeedData, PageHeader, Panel, Spinner } from '../components/Ui'

interface SchemaDiff {
  added_columns: string[]
  removed_columns: string[]
  common_columns: string[]
  dtype_changes: Record<string, string>
}

interface ColumnComparison {
  column: string
  dtype: string
  mean_a: number | null
  mean_b: number | null
  median_a: number | null
  median_b: number | null
  pct_change_mean: number | null
  missing_pct_a: number
  missing_pct_b: number
  is_significant: boolean | null
  p_value: number | null
  test_used: string | null
  verdict: string
  top_category_a: string | null
  top_category_b: string | null
  category_shift_detected: boolean | null
}

interface ComparisonReport {
  label_a: string
  label_b: string
  n_rows_a: number
  n_rows_b: number
  row_delta_pct: number
  dup_pct_a: number
  dup_pct_b: number
  schema_diff: SchemaDiff
  column_comparisons: ColumnComparison[]
  most_changed: ColumnComparison[]
  significant_changes: ColumnComparison[]
  summary: string
  warnings: string[]
}

const pct = (v: number | null) =>
  v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`

export default function ComparePage() {
  const dataset = useApp((s) => s.dataset)
  const ds = dataset?.dataset_id

  const [others, setOthers] = useState<DatasetMeta[]>([])
  const [otherId, setOtherId] = useState('')
  const [labelA, setLabelA] = useState('Current')
  const [labelB, setLabelB] = useState('Comparison')
  const [report, setReport] = useState<ComparisonReport | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!ds) return
    setReport(null)
    setError('')
    apiGet<{ datasets: DatasetMeta[] }>('/api/datasets')
      .then((r) => {
        const rest = r.datasets.filter((d) => d.dataset_id !== ds)
        setOthers(rest)
        setOtherId(rest[0]?.dataset_id ?? '')
      })
      .catch((e) => setError(e.message))
  }, [ds])

  if (!dataset) return <NeedData />

  const run = async () => {
    setBusy(true)
    setError('')
    try {
      setReport(
        await apiPost<ComparisonReport>(`/api/analytics/${ds}/compare`, {
          other_dataset_id: otherId,
          label_a: labelA,
          label_b: labelB,
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
        title="Compare Datasets"
        subtitle="This period vs last — what actually moved, and whether it's real"
      />

      {error && (
        <div className="mb-4">
          <ErrorBox message={error} />
        </div>
      )}

      <Panel title="Pick what to compare against">
        {others.length === 0 ? (
          <p className="text-sm text-mute">
            You only have one dataset uploaded. Upload a second file — last
            quarter&apos;s export, for example — and it will appear here.
          </p>
        ) : (
          <div className="flex flex-wrap items-end gap-3">
            <div className="min-w-48">
              <label className="mb-1 block text-[11px] text-mute uppercase">
                Current ({dataset.filename})
              </label>
              <input
                value={labelA}
                onChange={(e) => setLabelA(e.target.value)}
                className="w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink"
              />
            </div>
            <div className="min-w-56">
              <label className="mb-1 block text-[11px] text-mute uppercase">
                Compare against
              </label>
              <select
                value={otherId}
                onChange={(e) => setOtherId(e.target.value)}
                className="w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink"
              >
                {others.map((d) => (
                  <option key={d.dataset_id} value={d.dataset_id}>
                    {d.filename} ({d.rows.toLocaleString()} rows)
                  </option>
                ))}
              </select>
            </div>
            <div className="min-w-40">
              <label className="mb-1 block text-[11px] text-mute uppercase">
                Label
              </label>
              <input
                value={labelB}
                onChange={(e) => setLabelB(e.target.value)}
                className="w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink"
              />
            </div>
            <Btn onClick={run} disabled={busy || !otherId}>
              <span className="flex items-center gap-1.5">
                <GitCompare className="h-4 w-4" /> Compare
              </span>
            </Btn>
          </div>
        )}
      </Panel>

      {busy && <Spinner label="Diffing the two datasets…" />}

      {report && !busy && (
        <div className="mt-5 space-y-5">
          <Panel>
            <p className="text-sm text-ink">{report.summary}</p>
          </Panel>

          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat
              label={`${report.label_a} rows`}
              value={report.n_rows_a.toLocaleString()}
            />
            <Stat
              label={`${report.label_b} rows`}
              value={report.n_rows_b.toLocaleString()}
            />
            <Stat
              label="Row change"
              value={pct(report.row_delta_pct)}
              tone={report.row_delta_pct >= 0 ? 'teal' : 'rose'}
            />
            <Stat
              label="Significant shifts"
              value={String(report.significant_changes.length)}
              tone={report.significant_changes.length > 0 ? 'amber' : 'teal'}
            />
          </div>

          {(report.schema_diff.added_columns.length > 0 ||
            report.schema_diff.removed_columns.length > 0 ||
            Object.keys(report.schema_diff.dtype_changes).length > 0) && (
            <Panel title="Schema changes">
              <div className="space-y-2 text-xs">
                {report.schema_diff.added_columns.length > 0 && (
                  <div>
                    <span className="font-semibold text-teal">Added: </span>
                    <span className="text-mute">
                      {report.schema_diff.added_columns.join(', ')}
                    </span>
                  </div>
                )}
                {report.schema_diff.removed_columns.length > 0 && (
                  <div>
                    <span className="font-semibold text-rose">Removed: </span>
                    <span className="text-mute">
                      {report.schema_diff.removed_columns.join(', ')}
                    </span>
                  </div>
                )}
                {Object.entries(report.schema_diff.dtype_changes).map(
                  ([col, change]) => (
                    <div key={col}>
                      <span className="font-semibold text-amber">
                        Type changed:{' '}
                      </span>
                      <span className="text-mute">
                        {col} — {change}
                      </span>
                    </div>
                  ),
                )}
              </div>
            </Panel>
          )}

          {report.most_changed.length > 0 && (
            <Panel title="Biggest movers">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="text-mute">
                      {['Column', report.label_a, report.label_b, 'Change',
                        'Significant?', 'Verdict'].map((h) => (
                        <th key={h} className="px-2 py-2 whitespace-nowrap">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {report.most_changed.slice(0, 12).map((c) => (
                      <tr key={c.column} className="border-t border-edge/60 align-top">
                        <td className="px-2 py-2 font-semibold text-ink">
                          {c.column}
                        </td>
                        <td className="px-2 py-2 text-mute">
                          {c.mean_a != null
                            ? c.mean_a.toLocaleString(undefined, {
                                maximumFractionDigits: 2,
                              })
                            : (c.top_category_a ?? '—')}
                        </td>
                        <td className="px-2 py-2 text-mute">
                          {c.mean_b != null
                            ? c.mean_b.toLocaleString(undefined, {
                                maximumFractionDigits: 2,
                              })
                            : (c.top_category_b ?? '—')}
                        </td>
                        <td
                          className={`px-2 py-2 font-semibold ${
                            c.pct_change_mean == null
                              ? 'text-mute'
                              : c.pct_change_mean >= 0
                                ? 'text-teal'
                                : 'text-rose'
                          }`}
                        >
                          {pct(c.pct_change_mean)}
                        </td>
                        <td className="px-2 py-2">
                          {c.is_significant == null ? (
                            <span className="text-mute">—</span>
                          ) : c.is_significant ? (
                            <span className="text-teal">
                              yes{c.p_value != null && ` (p=${c.p_value.toFixed(3)})`}
                            </span>
                          ) : (
                            <span className="text-mute">no</span>
                          )}
                        </td>
                        <td className="px-2 py-2 text-mute">{c.verdict}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
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

function Stat({
  label,
  value,
  tone = 'ink',
}: {
  label: string
  value: string
  tone?: 'ink' | 'teal' | 'amber' | 'rose'
}) {
  const tones = {
    ink: 'text-ink',
    teal: 'text-teal',
    amber: 'text-amber',
    rose: 'text-rose',
  }
  return (
    <Panel>
      <div className="text-xs text-mute uppercase">{label}</div>
      <div className={`text-xl font-bold ${tones[tone]}`}>{value}</div>
    </Panel>
  )
}
