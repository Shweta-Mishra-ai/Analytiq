import { useCallback, useEffect, useState } from 'react'
import { Wrench, Undo2 } from 'lucide-react'
import { apiGet, apiPost, type TableData } from '../api/client'
import { useApp } from '../store/app'
import DataTable from '../components/DataTable'
import { Btn, ErrorBox, NeedData, PageHeader, Panel, Spinner } from '../components/Ui'

interface ColumnProfile {
  name: string
  inferred_type: string
  missing_pct: number
  unique_count: number
  quality_score: number
  outlier_count?: number
  quality_issues: string[]
}

interface Profile {
  overall_quality_score: number
  rows: number
  cols: number
  duplicate_rows: number
  data_quality_grade: string
  column_profiles: ColumnProfile[]
  recommendations?: string[]
}

interface CleanResult {
  summary: Record<string, unknown>
  actions: {
    column: string
    issue: string
    action: string
    rows_affected: number
  }[]
  preview: TableData
}

function scoreColor(v: number) {
  if (v >= 80) return 'text-teal'
  if (v >= 60) return 'text-amber'
  return 'text-rose'
}

export default function QualityPage() {
  const { dataset, setDataset } = useApp()
  const [profile, setProfile] = useState<Profile | null>(null)
  const [clean, setClean] = useState<CleanResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const ds = dataset?.dataset_id

  const loadProfile = useCallback(() => {
    if (!ds) return
    setProfile(null)
    apiGet<Profile>(`/api/datasets/${ds}/profile`)
      .then(setProfile)
      .catch((e) => setError(e.message))
  }, [ds])
  useEffect(loadProfile, [loadProfile])

  if (!dataset) return <NeedData />

  const runClean = async () => {
    setBusy(true)
    setError('')
    try {
      const r = await apiPost<CleanResult>(`/api/datasets/${ds}/clean`)
      setClean(r)
      setDataset({
        ...dataset,
        rows: r.preview.total_rows,
        cols: r.preview.columns.length,
      })
      loadProfile()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const undo = async () => {
    setBusy(true)
    try {
      await apiPost(`/api/datasets/${ds}/reset`)
      setClean(null)
      loadProfile()
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="p-8">
      <PageHeader
        title="Data Quality"
        subtitle="Per-column scores, missing data, outliers and one-click auto-clean"
        right={
          <div className="flex gap-2">
            {clean && (
              <Btn variant="ghost" onClick={undo} disabled={busy}>
                <span className="flex items-center gap-1.5">
                  <Undo2 className="h-4 w-4" /> Undo clean
                </span>
              </Btn>
            )}
            <Btn onClick={runClean} disabled={busy}>
              <span className="flex items-center gap-1.5">
                <Wrench className="h-4 w-4" /> Auto-clean
              </span>
            </Btn>
          </div>
        }
      />
      {error && (
        <div className="mb-4">
          <ErrorBox message={error} />
        </div>
      )}
      {!profile && !error && <Spinner label="Profiling dataset…" />}

      {profile && (
        <>
          <div className="mb-5 grid grid-cols-2 gap-3 md:grid-cols-4">
            <Panel>
              <div className="text-xs text-mute uppercase">Overall quality</div>
              <div
                className={`text-3xl font-bold ${scoreColor(profile.overall_quality_score)}`}
              >
                {Math.round(profile.overall_quality_score)}
                <span className="text-base text-mute">/100</span>
              </div>
            </Panel>
            <Panel>
              <div className="text-xs text-mute uppercase">Rows</div>
              <div className="text-3xl font-bold text-ink">
                {profile.rows.toLocaleString()}
              </div>
            </Panel>
            <Panel>
              <div className="text-xs text-mute uppercase">Columns</div>
              <div className="text-3xl font-bold text-ink">
                {profile.cols}
              </div>
            </Panel>
            <Panel>
              <div className="text-xs text-mute uppercase">Duplicates</div>
              <div className="text-3xl font-bold text-ink">
                {profile.duplicate_rows.toLocaleString()}
              </div>
            </Panel>
          </div>

          <Panel title="Column health">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="text-mute">
                    <th className="px-2 py-2">Column</th>
                    <th className="px-2 py-2">Type</th>
                    <th className="px-2 py-2">Score</th>
                    <th className="px-2 py-2">Missing</th>
                    <th className="px-2 py-2">Unique</th>
                    <th className="px-2 py-2">Recommendations</th>
                  </tr>
                </thead>
                <tbody>
                  {profile.column_profiles.map((c) => (
                    <tr key={c.name} className="border-t border-edge/60">
                      <td className="px-2 py-2 font-semibold text-ink">
                        {c.name}
                      </td>
                      <td className="px-2 py-2 text-mute">{c.inferred_type}</td>
                      <td
                        className={`px-2 py-2 font-bold ${scoreColor(c.quality_score)}`}
                      >
                        {Math.round(c.quality_score)}
                      </td>
                      <td className="px-2 py-2 text-mute">
                        {c.missing_pct.toFixed(1)}%
                      </td>
                      <td className="px-2 py-2 text-mute">
                        {c.unique_count.toLocaleString()}
                      </td>
                      <td className="max-w-md px-2 py-2 text-mute">
                        {c.quality_issues?.join(' · ') || '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}

      {busy && <Spinner label="Cleaning…" />}

      {clean && (
        <div className="mt-5 space-y-4">
          <Panel title={`Cleaning actions (${clean.actions.length})`}>
            <div className="max-h-64 space-y-1 overflow-y-auto">
              {clean.actions.map((a, i) => (
                <div
                  key={i}
                  className="rounded-lg bg-panel2 px-3 py-2 text-xs text-mute"
                >
                  <b className="text-ink">{a.column}</b> — {a.issue} →{' '}
                  <span className="text-teal">{a.action}</span>
                  {a.rows_affected > 0 && (
                    <span className="ml-1 opacity-70">
                      ({a.rows_affected.toLocaleString()} rows)
                    </span>
                  )}
                </div>
              ))}
            </div>
          </Panel>
          <DataTable data={clean.preview} />
        </div>
      )}
    </div>
  )
}
