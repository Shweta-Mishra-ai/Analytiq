import { useCallback, useEffect, useState } from 'react'
import { Wrench, Undo2, Download } from 'lucide-react'
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

interface ReadinessIssue {
  column: string
  issue: string
  consequence: string
  fix: string
}

interface Readiness {
  ready: boolean
  rows: number
  columns: number
  observed_pct: number
  summary: string
  blockers: ReadinessIssue[]
  advisories: ReadinessIssue[]
  personal_data_columns: string[]
}

interface CleanAction {
  column: string
  issue: string
  action: string
  rows_affected: number
  /** The same step expressed against the source table, for the client's
   *  data team to audit or apply upstream. Never executed by the app. */
  sql: string
}

interface CleanResult {
  summary: Record<string, unknown>
  actions: CleanAction[]
  sql: string
  sql_table: string
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
  const [readiness, setReadiness] = useState<Readiness | null>(null)
  const [clean, setClean] = useState<CleanResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [openSql, setOpenSql] = useState<number | null>(null)
  const [copied, setCopied] = useState(false)
  const ds = dataset?.dataset_id

  const copySql = async () => {
    if (!clean?.sql) return
    try {
      await navigator.clipboard.writeText(clean.sql)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      setError('Clipboard unavailable — use the .sql download instead')
    }
  }

  const downloadSql = () => {
    if (!clean?.sql) return
    const url = URL.createObjectURL(
      new Blob([clean.sql], { type: 'text/plain' }),
    )
    const a = document.createElement('a')
    a.href = url
    a.download = `${clean.sql_table}_cleaning.sql`
    a.click()
    URL.revokeObjectURL(url)
  }

  const loadProfile = useCallback(() => {
    if (!ds) return
    setProfile(null)
    apiGet<Profile>(`/api/datasets/${ds}/profile`)
      .then(setProfile)
      .catch((e) => setError(e.message))
    apiGet<Readiness>(`/api/datasets/${ds}/readiness`)
      .then(setReadiness)
      .catch(() => setReadiness(null))
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

      {/* The gate. Analysis on a dataset with a blocker is arithmetically
          correct and describes something other than the client's business,
          so this sits above the scores rather than below them. */}
      {readiness && (
        <Panel
          className={`mb-5 ${readiness.ready ? '' : 'border-amber/40'}`}
          title={
            readiness.ready
              ? 'Ready for analysis'
              : `Not ready for analysis — ${readiness.blockers.length} issue(s) to resolve first`
          }
        >
          <p className="text-xs text-mute">{readiness.summary}</p>
          {readiness.blockers.length > 0 && (
            <div className="mt-3 space-y-2">
              {readiness.blockers.map((b, i) => (
                <div key={i} className="rounded-lg bg-panel2 px-3 py-2 text-xs">
                  <div className="font-semibold text-ink">
                    {b.column} — {b.issue}
                  </div>
                  <div className="mt-0.5 text-mute">{b.consequence}</div>
                  <div className="mt-1 text-teal">Fix: {b.fix}</div>
                </div>
              ))}
            </div>
          )}
          {readiness.advisories.length > 0 && (
            <details className="mt-3">
              <summary className="cursor-pointer text-xs text-mute">
                {readiness.advisories.length} advisory note(s) — worth tidying,
                the analysis is still valid
              </summary>
              <div className="mt-2 space-y-1">
                {readiness.advisories.map((a, i) => (
                  <div key={i} className="text-xs text-mute">
                    <b className="text-ink">{a.column}</b> — {a.issue}. {a.fix}
                  </div>
                ))}
              </div>
            </details>
          )}
          {readiness.personal_data_columns.length > 0 && (
            <div className="mt-3 rounded-lg border border-edge px-3 py-2 text-xs text-amber">
              Personal data in {readiness.personal_data_columns.join(', ')} —
              confirm it may be processed, and keep it out of anything shared.
            </div>
          )}
        </Panel>
      )}

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
            <div className="max-h-80 space-y-1 overflow-y-auto">
              {clean.actions.map((a, i) => (
                <div key={i} className="rounded-lg bg-panel2 px-3 py-2 text-xs">
                  <div className="flex items-start justify-between gap-3">
                    <div className="text-mute">
                      <b className="text-ink">{a.column}</b> — {a.issue} →{' '}
                      <span className="text-teal">{a.action}</span>
                      {a.rows_affected > 0 && (
                        <span className="ml-1 opacity-70">
                          ({a.rows_affected.toLocaleString()} rows)
                        </span>
                      )}
                    </div>
                    {a.sql && (
                      <button
                        onClick={() => setOpenSql(openSql === i ? null : i)}
                        className="shrink-0 rounded-md border border-edge px-2 py-0.5 font-mono text-[10px] uppercase tracking-wide text-mute transition hover:border-teal hover:text-teal"
                      >
                        {openSql === i ? 'hide' : 'sql'}
                      </button>
                    )}
                  </div>
                  {openSql === i && a.sql && (
                    <pre className="mt-2 overflow-x-auto rounded-md border border-edge bg-bg px-3 py-2 font-mono text-[11px] leading-relaxed text-ink/90">
                      {a.sql.replace(/\{table\}/g, `"${clean.sql_table}"`)}
                    </pre>
                  )}
                </div>
              ))}
            </div>
          </Panel>

          <Panel
            title="Equivalent SQL"
            right={
              <div className="flex items-center gap-2">
                <Btn variant="ghost" size="sm" onClick={copySql}>
                  {copied ? 'Copied' : 'Copy'}
                </Btn>
                <Btn variant="ghost" size="sm" onClick={downloadSql}>
                  <span className="flex items-center gap-1.5">
                    <Download className="h-3.5 w-3.5" /> .sql
                  </span>
                </Btn>
              </div>
            }
          >
            <p className="mb-3 text-xs text-mute">
              Every step above, written against{' '}
              <code className="text-ink">{clean.sql_table}</code> in execution
              order. The analysis runs in pandas — this script is here so the
              cleaning can be audited and, if you want it, applied upstream in
              the warehouse instead. Nothing here has been executed.
            </p>
            <pre className="max-h-96 overflow-auto rounded-lg border border-edge bg-bg px-4 py-3 font-mono text-[11px] leading-relaxed text-ink/90">
              {clean.sql}
            </pre>
          </Panel>

          <DataTable data={clean.preview} />
        </div>
      )}
    </div>
  )
}
