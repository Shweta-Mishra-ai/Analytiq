import { useEffect, useState } from 'react'
import {
  Download,
  FileSpreadsheet,
  FileText,
  HeartPulse,
  Loader2,
} from 'lucide-react'
import { apiBlob, apiGet, downloadBlob } from '../api/client'
import { useApp } from '../store/app'
import { Btn, ErrorBox, NeedData, PageHeader, Panel } from '../components/Ui'

interface HealthInsight {
  tag: string
  title: string
  body: string
  action: string
  severity: string
}

interface HealthSummary {
  domain: string
  health: {
    score: number
    grade: string
    label: string
    color: string
    missing_pct: number
    dup_pct: number
    outlier_pct: number
    rows: number
    cols: number
  }
  insights: HealthInsight[]
}

const SEVERITY_TONE: Record<string, string> = {
  critical: 'text-rose',
  warning: 'text-amber',
  positive: 'text-teal',
  info: 'text-accent',
}

export default function ReportsPage() {
  const dataset = useApp((s) => s.dataset)
  const [title, setTitle] = useState('Data Analysis Report')
  const [clientName, setClientName] = useState('Client')
  const [subtitle, setSubtitle] = useState('')
  const [includeStats, setIncludeStats] = useState(true)
  const [includeBi, setIncludeBi] = useState(true)
  const [includeMl, setIncludeMl] = useState(false)
  const [confidential, setConfidential] = useState(false)
  const [agencyName, setAgencyName] = useState('Analytiq')
  const [health, setHealth] = useState<HealthSummary | null>(null)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const ds = dataset?.dataset_id

  useEffect(() => {
    if (!ds) return
    setHealth(null)
    apiGet<HealthSummary>(`/api/reports/${ds}/health`)
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [ds])

  if (!dataset) return <NeedData />

  const download = async (kind: 'pdf' | 'health-pdf' | 'csv' | 'excel') => {
    setBusy(kind)
    setError('')
    try {
      if (kind === 'pdf') {
        const blob = await apiBlob(`/api/reports/${ds}/pdf`, 'POST', {
          title,
          subtitle,
          client_name: clientName,
          confidential,
          include_stats: includeStats,
          include_bi: includeBi,
          include_ml: includeMl,
        })
        downloadBlob(blob, 'analytiq_report.pdf')
      } else if (kind === 'health-pdf') {
        const blob = await apiBlob(`/api/reports/${ds}/health-pdf`, 'POST', {
          agency_name: agencyName,
        })
        downloadBlob(blob, 'analytiq_health_report.pdf')
      } else {
        const blob = await apiBlob(`/api/reports/${ds}/${kind}`)
        downloadBlob(blob, kind === 'csv' ? 'analytiq_cleaned_data.csv' : 'analytiq_export.xlsx')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy('')
    }
  }

  const input =
    'w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink placeholder:text-mute focus:border-accent focus:outline-none'

  return (
    <div className="p-8">
      <PageHeader
        title="Reports"
        subtitle="Senior-analyst PDF with AI narratives, plus clean data exports"
      />
      {error && (
        <div className="mb-4">
          <ErrorBox message={error} />
        </div>
      )}

      <div className="grid gap-5 lg:grid-cols-2">
        <Panel title="PDF report">
          <div className="space-y-3">
            <div>
              <label className="mb-1 block text-xs text-mute">Report title</label>
              <input value={title} onChange={(e) => setTitle(e.target.value)} className={input} />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="mb-1 block text-xs text-mute">Client name</label>
                <input value={clientName} onChange={(e) => setClientName(e.target.value)} className={input} />
              </div>
              <div>
                <label className="mb-1 block text-xs text-mute">Subtitle</label>
                <input value={subtitle} onChange={(e) => setSubtitle(e.target.value)} placeholder="optional" className={input} />
              </div>
            </div>
            <div className="flex flex-wrap gap-4 pt-1 text-sm text-mute">
              {(
                [
                  ['Statistical analysis', includeStats, setIncludeStats],
                  ['Business intelligence', includeBi, setIncludeBi],
                  ['ML results (if trained)', includeMl, setIncludeMl],
                  ['Confidential banner', confidential, setConfidential],
                ] as const
              ).map(([label, val, set]) => (
                <label key={label} className="flex cursor-pointer items-center gap-2">
                  <input
                    type="checkbox"
                    checked={val}
                    onChange={(e) => set(e.target.checked)}
                    className="accent-[#4f8ef7]"
                  />
                  {label}
                </label>
              ))}
            </div>
            <Btn onClick={() => download('pdf')} disabled={!!busy} className="w-full">
              <span className="flex items-center justify-center gap-2">
                {busy === 'pdf' ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <FileText className="h-4 w-4" />
                )}
                {busy === 'pdf' ? 'Building report… (30–60s)' : 'Generate PDF report'}
              </span>
            </Btn>
            <p className="text-xs text-mute">
              Cover + TOC · executive summary · structured insight cards · industry
              benchmarks · statistics · BI · charts with AI narratives · recommendations.
            </p>
          </div>
        </Panel>

        <Panel title="Health report">
          <div className="space-y-3">
            {health ? (
              <>
                <div className="flex items-center gap-4 rounded-xl border border-edge bg-panel2 px-4 py-3">
                  <div
                    className="text-3xl font-bold"
                    style={{ color: health.health.color }}
                  >
                    {health.health.score}
                    <span className="text-base text-mute">/100</span>
                  </div>
                  <div className="min-w-0">
                    <div
                      className="text-sm font-semibold"
                      style={{ color: health.health.color }}
                    >
                      Grade {health.health.grade} — {health.health.label}
                    </div>
                    <div className="text-xs text-mute">
                      {health.health.missing_pct}% missing ·{' '}
                      {health.health.dup_pct}% duplicates ·{' '}
                      {health.health.outlier_pct}% outlier cols · domain{' '}
                      {health.domain}
                    </div>
                  </div>
                </div>

                {health.insights.length > 0 && (
                  <div className="space-y-1.5">
                    <div className="text-xs text-mute">
                      {health.insights.length} insight
                      {health.insights.length === 1 ? '' : 's'} will be included:
                    </div>
                    <ul className="max-h-40 space-y-1 overflow-y-auto pr-1">
                      {health.insights.map((ins, i) => (
                        <li key={i} className="flex gap-2 text-xs">
                          <span
                            className={`shrink-0 font-semibold ${
                              SEVERITY_TONE[ins.severity] ?? 'text-mute'
                            }`}
                          >
                            {ins.tag}
                          </span>
                          <span className="truncate text-mute">{ins.title}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            ) : (
              <p className="text-sm text-mute">Computing data health…</p>
            )}

            <div>
              <label className="mb-1 block text-xs text-mute">
                Agency name (shown on every page)
              </label>
              <input
                value={agencyName}
                onChange={(e) => setAgencyName(e.target.value)}
                className={input}
              />
            </div>

            <Btn
              onClick={() => download('health-pdf')}
              disabled={!!busy}
              className="w-full"
            >
              <span className="flex items-center justify-center gap-2">
                {busy === 'health-pdf' ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <HeartPulse className="h-4 w-4" />
                )}
                {busy === 'health-pdf'
                  ? 'Building health report…'
                  : 'Generate health report'}
              </span>
            </Btn>
            <p className="text-xs text-mute">
              Client-ready data-health scorecard: quality grade, dataset
              summary, column-level breakdown, and business insight cards in a
              What → Why → What&nbsp;to&nbsp;do format.
            </p>
          </div>
        </Panel>

        <Panel title="Data exports">
          <div className="space-y-3">
            <button
              onClick={() => download('csv')}
              disabled={!!busy}
              className="flex w-full items-center gap-3 rounded-xl border border-edge bg-panel2 px-4 py-3 text-left text-sm transition hover:border-accent/50"
            >
              <Download className="h-5 w-5 text-accent" />
              <div>
                <div className="font-semibold text-ink">Cleaned CSV</div>
                <div className="text-xs text-mute">Current active dataset as UTF-8 CSV</div>
              </div>
              {busy === 'csv' && <Loader2 className="ml-auto h-4 w-4 animate-spin text-mute" />}
            </button>
            <button
              onClick={() => download('excel')}
              disabled={!!busy}
              className="flex w-full items-center gap-3 rounded-xl border border-edge bg-panel2 px-4 py-3 text-left text-sm transition hover:border-accent/50"
            >
              <FileSpreadsheet className="h-5 w-5 text-teal" />
              <div>
                <div className="font-semibold text-ink">Excel workbook</div>
                <div className="text-xs text-mute">Data sheet + numeric summary sheet</div>
              </div>
              {busy === 'excel' && <Loader2 className="ml-auto h-4 w-4 animate-spin text-mute" />}
            </button>
          </div>
        </Panel>
      </div>
    </div>
  )
}
