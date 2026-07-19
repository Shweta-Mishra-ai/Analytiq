import { useState } from 'react'
import { Download, FileSpreadsheet, FileText, Loader2 } from 'lucide-react'
import { apiBlob, downloadBlob } from '../api/client'
import { useApp } from '../store/app'
import { Btn, ErrorBox, NeedData, PageHeader, Panel } from '../components/Ui'

export default function ReportsPage() {
  const dataset = useApp((s) => s.dataset)
  const [title, setTitle] = useState('Data Analysis Report')
  const [clientName, setClientName] = useState('Client')
  const [subtitle, setSubtitle] = useState('')
  const [includeStats, setIncludeStats] = useState(true)
  const [includeBi, setIncludeBi] = useState(true)
  const [includeMl, setIncludeMl] = useState(false)
  const [confidential, setConfidential] = useState(false)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const ds = dataset?.dataset_id

  if (!dataset) return <NeedData />

  const download = async (kind: 'pdf' | 'csv' | 'excel') => {
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
