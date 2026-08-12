import { useCallback, useEffect, useState } from 'react'
import { Users } from 'lucide-react'
import { apiGet, type TableData } from '../api/client'
import { useApp } from '../store/app'
import DataTable from '../components/DataTable'
import { Btn, ErrorBox, NeedData, PageHeader, Panel, Spinner } from '../components/Ui'

interface RfmColumns {
  customer_id: string
  date_col: string
  monetary_col: string | null
  quantity_col: string | null
  price_col: string | null
}

interface SegmentSummary {
  segment: string
  n_customers: number
  pct_customers: number
  total_monetary: number
  pct_revenue: number
  avg_recency: number
  avg_frequency: number
  avg_monetary: number
  description: string
  action: string
  color: string
}

interface RfmReport {
  customer_table: TableData
  segment_summary: SegmentSummary[]
  columns_used: RfmColumns
  n_customers: number
  n_transactions: number
  total_revenue: number
  top_segment: string
  at_risk_pct: number
  at_risk_revenue: number
  champions_count: number
  warnings: string[]
}

const money = (v: number) =>
  v.toLocaleString(undefined, { maximumFractionDigits: 0 })

export default function SegmentsPage() {
  const dataset = useApp((s) => s.dataset)
  const ds = dataset?.dataset_id

  const [detected, setDetected] = useState<RfmColumns | null>(null)
  const [report, setReport] = useState<RfmReport | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [checked, setChecked] = useState(false)

  useEffect(() => {
    if (!ds) return
    setReport(null)
    setError('')
    setChecked(false)
    apiGet<{ detected: RfmColumns | null }>(`/api/analytics/${ds}/rfm/columns`)
      .then((r) => setDetected(r.detected))
      .catch((e) => setError(e.message))
      .finally(() => setChecked(true))
  }, [ds])

  const run = useCallback(async () => {
    if (!ds) return
    setBusy(true)
    setError('')
    try {
      setReport(await apiGet<RfmReport>(`/api/analytics/${ds}/rfm`))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [ds])

  if (!dataset) return <NeedData />

  const maxRevenue = Math.max(
    1,
    ...(report?.segment_summary ?? []).map((s) => s.total_monetary),
  )

  return (
    <div className="p-8">
      <PageHeader
        title="Customer Segments"
        subtitle="RFM analysis — who your best customers are, and who you're about to lose"
        right={
          <Btn onClick={run} disabled={busy || !detected}>
            <span className="flex items-center gap-1.5">
              <Users className="h-4 w-4" /> Run segmentation
            </span>
          </Btn>
        }
      />

      {error && (
        <div className="mb-4">
          <ErrorBox message={error} />
        </div>
      )}

      {checked && !detected && (
        <Panel>
          <p className="text-sm text-mute">
            This dataset doesn&apos;t look like transaction data. RFM needs a
            customer/client ID column and a transaction date column — ideally
            with an amount or revenue column too.
          </p>
        </Panel>
      )}

      {detected && !report && !busy && (
        <Panel title="Detected columns">
          <div className="flex flex-wrap gap-4 text-sm">
            <Field label="Customer" value={detected.customer_id} />
            <Field label="Date" value={detected.date_col} />
            <Field
              label="Monetary"
              value={
                detected.monetary_col ??
                (detected.price_col && detected.quantity_col
                  ? `${detected.price_col} × ${detected.quantity_col}`
                  : 'none — will use frequency')
              }
            />
          </div>
        </Panel>
      )}

      {busy && <Spinner label="Scoring customers on recency, frequency and spend…" />}

      {report && !busy && (
        <div className="mt-5 space-y-5">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
            <Stat label="Customers" value={report.n_customers.toLocaleString()} />
            <Stat label="Transactions" value={report.n_transactions.toLocaleString()} />
            <Stat label="Total value" value={money(report.total_revenue)} tone="teal" />
            <Stat
              label="Champions"
              value={report.champions_count.toLocaleString()}
              tone="teal"
            />
            <Stat
              label="At risk"
              value={`${report.at_risk_pct.toFixed(1)}%`}
              tone={report.at_risk_pct > 25 ? 'rose' : 'amber'}
            />
          </div>

          {report.at_risk_revenue > 0 && (
            <Panel>
              <p className="text-sm text-ink">
                <span className="font-semibold text-rose">
                  {money(report.at_risk_revenue)}
                </span>{' '}
                of value sits in at-risk, can&apos;t-lose-them and hibernating
                segments. That is the revenue a win-back campaign is competing
                for.
              </p>
            </Panel>
          )}

          <Panel title="Value by segment">
            <div className="space-y-2">
              {report.segment_summary.map((s) => (
                <div key={s.segment} className="flex items-center gap-3 text-xs">
                  <span className="w-40 shrink-0 truncate font-semibold text-ink">
                    {s.segment}
                  </span>
                  <div className="h-3 flex-1 overflow-hidden rounded-full bg-panel2">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${(s.total_monetary / maxRevenue) * 100}%`,
                        backgroundColor: s.color,
                      }}
                    />
                  </div>
                  <span className="w-28 shrink-0 text-right text-mute">
                    {money(s.total_monetary)}
                  </span>
                  <span className="w-16 shrink-0 text-right text-mute">
                    {s.pct_revenue.toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </Panel>

          <Panel title="Segments and what to do about them">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="text-mute">
                    {['Segment', 'Customers', '% of base', '% of value',
                      'Avg recency', 'Avg freq', 'Recommended action'].map((h) => (
                      <th key={h} className="px-2 py-2 whitespace-nowrap">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {report.segment_summary.map((s) => (
                    <tr key={s.segment} className="border-t border-edge/60 align-top">
                      <td className="px-2 py-2">
                        <span className="flex items-center gap-2 font-semibold text-ink">
                          <span
                            className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                            style={{ backgroundColor: s.color }}
                          />
                          {s.segment}
                        </span>
                        <span className="mt-0.5 block text-[11px] text-mute">
                          {s.description}
                        </span>
                      </td>
                      <td className="px-2 py-2 text-mute">{s.n_customers}</td>
                      <td className="px-2 py-2 text-mute">{s.pct_customers.toFixed(1)}%</td>
                      <td className="px-2 py-2 text-mute">{s.pct_revenue.toFixed(1)}%</td>
                      <td className="px-2 py-2 text-mute">{s.avg_recency.toFixed(0)}d</td>
                      <td className="px-2 py-2 text-mute">{s.avg_frequency.toFixed(1)}</td>
                      <td className="px-2 py-2 text-mute">{s.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>

          <Panel title="Scored customers">
            <DataTable data={report.customer_table} />
          </Panel>

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

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[11px] text-mute uppercase">{label}</div>
      <div className="font-data text-ink">{value}</div>
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
