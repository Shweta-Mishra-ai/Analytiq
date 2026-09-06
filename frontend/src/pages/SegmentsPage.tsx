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

interface Candidate {
  column: string
  note: string
  distinct?: number
}

interface Candidates {
  customer: Candidate[]
  date: Candidate[]
  monetary: Candidate[]
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
  // What the user picked when detection came up empty. The endpoint has
  // always accepted these; the page just never offered them.
  const [candidates, setCandidates] = useState<Candidates | null>(null)
  const [pick, setPick] = useState({ customer: '', date: '', monetary: '' })
  const [report, setReport] = useState<RfmReport | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [checked, setChecked] = useState(false)

  useEffect(() => {
    if (!ds) return
    setReport(null)
    setError('')
    setChecked(false)
    setPick({ customer: '', date: '', monetary: '' })
    apiGet<{ detected: RfmColumns | null; candidates: Candidates }>(
      `/api/analytics/${ds}/rfm/columns`,
    )
      .then((r) => {
        setDetected(r.detected)
        setCandidates(r.candidates)
        if (!r.detected && r.candidates) {
          // Lead with the most likely answer rather than three empty
          // dropdowns: the point is to make the choice easy, not to
          // hand the work back.
          setPick({
            customer: r.candidates.customer[0]?.column ?? '',
            date: r.candidates.date[0]?.column ?? '',
            monetary: r.candidates.monetary[0]?.column ?? '',
          })
        }
      })
      .catch((e) => setError(e.message))
      .finally(() => setChecked(true))
  }, [ds])

  const run = useCallback(async () => {
    if (!ds) return
    setBusy(true)
    setError('')
    try {
      const q = new URLSearchParams()
      if (!detected) {
        if (pick.customer) q.set('customer_col', pick.customer)
        if (pick.date) q.set('date_col', pick.date)
        if (pick.monetary) q.set('monetary_col', pick.monetary)
      }
      const qs = q.toString()
      setReport(
        await apiGet<RfmReport>(
          `/api/analytics/${ds}/rfm${qs ? `?${qs}` : ''}`,
        ),
      )
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [ds, detected, pick])

  if (!dataset) return <NeedData />

  const maxRevenue = Math.max(
    1,
    ...(report?.segment_summary ?? []).map((s) => s.total_monetary),
  )

  return (
    <div className="p-8">
      <PageHeader
        title="Customer Segments"
        subtitle="Who your best customers are, who is slipping away, and what to do about each — ranked on how recently they bought, how often, and how much (RFM)"
        right={
          <Btn
            onClick={run}
            disabled={busy || (!detected && !(pick.customer && pick.date))}
          >
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

      {/* Detection reads column names, so a file that calls its customer
          column `account` or `member` used to end here — "this dataset
          doesn't look like transaction data", and nothing to do about
          it. The endpoint always accepted explicit columns; now the page
          asks, with its best guess already filled in. */}
      {checked && !detected && !report && (
        <Panel
          title="Which columns describe your customers?"
          subtitle="We could not tell from the column names. Confirm these three and the analysis will run."
        >
          {candidates && candidates.customer.length > 0 &&
          candidates.date.length > 0 ? (
            <div className="space-y-4">
              <div className="grid gap-4 md:grid-cols-3">
                <Picker
                  label="Who the customer is"
                  hint="The column that repeats once per customer"
                  options={candidates.customer}
                  value={pick.customer}
                  onChange={(v) => setPick((p) => ({ ...p, customer: v }))}
                />
                <Picker
                  label="When it happened"
                  hint="The date of each purchase or event"
                  options={candidates.date}
                  value={pick.date}
                  onChange={(v) => setPick((p) => ({ ...p, date: v }))}
                />
                <Picker
                  label="What it was worth"
                  hint="Optional — leave blank to rank on how often, not how much"
                  options={candidates.monetary}
                  value={pick.monetary}
                  onChange={(v) => setPick((p) => ({ ...p, monetary: v }))}
                  allowNone
                />
              </div>
              <p className="text-xs text-faint">
                Customers are then ranked on three things: how recently
                they bought, how often, and how much. Each gets a segment
                and an action.
              </p>
            </div>
          ) : (
            <p className="text-sm text-mute">
              This file has no column that repeats per customer and no
              date column, so there is nothing to score. Customer
              segmentation needs one row per purchase, with a customer
              and a date on each.
            </p>
          )}
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

/** One column choice, with what the data says about each option so the
 *  answer is checkable rather than a guess. */
function Picker({
  label,
  hint,
  options,
  value,
  onChange,
  allowNone = false,
}: {
  label: string
  hint: string
  options: Candidate[]
  value: string
  onChange: (v: string) => void
  allowNone?: boolean
}) {
  return (
    <label className="block">
      <span className="text-[11px] uppercase tracking-wide text-mute">
        {label}
      </span>
      <select
        aria-label={label}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 w-full rounded-md border border-edge bg-panel2 px-2 py-1.5 text-sm text-ink"
      >
        {allowNone && <option value="">— none —</option>}
        {options.map((o) => (
          <option key={o.column} value={o.column}>
            {o.column} · {o.note}
          </option>
        ))}
      </select>
      <span className="mt-1 block text-xs text-faint">{hint}</span>
    </label>
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
