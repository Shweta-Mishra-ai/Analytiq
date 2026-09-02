import { useEffect, useState } from 'react'
import {
  AlertTriangle,
  Fingerprint,
  FileLock2,
  ShieldCheck,
  ShieldX,
  Users,
} from 'lucide-react'
import { apiGet } from '../api/client'
import { useApp } from '../store/app'
import { ErrorBox, NeedData, PageHeader, Panel, Spinner } from '../components/Ui'
import * as fmt from '../lib/format'

interface ColumnRecord {
  name: string
  label: string
  dtype: string
  role: string
  sensitivity: string
  completeness_pct: number
  distinct: number
  example: string
}

interface Reidentification {
  quasi_identifiers: string[]
  k_min: number
  unique_rows: number
  unique_pct: number
  below_floor_rows: number
  below_floor_pct: number
  verdict: string
  explanation: string
}

interface AuditEntry {
  seq: number
  at: number
  event: string
  actor: string
  detail: Record<string, unknown>
}

interface Integrity {
  record: {
    source_filename: string
    source_bytes: number
    source_sha256: string
    raw_digest: string
    active_digest: string
    ingested_at: number
  } | null
  verdict: {
    intact: boolean
    verdict: string
    explanation: string
    events: number
  }
  audit: AuditEntry[]
  manifest: Record<string, string>
}

interface Governance {
  source_file: string
  ingested_at: string
  rows: number
  columns: number
  retention_days: number | null
  retention_note: string
  dictionary: ColumnRecord[]
  direct_identifiers: string[]
  quasi_identifiers: string[]
  special_category: string[]
  reidentification: Reidentification | null
  lineage: string[]
  obligations: string[]
}

const SENSITIVITY_TONE: Record<string, string> = {
  special: 'text-rose',
  direct: 'text-rose',
  'quasi-identifier': 'text-amber',
}

const SENSITIVITY_ORDER: Record<string, number> = {
  special: 0,
  direct: 1,
  'quasi-identifier': 2,
  none: 3,
}

export default function GovernancePage() {
  const dataset = useApp((s) => s.dataset)
  const [record, setRecord] = useState<Governance | null>(null)
  const [integrity, setIntegrity] = useState<Integrity | null>(null)
  const [error, setError] = useState('')
  const ds = dataset?.dataset_id

  useEffect(() => {
    if (!ds) return
    setRecord(null)
    setError('')
    apiGet<Governance>(`/api/datasets/${ds}/governance`)
      .then(setRecord)
      .catch((e) => setError(e.message))
    // Integrity is fetched separately and failing softly on purpose: a
    // dataset stored before integrity tracking existed has no record,
    // and that must not blank out the governance page.
    setIntegrity(null)
    apiGet<Integrity>(`/api/datasets/${ds}/integrity`)
      .then(setIntegrity)
      .catch(() => setIntegrity(null))
  }, [ds])

  if (!dataset) return <NeedData />

  const risk = record?.reidentification ?? null
  const riskTone =
    risk?.verdict === 'High'
      ? 'text-rose'
      : risk?.verdict === 'Moderate'
        ? 'text-amber'
        : 'text-teal'

  return (
    <div className="p-8">
      <PageHeader
        title="Data Governance"
        subtitle="What this data is, where it came from, and who it could still identify"
      />
      {error && (
        <div className="mb-4">
          <ErrorBox message={error} />
        </div>
      )}
      {!record && !error && <Spinner label="Classifying columns…" />}

      {record && (
        <div className="space-y-5">
          {/* What the classification requires of whoever holds the file —
              stated as the consequence, not as a citation. */}
          <Panel title="What this means for handling">
            <div className="space-y-2.5">
              {record.obligations.map((item, i) => {
                // The icon follows the sentence, not the file: a dataset
                // with no name columns can still carry a warning about
                // re-identification, and a green shield beside it reads
                // as an all-clear it is not.
                const clear = item.startsWith('No column identifies')
                return (
                  <div
                    key={i}
                    className={`flex gap-3 rounded-lg border px-4 py-3 ${
                      clear
                        ? 'border-teal/25 bg-teal/[0.04]'
                        : 'border-amber/25 bg-amber/[0.04]'
                    }`}
                  >
                    {clear ? (
                      <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-teal" />
                    ) : (
                      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber" />
                    )}
                    <p className="text-sm leading-relaxed text-ink2">{item}</p>
                  </div>
                )
              })}
            </div>
          </Panel>

          {integrity && <IntegrityPanel data={integrity} />}

          <div className="grid gap-5 lg:grid-cols-3">
            <Panel title="Provenance" className="lg:col-span-1">
              <dl className="space-y-2.5 text-sm">
                {(
                  [
                    ['Source', record.source_file || 'Supplied dataset'],
                    ['Received', record.ingested_at || '—'],
                    [
                      'Scope',
                      `${fmt.count(record.rows)} rows × ${record.columns} columns`,
                    ],
                    [
                      'Retention',
                      record.retention_days
                        ? `${record.retention_days} days`
                        : 'Not set',
                    ],
                  ] as const
                ).map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-3">
                    <dt className="shrink-0 text-mute">{k}</dt>
                    <dd className="truncate text-right text-ink2">{v}</dd>
                  </div>
                ))}
              </dl>
              {record.retention_note && (
                <p className="mt-3 border-t border-edge pt-3 text-[11px] leading-relaxed text-faint">
                  {record.retention_note}
                </p>
              )}
            </Panel>

            <Panel title="Re-identification risk" className="lg:col-span-2">
              {risk ? (
                <>
                  <div className="grid grid-cols-3 gap-3">
                    {(
                      [
                        [
                          'Smallest group',
                          fmt.count(risk.k_min),
                          risk.k_min < 5 ? 'text-rose' : 'text-teal',
                          'records sharing a profile',
                        ],
                        [
                          'Unique records',
                          fmt.count(risk.unique_rows),
                          risk.unique_pct >= 5 ? 'text-rose' : 'text-teal',
                          `${fmt.pct(risk.unique_pct)} of the file`,
                        ],
                        ['Verdict', risk.verdict, riskTone, 'on these fields'],
                      ] as const
                    ).map(([label, val, tone, sub]) => (
                      <div key={label}>
                        <div className="text-[11px] uppercase tracking-wide text-mute">
                          {label}
                        </div>
                        <div className={`font-data text-2xl font-bold ${tone}`}>
                          {val}
                        </div>
                        <div className="mt-0.5 text-[11px] text-faint">
                          {sub}
                        </div>
                      </div>
                    ))}
                  </div>
                  <p className="mt-4 border-t border-edge pt-3 text-sm leading-relaxed text-ink2">
                    {risk.explanation}
                  </p>
                  <p className="mt-2 flex items-start gap-1.5 text-[11px] leading-relaxed text-faint">
                    <Users className="mt-0.5 h-3 w-3 shrink-0" />
                    Measured as k-anonymity over{' '}
                    {fmt.joinAnd(risk.quasi_identifiers.map(fmt.label), 4)}.
                    A group of one is a person who can be picked out by anyone
                    holding the same facts, whether or not a name column is
                    present.
                  </p>
                </>
              ) : (
                <p className="text-sm text-mute">
                  Fewer than two quasi-identifying fields, so no combination
                  of them can single anyone out. There is nothing to measure.
                </p>
              )}
            </Panel>
          </div>

          <Panel
            title="Data dictionary"
            subtitle="Sensitive fields first — that is what this page is read for"
          >
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="text-mute">
                    {[
                      'Field',
                      'Type',
                      'Role',
                      'Sensitivity',
                      'Complete',
                      'Distinct',
                      'Example',
                    ].map((h) => (
                      <th key={h} className="px-2 py-2 font-medium">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[...record.dictionary]
                    .sort(
                      (a, b) =>
                        (SENSITIVITY_ORDER[a.sensitivity] ?? 3) -
                          (SENSITIVITY_ORDER[b.sensitivity] ?? 3) ||
                        a.name.localeCompare(b.name),
                    )
                    .map((col) => (
                      <tr
                        key={col.name}
                        className="border-t border-edge/60 align-top"
                      >
                        <td
                          className="px-2 py-2 font-semibold text-ink"
                          title={col.name}
                        >
                          {col.label}
                        </td>
                        <td className="px-2 py-2 font-data text-mute">
                          {col.dtype}
                        </td>
                        <td className="px-2 py-2 text-mute">{col.role}</td>
                        <td
                          className={`px-2 py-2 font-medium ${
                            SENSITIVITY_TONE[col.sensitivity] ?? 'text-faint'
                          }`}
                        >
                          {col.sensitivity === 'none' ? '—' : col.sensitivity}
                        </td>
                        <td className="px-2 py-2 font-data text-mute">
                          {fmt.pct(col.completeness_pct, 0)}
                        </td>
                        <td className="px-2 py-2 font-data text-mute">
                          {fmt.count(col.distinct)}
                        </td>
                        <td className="max-w-40 truncate px-2 py-2 font-data text-faint">
                          {/* A cleaned boolean example still reads
                              "False"; the uploaded file said "No". */}
                          {col.example ? fmt.value(col.example) : '—'}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </Panel>

          {record.lineage.length > 0 && (
            <Panel title="What was done to the data">
              <ul className="space-y-1.5 text-sm text-mute">
                {record.lineage.map((step, i) => (
                  <li key={i} className="flex gap-2">
                    <FileLock2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-faint" />
                    {step}
                  </li>
                ))}
              </ul>
            </Panel>
          )}
        </div>
      )}
    </div>
  )
}

// Wording per verdict. The verdicts are distinct on purpose: "the data
// changed" and "the record of changes was edited" are different failures
// with different responses, and collapsing them into one red banner
// tells the reader nothing they can act on.
const VERDICTS: Record<string, { label: string; tone: 'teal' | 'amber' | 'rose' }> = {
  intact: { label: 'Verified', tone: 'teal' },
  unaccounted: { label: 'Unverified change', tone: 'rose' },
  tampered: { label: 'Audit trail broken', tone: 'rose' },
  compromised: { label: 'Failed', tone: 'rose' },
  unverifiable: { label: 'Not tracked', tone: 'amber' },
}

function IntegrityPanel({ data }: { data: Integrity }) {
  const v = VERDICTS[data.verdict.verdict] ?? {
    label: 'Unknown',
    tone: 'amber' as const,
  }
  const good = data.verdict.intact
  const rec = data.record

  return (
    <Panel
      title="Data integrity"
      subtitle="Whether this is still the data that was uploaded, and whether every change to it is accounted for"
    >
      <div
        className={`flex gap-3 rounded-lg border px-4 py-3 ${
          v.tone === 'teal'
            ? 'border-teal/25 bg-teal/[0.04]'
            : v.tone === 'amber'
              ? 'border-amber/25 bg-amber/[0.04]'
              : 'border-rose/25 bg-rose/[0.04]'
        }`}
      >
        {good ? (
          <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-teal" />
        ) : (
          <ShieldX className="mt-0.5 h-4 w-4 shrink-0 text-rose" />
        )}
        <div>
          <div className="text-sm font-semibold text-ink">{v.label}</div>
          <p className="mt-0.5 text-sm leading-relaxed text-ink2">
            {data.verdict.explanation}
          </p>
        </div>
      </div>

      {rec?.source_sha256 && (
        <div className="mt-4 rounded-lg border border-edge p-3">
          <div className="flex items-center gap-2 text-xs font-semibold text-mute">
            <Fingerprint className="h-3.5 w-3.5" />
            Source file digest (SHA-256)
          </div>
          <div className="mt-1 break-all font-mono text-[11px] leading-relaxed text-ink2">
            {rec.source_sha256}
          </div>
          <p className="mt-1.5 text-xs text-faint">
            Taken from the uploaded bytes before parsing. Run{' '}
            <span className="font-mono">sha256sum</span> on your original file
            and compare — if it matches, every figure in this workspace came
            from that file and no other.
          </p>
        </div>
      )}

      {data.audit.length > 0 && (
        <div className="mt-4">
          <div className="mb-2 text-xs font-semibold text-mute">
            Chain of custody
          </div>
          <ol className="space-y-1.5">
            {data.audit.slice(-12).map((e) => (
              <li
                key={e.seq}
                className="flex flex-wrap items-baseline gap-x-2 rounded-md border border-edge px-3 py-2 text-sm"
              >
                <span className="font-mono text-xs text-faint">{e.seq}</span>
                <span className="font-medium text-ink">{fmt.label(e.event)}</span>
                <span className="text-xs text-mute">
                  {new Date(e.at * 1000).toLocaleString()}
                </span>
                <span className="text-xs text-faint">{describe(e.detail)}</span>
              </li>
            ))}
          </ol>
          <p className="mt-1.5 text-xs text-faint">
            Each entry carries the hash of the one before it, so a removed or
            edited entry breaks the chain and is reported rather than hidden.
          </p>
        </div>
      )}

      {Object.keys(data.manifest).length > 0 && (
        <p className="mt-4 text-xs text-faint">
          Computed with{' '}
          {Object.entries(data.manifest)
            .filter(([, val]) => val && val !== 'unavailable')
            .map(([k, val]) => `${k.replace(/_/g, '-')} ${val}`)
            .join(', ')}
          . Versions are recorded because they move results — a quantile or a
          solver default can change between releases.
        </p>
      )}
    </Panel>
  )
}

function describe(detail: Record<string, unknown>): string {
  const parts = Object.entries(detail || {})
    .slice(0, 3)
    .map(([k, val]) => {
      const shown = Array.isArray(val) ? `${val.length} item(s)` : String(val)
      return `${fmt.label(k)}: ${shown.length > 24 ? shown.slice(0, 24) + '…' : shown}`
    })
  return parts.join(' · ')
}
