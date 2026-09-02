import { useEffect, useState } from 'react'
import { AlertTriangle, FileLock2, ShieldCheck, Users } from 'lucide-react'
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
  const [error, setError] = useState('')
  const ds = dataset?.dataset_id

  useEffect(() => {
    if (!ds) return
    setRecord(null)
    setError('')
    apiGet<Governance>(`/api/datasets/${ds}/governance`)
      .then(setRecord)
      .catch((e) => setError(e.message))
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
