import { useCallback, useEffect, useMemo, useState } from 'react'
import { CheckCircle2, Save } from 'lucide-react'
import { apiGet, apiPost } from '../api/client'
import { useApp } from '../store/app'
import { Badge, Btn, ErrorBox, NeedData, PageHeader, Panel, Spinner } from '../components/Ui'

type Role = 'identifier' | 'date' | 'measure' | 'category' | 'text'

interface ColumnInfo {
  name: string
  role: Role
  direction: number
  confirmed: boolean
  dtype: string
  unique: number
  missing_pct: number
  examples: string[]
}

interface ColumnsResponse {
  columns: ColumnInfo[]
  confirmed: boolean
  roles: Role[]
}

// What each role means for the analysis, in the words the user decides by.
const ROLE_LABEL: Record<Role, string> = {
  identifier: 'Identifier — names a row, never analysed',
  date: 'Date',
  measure: 'Measure — a quantity to add up or average',
  category: 'Category — groups rows',
  text: 'Free text',
}

const field =
  'w-full rounded-lg border border-edge bg-panel2 px-2.5 py-1.5 text-xs text-ink'

interface Choice {
  role: Role
  direction: number
}

export default function ColumnsPage() {
  const dataset = useApp((s) => s.dataset)
  const ds = dataset?.dataset_id
  const [data, setData] = useState<ColumnsResponse | null>(null)
  const [choices, setChoices] = useState<Record<string, Choice>>({})
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const [notes, setNotes] = useState<string[] | null>(null)

  const load = useCallback(async () => {
    if (!ds) return
    setError('')
    try {
      const r = await apiGet<ColumnsResponse>(`/api/datasets/${ds}/columns`)
      setData(r)
      setChoices(
        Object.fromEntries(r.columns.map((c) => [c.name, { role: c.role, direction: c.direction }])),
      )
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [ds])

  useEffect(() => {
    setData(null)
    setNotes(null)
    load()
  }, [load])

  const changed = useMemo(
    () =>
      (data?.columns ?? []).filter(
        (c) =>
          choices[c.name] &&
          (choices[c.name].role !== c.role || choices[c.name].direction !== c.direction),
      ).length,
    [data, choices],
  )

  if (!dataset) return <NeedData />

  const set = (name: string, patch: Partial<Choice>) =>
    setChoices((x) => ({ ...x, [name]: { ...x[name], ...patch } }))

  const save = async () => {
    setSaving(true)
    setError('')
    setNotes(null)
    try {
      const r = await apiPost<{ saved: number; notes: string[] }>(
        `/api/datasets/${ds}/columns`,
        { columns: choices },
      )
      // Reload first, so "Saved" never sits above a table still showing
      // the previous, unconfirmed state.
      await load()
      setNotes(r.notes)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="mx-auto max-w-6xl p-6">
      <PageHeader
        title="Columns"
        subtitle="What each column is. Every analysis uses this — identifiers are never averaged, labels are never summed, and a measure is judged the right way round."
        right={
          <Btn onClick={save} disabled={!data || saving}>
            <span className="inline-flex items-center gap-1.5">
              <Save className="h-4 w-4" />
              {saving ? 'Saving…' : data?.confirmed ? 'Save changes' : 'Confirm columns'}
            </span>
          </Btn>
        }
      />

      {error && <ErrorBox message={error} />}
      {notes && (
        <div className="mb-4 rounded-lg border border-teal/30 bg-teal/5 px-4 py-3 text-sm text-ink">
          <div className="flex items-center gap-1.5 font-semibold text-teal">
            <CheckCircle2 className="h-4 w-4" /> Saved — every page now uses these columns.
          </div>
          {notes.map((n) => (
            <p key={n} className="mt-1 text-xs text-amber">
              {n}
            </p>
          ))}
        </div>
      )}
      {!data && !error && <Spinner label="Reading columns…" />}

      {data && (
        <Panel
          title={`${data.columns.length} columns`}
          subtitle={
            data.confirmed
              ? 'Confirmed. Change any row and save again.'
              : 'These are guesses from the names and values. Check them, then confirm.'
          }
          right={
            changed > 0 ? <Badge tone="amber">{changed} changed</Badge> : undefined
          }
        >
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-left text-xs">
              <thead className="text-mute">
                <tr className="border-b border-edge">
                  <th className="py-2 pr-3 font-semibold">Column</th>
                  <th className="py-2 pr-3 font-semibold">Read as</th>
                  <th className="py-2 pr-3 font-semibold">Better when</th>
                  <th className="py-2 text-right font-semibold">Distinct · blank</th>
                </tr>
              </thead>
              <tbody>
                {data.columns.map((c) => {
                  const choice = choices[c.name] ?? { role: c.role, direction: c.direction }
                  return (
                    <tr key={c.name} className="border-b border-edge/60 align-top">
                      <td className="py-2.5 pr-3">
                        <div className="flex items-center gap-1.5 font-semibold text-ink">
                          {c.name}
                          {c.confirmed && <Badge tone="teal">confirmed</Badge>}
                        </div>
                        <div className="mt-0.5 truncate text-[11px] text-mute" title={c.examples.join(', ')}>
                          {c.examples.length ? `e.g. ${c.examples.join(', ')}` : 'no values'}
                        </div>
                      </td>
                      <td className="w-72 py-2.5 pr-3">
                        <select
                          aria-label={`How to read ${c.name}`}
                          className={field}
                          value={choice.role}
                          onChange={(e) =>
                            set(c.name, {
                              role: e.target.value as Role,
                              direction: e.target.value === 'measure' ? choice.direction : 0,
                            })
                          }
                        >
                          {data.roles.map((r) => (
                            <option key={r} value={r}>
                              {ROLE_LABEL[r]}
                            </option>
                          ))}
                        </select>
                      </td>
                      <td className="w-48 py-2.5 pr-3">
                        {choice.role === 'measure' ? (
                          <select
                            aria-label={`Which way is better for ${c.name}`}
                            className={field}
                            value={choice.direction}
                            onChange={(e) => set(c.name, { direction: Number(e.target.value) })}
                          >
                            <option value={1}>Higher is better</option>
                            <option value={-1}>Lower is better</option>
                            <option value={0}>Neither</option>
                          </select>
                        ) : (
                          <span className="text-mute">—</span>
                        )}
                      </td>
                      <td className="py-2.5 text-right text-mute">
                        {c.unique.toLocaleString()} · {c.missing_pct}%
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-[11px] text-mute">
            Turning text into dates or measures needs at least 95% of a column’s values to
            convert; any that don’t are left blank in the analysis and counted for you. Your
            uploaded file is never changed.
          </p>
        </Panel>
      )}
    </div>
  )
}
