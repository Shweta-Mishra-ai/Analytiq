/**
 * Pulling a dataset straight out of a client's database.
 *
 * A CSV export is the weakest link in an analysis — a stale copy with
 * no lineage and no types — so this is the path that avoids it. The
 * connection string is typed here, used for the request, and never
 * stored: neither by the browser nor by the server, which redacts the
 * password out of everything it writes down.
 */
import { useCallback, useEffect, useState } from 'react'
import { Database, Loader2, Play, Table2 } from 'lucide-react'
import { apiGet, apiPost } from '../api/client'
import type { DatasetMeta } from '../api/client'
import { Badge, Btn, ErrorBox, Panel } from '../components/Ui'
import * as fmt from '../lib/format'

interface Backend {
  key: string
  label: string
  available: boolean
  url_prefix: string
  example: string
  install: string
}

interface TableRow {
  schema: string
  name: string
  qualified: string
}

interface PreviewResult {
  source: string
  sql: string
  rows: number
  columns: string[]
  warnings: string[]
}

export default function WarehouseSource({
  onImported,
}: {
  onImported: (meta: DatasetMeta) => void
}) {
  const [backends, setBackends] = useState<Backend[]>([])
  const [url, setUrl] = useState('')
  const [sql, setSql] = useState('')
  const [tables, setTables] = useState<TableRow[] | null>(null)
  const [preview, setPreview] = useState<PreviewResult | null>(null)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [note, setNote] = useState('')

  useEffect(() => {
    apiGet<{ backends: Backend[] }>('/api/datasets/warehouse/backends')
      // An older server, or one that answered with something else
      // entirely, must cost this panel its badges — not the page.
      .then((r) => setBackends(Array.isArray(r?.backends) ? r.backends : []))
      .catch(() => setBackends([]))
  }, [])

  const run = useCallback(
    async (label: string, fn: () => Promise<void>) => {
      setBusy(label)
      setError('')
      try {
        await fn()
      } catch (e) {
        setError((e as Error).message)
      } finally {
        setBusy('')
      }
    },
    [],
  )

  const test = () =>
    run('test', async () => {
      const r = await apiPost<{ ok: boolean; dialect?: string; error: string; source: string }>(
        '/api/datasets/warehouse/test',
        { url },
      )
      if (!r.ok) throw new Error(r.error)
      setNote(`Connected to ${r.source}${r.dialect ? ` (${r.dialect})` : ''}.`)
    })

  const list = () =>
    run('tables', async () => {
      const r = await apiPost<{ tables: TableRow[] }>(
        '/api/datasets/warehouse/tables',
        { url },
      )
      setTables(r.tables)
      setNote(`${fmt.count(r.tables.length)} table(s) and view(s) found.`)
    })

  const doPreview = () =>
    run('preview', async () => {
      const r = await apiPost<PreviewResult>('/api/datasets/warehouse/preview', {
        url,
        sql,
      })
      setPreview(r)
      setNote('')
    })

  const doImport = () =>
    run('import', async () => {
      const r = await apiPost<{ meta: DatasetMeta }>(
        '/api/datasets/warehouse/import',
        { url, sql },
      )
      onImported(r.meta)
      setNote(`Imported ${fmt.count(r.meta.rows)} rows.`)
    })

  const unavailable = backends.filter((b) => !b.available && b.install)

  return (
    <Panel
      title="Import from a database"
      subtitle="Read straight from Postgres, MySQL, Snowflake, BigQuery or SQL Server instead of exporting a CSV first"
    >
      <div className="mb-3 flex flex-wrap gap-1.5">
        {backends.map((b) => (
          <Badge key={b.key} tone={b.available ? 'teal' : 'neutral'}>
            {b.label}
          </Badge>
        ))}
      </div>

      <label className="block text-xs font-semibold text-mute" htmlFor="wh-url">
        Connection URL
      </label>
      <input
        id="wh-url"
        type="password"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        placeholder="postgresql+psycopg://user:password@host:5432/database"
        className="mt-1 w-full rounded-lg border border-edge bg-panel2 px-3 py-2 font-mono text-sm text-ink"
      />
      <p className="mt-1 text-xs text-faint">
        Used for this request only. It is never saved, and the password is
        stripped from anything the server writes down — including the audit
        trail and the report.
      </p>

      <div className="mt-3 flex flex-wrap gap-2">
        <Btn variant="subtle" size="sm" onClick={test} disabled={!url || !!busy}>
          {busy === 'test' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Database className="h-3.5 w-3.5" />}
          {' '}Test connection
        </Btn>
        <Btn variant="subtle" size="sm" onClick={list} disabled={!url || !!busy}>
          {busy === 'tables' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Table2 className="h-3.5 w-3.5" />}
          {' '}List tables
        </Btn>
      </div>

      {note && <p className="mt-2 text-xs text-teal">{note}</p>}
      {error && (
        <div className="mt-3">
          <ErrorBox message={error} />
        </div>
      )}

      {tables && tables.length > 0 && (
        <div className="mt-3 max-h-40 overflow-y-auto rounded-lg border border-edge">
          {tables.map((t) => (
            <button
              key={t.qualified}
              onClick={() => setSql(`SELECT * FROM ${t.qualified}`)}
              className="block w-full px-3 py-1.5 text-left font-mono text-xs text-ink2 hover:bg-panel2"
            >
              {t.qualified}
            </button>
          ))}
        </div>
      )}

      <label className="mt-4 block text-xs font-semibold text-mute" htmlFor="wh-sql">
        Query
      </label>
      <textarea
        id="wh-sql"
        value={sql}
        onChange={(e) => setSql(e.target.value)}
        rows={4}
        placeholder="SELECT * FROM public.employees WHERE hire_date >= '2024-01-01'"
        className="mt-1 w-full rounded-lg border border-edge bg-panel2 px-3 py-2 font-mono text-sm text-ink"
      />
      <p className="mt-1 text-xs text-faint">
        SELECT and WITH only. Anything else is refused before it reaches your
        database, and the connection is rolled back either way.
      </p>

      <div className="mt-3 flex flex-wrap gap-2">
        <Btn variant="subtle" size="sm" onClick={doPreview} disabled={!url || !sql || !!busy}>
          {busy === 'preview' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
          {' '}Preview 100 rows
        </Btn>
        <Btn size="sm" onClick={doImport} disabled={!url || !sql || !!busy}>
          {busy === 'import' ? 'Importing…' : 'Import as dataset'}
        </Btn>
      </div>

      {preview && (
        <div className="mt-3 rounded-lg border border-edge p-3 text-sm">
          <div className="text-ink">
            {fmt.count(preview.rows)} row(s), {fmt.count(preview.columns.length)}{' '}
            column(s) from {preview.source}
          </div>
          <div className="mt-1 font-mono text-xs text-faint">
            {preview.columns.slice(0, 12).join(', ')}
            {preview.columns.length > 12 ? ' …' : ''}
          </div>
          {preview.warnings.map((w, i) => (
            <p key={i} className="mt-1 text-xs text-amber">
              {w}
            </p>
          ))}
        </div>
      )}

      {unavailable.length > 0 && (
        <p className="mt-4 text-xs text-faint">
          Not installed on this server:{' '}
          {unavailable.map((b) => `${b.label} (${b.install})`).join(', ')}.
        </p>
      )}
    </Panel>
  )
}
