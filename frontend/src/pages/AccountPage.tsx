import { useEffect, useState, type FormEvent } from 'react'
import { apiDelete, apiGet, apiPost, setToken } from '../api/client'
import { ErrorBox, PageHeader, Panel, Spinner } from '../components/Ui'

interface Account {
  username: string
  has_password: boolean
  usage: { used_mb: number; datasets: number; quota_mb: number; max_datasets: number } | null
  max_file_mb: number
  data_ttl_days: number
  support_email: string
  connector: Connector | null
}

interface Connector {
  url: string
  has_key: boolean
  key_created_at: number | null
}

function CopyField({ label, value }: { label: string; value: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      /* clipboard blocked: the text is selectable */
    }
  }
  return (
    <div>
      <div className="text-[10px] uppercase text-mute">{label}</div>
      <div className="mt-1 flex gap-2">
        <code className="min-w-0 flex-1 select-all break-all rounded-lg border border-edge bg-panel2 px-3 py-2 text-[11px] text-ink">
          {value}
        </code>
        <button
          type="button"
          onClick={copy}
          className="shrink-0 rounded-lg border border-edge px-3 text-xs text-ink2 hover:bg-white/5"
        >
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
    </div>
  )
}

/** Connecting this account to Claude: the address and a key that does not expire. */
function ClaudeConnector({ connector }: { connector: Connector }) {
  const [state, setState] = useState(connector)
  const [key, setKey] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const create = async () => {
    if (state.has_key && !window.confirm('The current key will stop working and Claude will need the new one. Continue?')) return
    setBusy(true)
    setError('')
    try {
      const r = await apiPost<{ key: string; connector: Connector }>('/api/account/connector-key')
      setKey(r.key)
      setState(r.connector)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }
  const remove = async () => {
    if (!window.confirm('Claude will lose access to your Analytiq data until you create a new key. Continue?')) return
    setBusy(true)
    setError('')
    try {
      await apiDelete('/api/account/connector-key')
      setKey('')
      setState({ ...state, has_key: false, key_created_at: null })
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel
      title="Use Analytiq in Claude"
      subtitle="Ask Claude about your data in plain words; Analytiq does the calculations"
    >
      <div className="space-y-3" data-testid="connector">
        <ol className="list-decimal space-y-1 pl-4 text-xs text-mute">
          <li>Create a key below.</li>
          <li>
            In Claude, open <span className="text-ink">Customize → Connectors → Add → Add custom connector</span>.
          </li>
          <li>Paste the address, choose <span className="text-ink">No sign in</span>, and add the request header.</li>
          <li>Ask, for example: “Summarise my latest dataset in Analytiq and write the report.”</li>
        </ol>
        <CopyField label="Connector address" value={state.url} />
        {key ? (
          <>
            <CopyField label="Request header name" value="Authorization" />
            <CopyField label="Request header value" value={`Bearer ${key}`} />
            <p className="text-[11px] text-amber">
              Copy the key now: it is not shown again. It opens your data to whoever holds it, so keep it
              private.
            </p>
          </>
        ) : (
          <p className="text-[11px] text-mute">
            {state.has_key && state.key_created_at
              ? `A key was created on ${new Date(state.key_created_at * 1000).toLocaleDateString()}. It cannot be shown again; create a new one if it is lost.`
              : 'No key yet.'}
          </p>
        )}
        {error && <p className="text-xs text-rose">{error}</p>}
        <div className="flex gap-2">
          <button
            type="button"
            disabled={busy}
            onClick={create}
            className="rounded-lg bg-accent px-4 py-2 text-xs font-semibold text-white disabled:opacity-50"
          >
            {state.has_key ? 'Replace key' : 'Create key'}
          </button>
          {state.has_key && (
            <button
              type="button"
              disabled={busy}
              onClick={remove}
              className="rounded-lg border border-edge px-4 py-2 text-xs text-ink2 hover:bg-white/5 disabled:opacity-50"
            >
              Remove key
            </button>
          )}
        </div>
        <p className="text-[11px] text-mute">
          The key does not expire and opens the analysis tools only, not this page or your password. Changing
          your password does not change it.
        </p>
      </div>
    </Panel>
  )
}

export default function AccountPage() {
  const [acc, setAcc] = useState<Account | null>(null)
  const [error, setError] = useState('')
  const [current, setCurrent] = useState('')
  const [next, setNext] = useState('')
  const [again, setAgain] = useState('')
  const [pwError, setPwError] = useState('')
  const [done, setDone] = useState(false)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    apiGet<Account>('/api/account')
      .then(setAcc)
      .catch((e) => setError(e.message))
  }, [])

  const change = async (e: FormEvent) => {
    e.preventDefault()
    setPwError('')
    setDone(false)
    if (next !== again) return setPwError('The two new passwords are not the same.')
    setBusy(true)
    try {
      const r = await apiPost<{ token: string }>('/api/auth/change-password', {
        current_password: current,
        new_password: next,
      })
      setToken(r.token)
      setCurrent('')
      setNext('')
      setAgain('')
      setDone(true)
    } catch (err) {
      setPwError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const input =
    'w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink outline-none focus:border-accent'
  const pct =
    acc?.usage && acc.usage.quota_mb ? Math.min(100, (acc.usage.used_mb / acc.usage.quota_mb) * 100) : 0

  return (
    <div className="page">
      <PageHeader title="Account" subtitle="Your storage, limits and password" />
      {error && <ErrorBox message={error} />}
      {!acc && !error && <Spinner label="Loading account…" />}
      {acc && (
        <div className="grid gap-5 lg:grid-cols-2">
          <Panel title="Storage and limits" subtitle={`Signed in as ${acc.username}`}>
            {acc.usage && (
              <div data-testid="usage">
                <div className="flex justify-between text-xs text-mute">
                  <span>
                    {acc.usage.used_mb.toLocaleString()} MB used
                    {acc.usage.quota_mb ? ` of ${acc.usage.quota_mb.toLocaleString()} MB` : ''}
                  </span>
                  <span>
                    {acc.usage.datasets} dataset{acc.usage.datasets === 1 ? '' : 's'}
                    {acc.usage.max_datasets ? ` of ${acc.usage.max_datasets}` : ''}
                  </span>
                </div>
                <div className="mt-2 h-2 rounded bg-white/5">
                  <div
                    className={`h-2 rounded ${pct > 85 ? 'bg-rose/70' : 'bg-accent/70'}`}
                    style={{ width: `${Math.max(2, pct)}%` }}
                  />
                </div>
              </div>
            )}
            <ul className="mt-4 space-y-1.5 text-xs text-mute">
              <li>Largest file you can upload: {acc.max_file_mb} MB.</li>
              <li>
                {acc.data_ttl_days > 0
                  ? `Each dataset is deleted ${acc.data_ttl_days} days after it was uploaded. Download any report or export you need before then.`
                  : 'Datasets are kept until you delete them.'}
              </li>
              {acc.support_email && (
                <li>
                  Help: <a className="text-accent" href={`mailto:${acc.support_email}`}>{acc.support_email}</a>
                </li>
              )}
            </ul>
          </Panel>

          {acc.has_password && (
            <Panel title="Change password" subtitle="Signs out your other devices">
              <form onSubmit={change} className="space-y-3">
                <input
                  className={input}
                  type="password"
                  placeholder="Current password"
                  autoComplete="current-password"
                  value={current}
                  onChange={(e) => setCurrent(e.target.value)}
                  required
                />
                <input
                  className={input}
                  type="password"
                  placeholder="New password (8 characters or more)"
                  autoComplete="new-password"
                  value={next}
                  onChange={(e) => setNext(e.target.value)}
                  required
                  minLength={8}
                />
                <input
                  className={input}
                  type="password"
                  placeholder="New password again"
                  autoComplete="new-password"
                  value={again}
                  onChange={(e) => setAgain(e.target.value)}
                  required
                />
                {pwError && <p className="text-xs text-rose">{pwError}</p>}
                {done && <p className="text-xs text-teal">Password changed.</p>}
                <button
                  disabled={busy}
                  className="rounded-lg bg-accent px-4 py-2 text-xs font-semibold text-white disabled:opacity-50"
                >
                  {busy ? 'Changing…' : 'Change password'}
                </button>
              </form>
            </Panel>
          )}

          {acc.connector && <ClaudeConnector connector={acc.connector} />}
        </div>
      )}
    </div>
  )
}
