import { useCallback, useEffect, useState, type FormEvent } from 'react'

/**
 * The operator's screen: client accounts, server status and backups. It
 * uses the admin key (APP_ADMIN_KEY), not a client sign-in, and keeps the
 * key for this browser tab only.
 */
interface ClientUser {
  username: string
  created_at: number
  usage: { used_mb: number; datasets: number; quota_mb: number }
}
interface Status {
  memory_mb: number | null
  heavy_jobs: { slots: number; running: number; waiting: number }
  disk_free_mb: number
  disk_total_mb: number
  clients: number
  limits: Record<string, number>
  recent_errors: { reference: string; at: number; method: string; path: string; user: string; error: string }[]
  alerts_configured: boolean
  backups: { name: string; created_at: number; size_mb: number }[]
  backup_every_hours: number
}

const KEY = 'analytiq-admin-key'

async function call<T>(key: string, path: string, method = 'GET', body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: { Authorization: `Bearer ${key}`, ...(body ? { 'Content-Type': 'application/json' } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* not JSON */
    }
    throw new Error(res.status === 401 ? 'That admin key was not accepted.' : String(detail))
  }
  return res.json() as Promise<T>
}

export default function AdminPage() {
  const [key, setKey] = useState(() => {
    try {
      return sessionStorage.getItem(KEY) ?? ''
    } catch {
      return ''
    }
  })
  const [typed, setTyped] = useState('')
  const [users, setUsers] = useState<ClientUser[] | null>(null)
  const [status, setStatus] = useState<Status | null>(null)
  const [error, setError] = useState('')
  const [note, setNote] = useState('')
  const [name, setName] = useState('')
  const [pass, setPass] = useState('')

  const load = useCallback(async (k: string) => {
    setError('')
    try {
      const [u, s] = await Promise.all([
        call<{ users: ClientUser[] }>(k, '/api/admin/users'),
        call<Status>(k, '/api/admin/status'),
      ])
      setUsers(u.users)
      setStatus(s)
      try {
        sessionStorage.setItem(KEY, k)
      } catch {
        /* private window */
      }
    } catch (e) {
      setUsers(null)
      setKey('')
      setError((e as Error).message)
    }
  }, [])

  useEffect(() => {
    if (key) load(key)
  }, [key, load])

  const act = async (fn: () => Promise<unknown>, ok: string) => {
    setError('')
    setNote('')
    try {
      await fn()
      setNote(ok)
      await load(key)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const create = (e: FormEvent) => {
    e.preventDefault()
    act(
      () => call(key, '/api/admin/users', 'POST', { username: name, password: pass }),
      `Client '${name}' created. Give them the username and password.`,
    ).then(() => {
      setName('')
      setPass('')
    })
  }

  const backup = async (saved?: string) => {
    setError('')
    try {
      const res = await fetch(saved ? `/api/admin/backups/${encodeURIComponent(saved)}` : '/api/admin/backup', {
        headers: { Authorization: `Bearer ${key}` },
      })
      if (!res.ok) throw new Error('The backup could not be downloaded.')
      const url = URL.createObjectURL(await res.blob())
      const a = document.createElement('a')
      a.href = url
      a.download = saved ?? `analytiq-backup-${new Date().toISOString().slice(0, 10)}.tar.gz`
      document.body.appendChild(a)
      a.click()
      a.remove()
      setTimeout(() => URL.revokeObjectURL(url), 60_000)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const input =
    'rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink outline-none focus:border-accent'
  const btn = 'rounded-lg border border-edge px-2.5 py-1 text-[11px] font-semibold text-ink2 hover:bg-panel2'

  if (!key || !users) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            setKey(typed.trim())
          }}
          className="w-full max-w-sm rounded-xl border border-edge bg-panel p-6"
        >
          <h1 className="text-base font-bold text-ink">Analytiq admin</h1>
          <p className="mt-1 text-xs text-mute">Enter the admin key set on the server (APP_ADMIN_KEY).</p>
          <input
            className={`${input} mt-4 w-full`}
            type="password"
            placeholder="Admin key"
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
            autoFocus
          />
          {error && <p className="mt-2 text-xs text-rose">{error}</p>}
          <button className="mt-4 w-full rounded-lg bg-accent px-4 py-2 text-xs font-semibold text-white">
            Open
          </button>
        </form>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-4xl space-y-5 p-6" data-testid="admin">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-bold text-ink">Analytiq admin</h1>
        <button
          className={btn}
          onClick={() => {
            try {
              sessionStorage.removeItem(KEY)
            } catch {
              /* private window */
            }
            setKey('')
            setUsers(null)
          }}
        >
          Lock
        </button>
      </div>
      {error && <p className="rounded-lg border border-rose/40 bg-rose/10 p-3 text-xs text-rose">{error}</p>}
      {note && <p className="rounded-lg border border-teal/40 bg-teal/10 p-3 text-xs text-teal">{note}</p>}

      {status && (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          {[
            ['Memory in use', status.memory_mb != null ? `${status.memory_mb} MB` : '—'],
            ['Disk free', `${(status.disk_free_mb / 1024).toFixed(1)} of ${(status.disk_total_mb / 1024).toFixed(1)} GB`],
            ['Heavy jobs', `${status.heavy_jobs.running} running, ${status.heavy_jobs.waiting} waiting`],
            ['Clients', String(status.clients)],
          ].map(([label, value]) => (
            <div key={label} className="rounded-xl border border-edge bg-panel p-3">
              <div className="text-[10px] uppercase text-mute">{label}</div>
              <div className="mt-1 text-sm font-bold text-ink">{value}</div>
            </div>
          ))}
        </div>
      )}

      <div className="rounded-xl border border-edge bg-panel p-4">
        <h2 className="text-sm font-bold text-ink">New client</h2>
        <form onSubmit={create} className="mt-3 flex flex-wrap gap-2">
          <input className={input} placeholder="Username" value={name} onChange={(e) => setName(e.target.value)} required />
          <input
            className={input}
            placeholder="Password (8+ characters)"
            value={pass}
            onChange={(e) => setPass(e.target.value)}
            required
            minLength={8}
          />
          <button className="rounded-lg bg-accent px-4 py-2 text-xs font-semibold text-white">Create</button>
        </form>
      </div>

      <div className="rounded-xl border border-edge bg-panel p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold text-ink">Clients</h2>
          <button className={btn} onClick={() => backup()}>
            Download backup
          </button>
        </div>
        <table className="mt-3 w-full text-left text-xs">
          <thead className="text-[10px] uppercase text-mute">
            <tr>
              <th className="py-1.5">Username</th>
              <th>Created</th>
              <th>Datasets</th>
              <th>Storage</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.username} className="border-t border-edge text-ink2">
                <td className="py-2 font-semibold">{u.username}</td>
                <td>{new Date(u.created_at * 1000).toLocaleDateString()}</td>
                <td>{u.usage.datasets}</td>
                <td>
                  {u.usage.used_mb} MB{u.usage.quota_mb ? ` of ${u.usage.quota_mb}` : ''}
                </td>
                <td className="space-x-2 text-right">
                  <button
                    className={btn}
                    onClick={() => {
                      const p = window.prompt(`New password for ${u.username} (8+ characters):`)
                      if (p)
                        act(
                          () => call(key, `/api/admin/users/${u.username}/password`, 'POST', { password: p }),
                          `Password reset for '${u.username}'. Their other sessions are signed out.`,
                        )
                    }}
                  >
                    Reset password
                  </button>
                  <button
                    className={`${btn} text-rose`}
                    onClick={() => {
                      if (window.confirm(`Delete '${u.username}' and ALL of their data? This cannot be undone.`))
                        act(() => call(key, `/api/admin/users/${u.username}`, 'DELETE'), `Client '${u.username}' and their data deleted.`)
                    }}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
            {users.length === 0 && (
              <tr>
                <td colSpan={5} className="py-4 text-center text-mute">
                  No clients yet. Create the first one above.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {status && (
        <div className="grid gap-5 lg:grid-cols-2">
          <div className="rounded-xl border border-edge bg-panel p-4" data-testid="backups">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold text-ink">Saved backups</h2>
              <button
                className={btn}
                onClick={() => act(() => call(key, '/api/admin/backups', 'POST'), 'Backup taken.')}
              >
                Back up now
              </button>
            </div>
            <p className="mt-1 text-[11px] text-mute">
              {status.backup_every_hours > 0
                ? `Taken every ${status.backup_every_hours} hours and kept on the server's disk.`
                : 'Scheduled backups are off (BACKUP_EVERY_HOURS=0).'}{' '}
              They do not survive losing that disk: download one regularly and keep it elsewhere.
            </p>
            <ul className="mt-3 space-y-1.5 text-xs">
              {status.backups.map((b) => (
                <li key={b.name} className="flex items-center justify-between text-ink2">
                  <span>
                    {new Date(b.created_at * 1000).toLocaleString()} · {b.size_mb} MB
                  </span>
                  <button className={btn} onClick={() => backup(b.name)}>
                    Download
                  </button>
                </li>
              ))}
              {status.backups.length === 0 && <li className="text-mute">None yet.</li>}
            </ul>
          </div>

          <div className="rounded-xl border border-edge bg-panel p-4" data-testid="errors">
            <h2 className="text-sm font-bold text-ink">Recent errors</h2>
            <p className="mt-1 text-[11px] text-mute">
              Unexpected failures since the server last started, newest first. A client quoting a reference
              is quoting one of these.{' '}
              {status.alerts_configured
                ? 'Alerts are being sent to your webhook.'
                : 'Set ALERT_WEBHOOK_URL to be told as they happen.'}
            </p>
            <ul className="mt-3 space-y-2 text-xs">
              {status.recent_errors.map((e) => (
                <li key={e.reference} className="text-ink2">
                  <div>
                    <code className="text-ink">{e.reference}</code> · {new Date(e.at * 1000).toLocaleString()} ·{' '}
                    {e.user}
                  </div>
                  <div className="break-all text-mute">
                    {e.method} {e.path} — {e.error}
                  </div>
                </li>
              ))}
              {status.recent_errors.length === 0 && <li className="text-mute">None.</li>}
            </ul>
          </div>
        </div>
      )}
    </div>
  )
}
