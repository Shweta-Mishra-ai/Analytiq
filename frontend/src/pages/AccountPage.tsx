import { useEffect, useState, type FormEvent } from 'react'
import { apiGet, apiPost, setToken } from '../api/client'
import { ErrorBox, PageHeader, Panel, Spinner } from '../components/Ui'

interface Account {
  username: string
  has_password: boolean
  usage: { used_mb: number; datasets: number; quota_mb: number; max_datasets: number } | null
  max_file_mb: number
  data_ttl_days: number
  support_email: string
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
        </div>
      )}
    </div>
  )
}
