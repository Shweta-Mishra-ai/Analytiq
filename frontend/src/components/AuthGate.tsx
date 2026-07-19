/**
 * Blocks the app behind a workspace password when the server has
 * APP_PASSWORD set. Open mode (no password configured) passes through.
 */
import { useEffect, useState, type ReactNode } from 'react'
import { Database, Lock } from 'lucide-react'
import { apiGet, apiPost, getToken, setToken } from '../api/client'

type Status = 'checking' | 'open' | 'locked' | 'authed'

export default function AuthGate({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<Status>('checking')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const probe = async () => {
    try {
      const h = await apiGet<{ auth_required: boolean }>('/api/health')
      if (!h.auth_required) return setStatus('open')
      if (!getToken()) return setStatus('locked')
      // verify the stored token still works
      try {
        await apiGet('/api/datasets')
        setStatus('authed')
      } catch {
        setStatus('locked')
      }
    } catch {
      // backend unreachable — let the app render its own errors
      setStatus('open')
    }
  }

  useEffect(() => {
    probe()
    const onExpired = () => setStatus('locked')
    window.addEventListener('analytiq-unauthorized', onExpired)
    return () => window.removeEventListener('analytiq-unauthorized', onExpired)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (status === 'checking') {
    return (
      <div className="flex h-full items-center justify-center text-sm text-mute">
        Loading…
      </div>
    )
  }

  if (status === 'locked') {
    const submit = async (e: React.FormEvent) => {
      e.preventDefault()
      setBusy(true)
      setError('')
      try {
        const r = await apiPost<{ token: string }>('/api/auth/login', {
          password,
        })
        setToken(r.token)
        setStatus('authed')
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
      } finally {
        setBusy(false)
      }
    }
    return (
      <div className="flex h-full items-center justify-center">
        <form
          onSubmit={submit}
          className="w-80 rounded-2xl border border-edge bg-panel p-6"
        >
          <div className="mb-4 flex items-center gap-2">
            <Database className="h-6 w-6 text-accent" />
            <div className="text-sm font-bold">Analytiq</div>
          </div>
          <label className="mb-1 flex items-center gap-1.5 text-xs text-mute">
            <Lock className="h-3.5 w-3.5" /> Workspace password
          </label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoFocus
            className="w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink focus:border-accent focus:outline-none"
          />
          {error && <p className="mt-2 text-xs text-rose">{error}</p>}
          <button
            type="submit"
            disabled={busy || !password}
            className="mt-4 w-full rounded-lg bg-accent py-2 text-sm font-semibold text-white disabled:opacity-40"
          >
            {busy ? 'Checking…' : 'Enter workspace'}
          </button>
        </form>
      </div>
    )
  }

  return <>{children}</>
}
