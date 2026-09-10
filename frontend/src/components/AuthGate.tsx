/**
 * Blocks the app behind login when the server requires auth (any admin
 * key set, or at least one client account exists). Open mode (fresh
 * install, zero setup) passes through untouched.
 *
 * The same panel signs people up where the deployment allows it. Until
 * it did, accounts could only be created by an operator holding the
 * admin key — so a visitor who reached this screen had no way forward
 * at all, and the form told them nothing about that.
 */
import { useEffect, useState, type ReactNode } from 'react'
import { Database, Lock, Mail, User } from 'lucide-react'
import { apiGet, apiPost, getToken, setToken } from '../api/client'

type Status = 'checking' | 'open' | 'locked' | 'authed'

export default function AuthGate({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<Status>('checking')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [email, setEmail] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [mode, setMode] = useState<'signin' | 'signup'>('signin')
  const [signupOpen, setSignupOpen] = useState(false)

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
    // Whether this deployment takes new accounts. Failing quietly is
    // right here: a server that cannot answer is one where signup is
    // not on offer, and the sign-in form still works.
    apiGet<{ enabled: boolean }>('/api/auth/signup-status')
      .then((r) => setSignupOpen(Boolean(r.enabled)))
      .catch(() => setSignupOpen(false))
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
    const joining = mode === 'signup'
    const submit = async (e: React.FormEvent) => {
      e.preventDefault()
      setBusy(true)
      setError('')
      try {
        const r = joining
          ? await apiPost<{ token: string }>('/api/auth/signup', {
              username,
              email,
              password,
            })
          : await apiPost<{ token: string }>('/api/auth/login', {
              username,
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
          <label
            htmlFor="analytiq-username"
            className="mb-1 flex items-center gap-1.5 text-xs text-mute"
          >
            <User className="h-3.5 w-3.5" /> Username
          </label>
          <input
            id="analytiq-username"
            name="username"
            autoComplete="username"
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoFocus
            autoCapitalize="off"
            className="mb-3 w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink focus:border-accent focus:outline-none"
          />
          {joining && (
            <>
              <label
                htmlFor="analytiq-email"
                className="mb-1 flex items-center gap-1.5 text-xs text-mute"
              >
                <Mail className="h-3.5 w-3.5" /> Email
              </label>
              <input
                id="analytiq-email"
                name="email"
                autoComplete="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoCapitalize="off"
                className="mb-3 w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink focus:border-accent focus:outline-none"
              />
            </>
          )}
          <label
            htmlFor="analytiq-password"
            className="mb-1 flex items-center gap-1.5 text-xs text-mute"
          >
            <Lock className="h-3.5 w-3.5" /> Password
          </label>
          <input
            id="analytiq-password"
            name="password"
            autoComplete={joining ? 'new-password' : 'current-password'}
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink focus:border-accent focus:outline-none"
          />
          {joining && (
            <p className="mt-1 text-[11px] text-faint">
              At least 8 characters.
            </p>
          )}
          {error && <p className="mt-2 text-xs text-rose">{error}</p>}
          <button
            type="submit"
            disabled={busy || !username || !password || (joining && !email)}
            className="mt-4 w-full rounded-lg bg-accent py-2 text-sm font-semibold text-white disabled:opacity-40"
          >
            {busy
              ? joining
                ? 'Creating your account…'
                : 'Signing in…'
              : joining
                ? 'Create account'
                : 'Sign in'}
          </button>
          {signupOpen && (
            <button
              type="button"
              onClick={() => {
                setMode(joining ? 'signin' : 'signup')
                setError('')
              }}
              className="mt-3 w-full text-center text-xs text-mute hover:text-ink"
            >
              {joining
                ? 'Already have an account? Sign in'
                : 'New here? Create an account'}
            </button>
          )}
        </form>
      </div>
    )
  }

  return <>{children}</>
}
