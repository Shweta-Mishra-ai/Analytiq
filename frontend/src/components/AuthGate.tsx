/**
 * Blocks the app behind login when the server requires auth (any admin
 * key set, or at least one client account exists). Open mode (fresh
 * install, zero setup) passes through untouched.
 */
import { useEffect, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { Database, Lock, Mail, User } from 'lucide-react'
import { ApiError, apiGet, apiPost, getToken, setToken } from '../api/client'
import { useApp } from '../store/app'

type Status = 'checking' | 'open' | 'locked' | 'authed'

export default function AuthGate({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<Status>('checking')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [mode, setMode] = useState<'signin' | 'signup' | 'forgot'>('signin')
  const [emailOn, setEmailOn] = useState(false)
  const [emailRequired, setEmailRequired] = useState(false)
  const [info, setInfo] = useState('')
  const [email, setEmail] = useState('')
  const [signupOpen, setSignupOpen] = useState(false)

  const probe = async () => {
    try {
      const h = await apiGet<{
        auth_required: boolean
        signup_enabled?: boolean
        email_enabled?: boolean
        email_required?: boolean
      }>('/api/health')
      setSignupOpen(Boolean(h.signup_enabled))
      setEmailOn(Boolean(h.email_enabled))
      setEmailRequired(Boolean(h.email_required))
      if (!h.auth_required) return setStatus('open')
      if (!getToken()) return setStatus('locked')
      // verify the stored token still works
      try {
        await apiGet('/api/datasets')
        setStatus('authed')
      } catch (e) {
        // Only a 401 means the token is no longer valid. A server error
        // used to send a signed-in user back to the login screen; let the
        // page show that error instead.
        setStatus(e instanceof ApiError && e.status === 401 ? 'locked' : 'authed')
      }
    } catch {
      // backend unreachable — let the app render its own errors
      setStatus('open')
    }
  }

  useEffect(() => {
    probe()
    const onExpired = () => {
      // Whoever signs in next may be a different client.
      useApp.getState().setDataset(null)
      setStatus('locked')
    }
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
      setInfo('')
      if (mode === 'forgot') {
        try {
          const r = await apiPost<{ message: string }>('/api/auth/forgot-password', {
            login: username,
          })
          setInfo(r.message)
        } catch (err) {
          setError(err instanceof Error ? err.message : String(err))
        } finally {
          setBusy(false)
        }
        return
      }
      try {
        const r = await apiPost<{ token: string }>(
          mode === 'signup' ? '/api/auth/signup' : '/api/auth/login',
          mode === 'signup' ? { username, password, email } : { username, password },
        )
        setToken(r.token)
        setStatus('authed')
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
      } finally {
        setBusy(false)
      }
    }
    return (
      <div className="flex h-full items-center justify-center px-4">
        <form
          onSubmit={submit}
          className="w-full max-w-sm rounded-2xl border border-edge bg-panel p-6"
        >
          <div className="mb-4 flex items-center gap-2">
            <Database className="h-6 w-6 text-accent" />
            <div className="text-sm font-bold">Analytiq</div>
          </div>
          <label className="mb-1 flex items-center gap-1.5 text-xs text-mute">
            <User className="h-3.5 w-3.5" /> {mode === 'forgot' ? 'Username or email' : 'Username'}
          </label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoFocus
            autoCapitalize="off"
            className="mb-3 w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink focus:border-accent focus:outline-none"
          />
          {mode !== 'forgot' && (
            <>
              <label className="mb-1 flex items-center gap-1.5 text-xs text-mute">
                <Lock className="h-3.5 w-3.5" /> Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink focus:border-accent focus:outline-none"
              />
            </>
          )}
          {mode === 'signin' && emailOn && (
            <button
              type="button"
              onClick={() => {
                setMode('forgot')
                setError('')
              }}
              className="mt-1.5 text-[11px] text-mute hover:text-accent"
            >
              Forgot password?
            </button>
          )}
          {mode === 'signup' && (
            <>
              <p className="mt-1 text-[11px] text-faint">At least 8 characters, not a common password.</p>
              <label className="mt-3 mb-1 flex items-center gap-1.5 text-xs text-mute">
                <Mail className="h-3.5 w-3.5" />{' '}
                {emailRequired
                  ? 'Email (we send a link to confirm it)'
                  : 'Email (for receipts and password help)'}
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink focus:border-accent focus:outline-none"
              />
            </>
          )}
          {error && <p className="mt-2 text-xs text-rose">{error}</p>}
          {info && <p className="mt-2 text-xs text-teal">{info}</p>}
          <button
            type="submit"
            disabled={busy || !username || (mode !== 'forgot' && !password)}
            className="mt-4 w-full rounded-lg bg-accent py-2 text-sm font-semibold text-white disabled:opacity-40"
          >
            {busy
              ? 'Please wait…'
              : mode === 'signup'
                ? 'Create free account'
                : mode === 'forgot'
                  ? 'Email me a reset link'
                  : 'Sign in'}
          </button>
          {mode === 'forgot' && (
            <button
              type="button"
              onClick={() => {
                setMode('signin')
                setInfo('')
                setError('')
              }}
              className="mt-3 w-full text-center text-xs text-accent hover:underline"
            >
              Back to sign in
            </button>
          )}
          {signupOpen && mode !== 'forgot' && (
            <button
              type="button"
              onClick={() => {
                setMode(mode === 'signup' ? 'signin' : 'signup')
                setError('')
              }}
              className="mt-3 w-full text-center text-xs text-accent hover:underline"
            >
              {mode === 'signup'
                ? 'Already have an account? Sign in'
                : 'New here? Create a free account — free credits every day'}
            </button>
          )}
          <div className="mt-4 flex justify-center gap-3 border-t border-edge pt-3 text-[11px] text-faint">
            <Link to="/pricing" className="hover:text-ink">Pricing</Link>
            <Link to="/terms" className="hover:text-ink">Terms</Link>
            <Link to="/privacy" className="hover:text-ink">Privacy</Link>
            <Link to="/contact" className="hover:text-ink">Contact</Link>
          </div>
          {mode === 'signup' && (
            <p className="mt-2 text-center text-[10px] text-faint">
              By creating an account you accept the Terms and Privacy Policy.
            </p>
          )}
        </form>
      </div>
    )
  }

  return <>{children}</>
}
