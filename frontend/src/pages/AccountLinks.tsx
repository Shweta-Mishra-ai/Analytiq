/**
 * Pages opened from links in emails: confirming an address and choosing a
 * new password. Reachable without signing in.
 */
import { useEffect, useRef, useState, type FormEvent } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { apiPost, setToken } from '../api/client'
import { PublicShell } from './PublicPages'

export function VerifyEmailPage() {
  const [params] = useSearchParams()
  const [state, setState] = useState<'working' | 'done' | 'failed'>('working')
  const [message, setMessage] = useState('')
  const sent = useRef(false)
  useEffect(() => {
    if (sent.current) return
    sent.current = true
    apiPost('/api/auth/verify-email', { token: params.get('token') ?? '' })
      .then(() => setState('done'))
      .catch((e) => {
        setState('failed')
        setMessage((e as Error).message)
      })
  }, [params])
  return (
    <PublicShell title="Confirm your email">
      {state === 'working' && <p>Confirming…</p>}
      {state === 'done' && (
        <p data-testid="verified">
          Your email address is confirmed and your free daily credits are switched on.{' '}
          <Link to="/" className="font-semibold text-accent">
            Go to the app
          </Link>
        </p>
      )}
      {state === 'failed' && <p className="text-rose">{message}</p>}
    </PublicShell>
  )
}

export function ResetPasswordPage() {
  const [params] = useSearchParams()
  const [password, setPassword] = useState('')
  const [again, setAgain] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (password !== again) return setError('The two passwords are not the same.')
    setBusy(true)
    setError('')
    try {
      const r = await apiPost<{ token: string }>('/api/auth/reset-password', {
        token: params.get('token') ?? '',
        password,
      })
      setToken(r.token)
      window.location.href = '/'
    } catch (err) {
      setError((err as Error).message)
      setBusy(false)
    }
  }
  const field =
    'w-full max-w-sm rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink focus:border-accent focus:outline-none'
  return (
    <PublicShell title="Choose a new password">
      <form onSubmit={submit} className="space-y-3">
        <input type="password" placeholder="New password (8+ characters)" value={password}
          onChange={(e) => setPassword(e.target.value)} className={field} autoFocus />
        <input type="password" placeholder="The same again" value={again}
          onChange={(e) => setAgain(e.target.value)} className={field} />
        {error && <p className="text-rose">{error}</p>}
        <button type="submit" disabled={busy || !password}
          className="btn-primary rounded-lg px-4 py-2 text-sm font-semibold text-white disabled:opacity-40">
          {busy ? 'Saving…' : 'Save and sign in'}
        </button>
      </form>
    </PublicShell>
  )
}
