import { afterEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import BillingPage from '../pages/BillingPage'
import AuthGate from '../components/AuthGate'
import { OUT_OF_CREDITS_EVENT, apiPost } from '../api/client'

type Reply = unknown | ((body: unknown) => [number, unknown])

function serve(routes: Record<string, Reply>) {
  const calls: { key: string; body: unknown }[] = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      const key = `${init?.method ?? 'GET'} ${url}`
      const body = init?.body ? JSON.parse(String(init.body)) : undefined
      calls.push({ key, body })
      const r = routes[key]
      const [status, out] =
        typeof r === 'function' ? (r as (b: unknown) => [number, unknown])(body) : r ? [200, r] : [404, { detail: 'not found' }]
      return new Response(JSON.stringify(out), { status, headers: { 'Content-Type': 'application/json' } })
    }),
  )
  return calls
}

afterEach(() => vi.unstubAllGlobals())

const overview = (extra: Record<string, unknown> = {}) => ({
  balance: { unlimited: false, free_left: 0, free_daily: 0, paid: 0, total: 0, needs_verification: true },
  history: [],
  recovered: 0,
  packs: [{ id: 'starter', name: 'Starter', credits: 50, inr: 19900, usd: 500 }],
  costs: [{ action: 'pdf_report', label: 'PDF report', credits: 2 }],
  free_daily: 5,
  providers: { razorpay: false, stripe: false, paypal: false, upi: true },
  stripe_currency: 'USD',
  invoices: [],
  upi: { id: 'shop@okaxis', payee: 'Shop', links: { starter: 'upi://pay?pa=shop%40okaxis' } },
  upi_requests: [],
  ...extra,
})

describe('BillingPage', () => {
  it('asks an unverified account to confirm its email, and takes a UPI reference', async () => {
    const calls = serve({
      'GET /api/billing': overview(),
      'POST /api/billing/upi': { id: 'x', status: 'pending' },
    })
    render(
      <MemoryRouter>
        <BillingPage />
      </MemoryRouter>,
    )
    expect(await screen.findByText('Confirm your email to switch these on')).toBeInTheDocument()
    expect(screen.getByTestId('upi-id')).toHaveTextContent('shop@okaxis')
    const button = screen.getByRole('button', { name: /I have paid/ })
    expect(button).toBeDisabled()
    fireEvent.change(screen.getByPlaceholderText(/12 digits/), { target: { value: '4123 4567 8901' } })
    expect(button).toBeEnabled()
    fireEvent.click(button)
    await screen.findByText(/We will check the payment/)
    const sent = calls.find((c) => c.key === 'POST /api/billing/upi')
    expect(sent?.body).toEqual({ pack: 'starter', utr: '4123 4567 8901' })
  })

  it('lists invoices and shows a rejected UPI payment with its reason', async () => {
    serve({
      'GET /api/billing': overview({
        invoices: [{ number: 'AQ-2627-00001', issued_at: 1795000000, amount: 19900, currency: 'INR', credits: 50, tax_invoice: true }],
        upi_requests: [{ id: 'a', pack: 'starter', credits: 50, utr: '412345678901', status: 'rejected', created: 1, note: 'Payment not found' }],
      }),
    })
    render(
      <MemoryRouter>
        <BillingPage />
      </MemoryRouter>,
    )
    expect(await screen.findByText('AQ-2627-00001')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Tax invoice PDF' })).toBeInTheDocument()
    expect(screen.getByText('Payment not found')).toBeInTheDocument()
  })
})

describe('out of credits', () => {
  it('a 402 answer raises the out-of-credits event with the server message', async () => {
    serve({ 'POST /api/reports/x/pdf': () => [402, { detail: 'This needs 2 credits and you have 1.' }] })
    const seen: string[] = []
    const on = (e: Event) => seen.push((e as CustomEvent<string>).detail)
    window.addEventListener(OUT_OF_CREDITS_EVENT, on)
    await expect(apiPost('/api/reports/x/pdf')).rejects.toThrow('This needs 2 credits')
    window.removeEventListener(OUT_OF_CREDITS_EVENT, on)
    expect(seen).toEqual(['This needs 2 credits and you have 1.'])
  })
})

describe('AuthGate', () => {
  const health = { auth_required: true, signup_enabled: true, email_enabled: true, email_required: true }

  it('signs up with an email and stores the token', async () => {
    const calls = serve({
      'GET /api/health': health,
      'POST /api/auth/signup': { token: 'tok-1', username: 'newbie' },
    })
    render(
      <MemoryRouter>
        <AuthGate>
          <div>inside the app</div>
        </AuthGate>
      </MemoryRouter>,
    )
    fireEvent.click(await screen.findByText(/New here\?/))
    expect(screen.getByText(/we send a link to confirm it/)).toBeInTheDocument()
    const inputs = document.querySelectorAll('form input')
    fireEvent.change(inputs[0], { target: { value: 'newbie' } })
    fireEvent.change(inputs[1], { target: { value: 'a-good-passphrase' } })
    fireEvent.change(inputs[2], { target: { value: 'n@example.com' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create free account' }))
    expect(await screen.findByText('inside the app')).toBeInTheDocument()
    expect(localStorage.getItem('analytiq-token')).toBe('tok-1')
    expect(calls.find((c) => c.key === 'POST /api/auth/signup')?.body).toEqual({
      username: 'newbie',
      password: 'a-good-passphrase',
      email: 'n@example.com',
    })
  })

  it('asks for a reset link without a password and shows the neutral answer', async () => {
    serve({
      'GET /api/health': health,
      'POST /api/auth/forgot-password': { message: 'If an account exists, a link is on its way.' },
    })
    render(
      <MemoryRouter>
        <AuthGate>
          <div>inside</div>
        </AuthGate>
      </MemoryRouter>,
    )
    fireEvent.click(await screen.findByText('Forgot password?'))
    expect(screen.queryByText('Password')).not.toBeInTheDocument()
    fireEvent.change(document.querySelector('form input')!, { target: { value: 'me@example.com' } })
    fireEvent.click(screen.getByRole('button', { name: 'Email me a reset link' }))
    await waitFor(() => expect(screen.getByText(/a link is on its way/)).toBeInTheDocument())
  })
})
