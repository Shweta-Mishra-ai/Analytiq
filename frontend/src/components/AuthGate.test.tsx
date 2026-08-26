/**
 * The gate decides whether one client can see another's data. Its
 * failure modes are asymmetric: showing a login screen to someone who
 * did not need one is an annoyance, letting the app through when the
 * server wants auth is a breach.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import AuthGate from './AuthGate'
import * as client from '../api/client'

const APP = 'the app'
const app = () => (
  <AuthGate>
    <div>{APP}</div>
  </AuthGate>
)

beforeEach(() => client.setToken(''))

function stubHealth(authRequired: boolean, datasetsOk = true) {
  return vi.spyOn(client, 'apiGet').mockImplementation(async (path: string) => {
    if (path === '/api/health') return { auth_required: authRequired }
    if (path === '/api/datasets') {
      if (!datasetsOk) throw new Error('Not authenticated')
      return { datasets: [] }
    }
    throw new Error(`unexpected ${path}`)
  })
}

describe('open mode', () => {
  it('lets a fresh install through with no login at all', async () => {
    stubHealth(false)
    render(app())
    expect(await screen.findByText(APP)).toBeInTheDocument()
  })
})

describe('locked mode', () => {
  it('asks for a login when the server requires one and no token is held', async () => {
    stubHealth(true)
    render(app())
    expect(
      await screen.findByRole('button', { name: 'Sign in' }),
    ).toBeInTheDocument()
    expect(screen.queryByText(APP)).toBeNull()
  })

  it('does not trust a stored token the server rejects', async () => {
    // A token surviving in localStorage past its expiry is the normal
    // case, not an edge case.
    client.setToken('stale')
    stubHealth(true, false)
    render(app())
    expect(
      await screen.findByRole('button', { name: 'Sign in' }),
    ).toBeInTheDocument()
  })

  it('lets a valid stored token straight through', async () => {
    client.setToken('good')
    stubHealth(true, true)
    render(app())
    expect(await screen.findByText(APP)).toBeInTheDocument()
  })
})

describe('signing in', () => {
  it('stores the token and reveals the app', async () => {
    stubHealth(true)
    vi.spyOn(client, 'apiPost').mockResolvedValue({ token: 'fresh-token' })
    const user = userEvent.setup()
    render(app())

    await user.type(await screen.findByLabelText(/Username/), 'acme')
    await user.type(screen.getByLabelText(/Password/), 'password123')
    await user.click(screen.getByRole('button', { name: 'Sign in' }))

    expect(await screen.findByText(APP)).toBeInTheDocument()
    expect(client.getToken()).toBe('fresh-token')
  })

  it('keeps the form up and shows why when the password is wrong', async () => {
    stubHealth(true)
    vi.spyOn(client, 'apiPost').mockRejectedValue(
      new Error('Invalid username or password'),
    )
    const user = userEvent.setup()
    render(app())

    await user.type(await screen.findByLabelText(/Username/), 'acme')
    await user.type(screen.getByLabelText(/Password/), 'wrong')
    await user.click(screen.getByRole('button', { name: 'Sign in' }))

    expect(
      await screen.findByText('Invalid username or password'),
    ).toBeInTheDocument()
    expect(screen.queryByText(APP)).toBeNull()
    expect(client.getToken()).toBe('')
  })

  it('will not submit an empty form', async () => {
    stubHealth(true)
    render(app())
    expect(await screen.findByRole('button', { name: 'Sign in' })).toBeDisabled()
  })

  it('masks the password and never echoes it back as visible text', async () => {
    // React does write `value` onto a controlled input's attribute, so
    // the guarantee is masking plus not repeating it anywhere else — not
    // that the string is absent from the markup.
    stubHealth(true)
    const user = userEvent.setup()
    render(app())
    const field = await screen.findByLabelText(/Password/)
    await user.type(field, 'password123')
    expect(field).toHaveAttribute('type', 'password')
    expect(document.body.textContent).not.toContain('password123')
  })

  it('labels both fields so a screen reader can tell them apart', async () => {
    stubHealth(true)
    render(app())
    expect(await screen.findByLabelText(/Username/)).toHaveAttribute(
      'type',
      'text',
    )
    expect(screen.getByLabelText(/Password/)).toHaveAttribute(
      'type',
      'password',
    )
  })
})

describe('a session that expires while in use', () => {
  it('returns to the login screen when a call reports 401', async () => {
    client.setToken('good')
    stubHealth(true, true)
    render(app())
    await screen.findByText(APP)

    // Any api call hitting a 401 fires this; the gate must react.
    window.dispatchEvent(new Event('analytiq-unauthorized'))
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Sign in' })).toBeInTheDocument(),
    )
  })
})

describe('an unreachable backend', () => {
  it('renders the app so its own error handling can show the problem', async () => {
    vi.spyOn(client, 'apiGet').mockRejectedValue(new Error('fetch failed'))
    render(app())
    // A login form here would be a lie — the server never said auth was
    // required, it never answered at all.
    expect(await screen.findByText(APP)).toBeInTheDocument()
  })
})
