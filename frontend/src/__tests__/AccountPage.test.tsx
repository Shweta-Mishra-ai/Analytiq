import { afterEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import AccountPage from '../pages/AccountPage'
import ErrorBoundary from '../components/ErrorBoundary'

const account = (connector: unknown) => ({
  username: 'acme',
  has_password: true,
  usage: { used_mb: 120, datasets: 3, quota_mb: 1024, max_datasets: 50 },
  max_file_mb: 50,
  data_ttl_days: 30,
  support_email: '',
  connector,
})

function serve(routes: Record<string, unknown>) {
  const calls: string[] = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      const key = `${init?.method ?? 'GET'} ${url}`
      calls.push(key)
      const body = routes[key]
      return new Response(JSON.stringify(body ?? { detail: 'not found' }), {
        status: body ? 200 : 404,
        headers: { 'Content-Type': 'application/json' },
      })
    }),
  )
  return calls
}

afterEach(() => vi.unstubAllGlobals())

describe('AccountPage', () => {
  it('shows storage against the limit and when data is deleted', async () => {
    serve({ 'GET /api/account': account(null) })
    render(<AccountPage />)
    expect(await screen.findByTestId('usage')).toHaveTextContent('120 MB used of 1,024 MB')
    expect(screen.getByText(/deleted 30 days after/)).toBeInTheDocument()
    expect(screen.queryByTestId('connector')).toBeNull()
  })

  it('creates a Claude connector key and shows it once with the address', async () => {
    const calls = serve({
      'GET /api/account': account({ url: 'https://a.example/mcp/', has_key: false, key_created_at: null }),
      'POST /api/account/connector-key': {
        key: 'aqk-YWNtZQ-secret',
        connector: { url: 'https://a.example/mcp/', has_key: true, key_created_at: 1_700_000_000 },
      },
    })
    render(<AccountPage />)
    const panel = await screen.findByTestId('connector')
    expect(panel).toHaveTextContent('https://a.example/mcp/')
    expect(panel).toHaveTextContent('No key yet.')
    fireEvent.click(screen.getByText('Create key'))
    await waitFor(() => expect(panel).toHaveTextContent('Bearer aqk-YWNtZQ-secret'))
    expect(panel).toHaveTextContent('not shown again')
    expect(screen.getByText('Replace key')).toBeInTheDocument()
    expect(calls).toContain('POST /api/account/connector-key')
  })

  it('refuses a new password typed two different ways without calling the server', async () => {
    const calls = serve({ 'GET /api/account': account(null) })
    render(<AccountPage />)
    await screen.findByTestId('usage')
    fireEvent.change(screen.getByPlaceholderText('Current password'), { target: { value: 'old-secret-1' } })
    fireEvent.change(screen.getByPlaceholderText(/New password \(8/), { target: { value: 'new-secret-1' } })
    fireEvent.change(screen.getByPlaceholderText('New password again'), { target: { value: 'new-secret-2' } })
    fireEvent.click(screen.getByRole('button', { name: 'Change password' }))
    expect(await screen.findByText('The two new passwords are not the same.')).toBeInTheDocument()
    expect(calls.filter((c) => c.includes('change-password'))).toHaveLength(0)
  })
})

describe('ErrorBoundary', () => {
  it('shows a message instead of a blank screen when a page throws', () => {
    const Broken = (): never => {
      throw new Error('boom')
    }
    const quiet = vi.spyOn(console, 'error').mockImplementation(() => {})
    render(
      <ErrorBoundary>
        <Broken />
      </ErrorBoundary>,
    )
    expect(screen.getByText('This screen could not be shown')).toBeInTheDocument()
    quiet.mockRestore()
  })
})
