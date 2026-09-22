/**
 * The page a customer lands on when they want to know what they are
 * paying for. Two things carry real consequences: the usage meters have
 * to warn before a ceiling bites rather than after, and a deployment
 * that cannot take money must say so instead of offering a dead button.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { render } from '../test/render'
import AccountPage from './AccountPage'
import * as client from '../api/client'

const account = {
  username: 'amy',
  email: 'amy@example.com',
  is_admin: false,
  plan: 'free',
  plan_label: 'Free',
  branding: true,
  limits: {
    datasets: 3,
    rows_per_dataset: 25000,
    max_upload_mb: 10,
    reports_per_month: 3,
    models_per_month: 2,
    scheduled_reports: 0,
    share_links: 3,
  },
  used: { datasets: 1, reports: 1, models: 0 },
  remaining: { datasets: 2, reports_per_month: 2, models_per_month: 2 },
  metered: true,
}

const catalogue = {
  plans: [
    {
      key: 'free',
      label: 'Free',
      price_inr_month: 0,
      datasets: 3,
      rows_per_dataset: 25000,
      max_upload_mb: 10,
      reports_per_month: 3,
      scheduled_reports: 0,
      branding: true,
      notes: 'Enough to judge the product on.',
    },
    {
      key: 'solo',
      label: 'Solo',
      price_inr_month: 2400,
      datasets: 100,
      rows_per_dataset: 2000000,
      max_upload_mb: 200,
      reports_per_month: null,
      scheduled_reports: 10,
      branding: false,
      notes: 'For one freelancer and their clients.',
    },
  ],
  checkout_available: true,
  metered: true,
  currency: 'INR',
}

function mockApi(
  over: { account?: unknown; catalogue?: unknown } = {},
) {
  vi.spyOn(client, 'apiGet').mockImplementation((path: string) => {
    if (path === '/api/account')
      return Promise.resolve((over.account ?? account) as never)
    if (path === '/api/billing/plans')
      return Promise.resolve((over.catalogue ?? catalogue) as never)
    return Promise.reject(new Error(`unexpected GET ${path}`))
  })
}

beforeEach(() => {
  vi.restoreAllMocks()
})

describe('what the account is on', () => {
  it('names the plan and who it belongs to', async () => {
    mockApi()
    render(<AccountPage />)

    expect(await screen.findByText('amy')).toBeTruthy()
    expect(screen.getByText('amy@example.com')).toBeTruthy()
    expect(screen.getAllByText('Free').length).toBeGreaterThan(0)
  })

  it('does not advertise the Analytiq mark as a feature', async () => {
    /* plans.branding is true when the deliverable carries OUR name —
       the one thing a freelancer pays to remove — so the paid tiers are
       the false ones. Read the intuitive way round, this row told every
       customer the exact opposite of what they were buying. */
    mockApi()
    render(<AccountPage />)

    await screen.findByText('amy')
    expect(screen.getByText('Analytiq mark')).toBeTruthy()
    expect(screen.getByText('Your name only')).toBeTruthy()
  })

  it('marks which plan in the list is the current one', async () => {
    mockApi()
    render(<AccountPage />)

    expect(await screen.findByText('Your plan')).toBeTruthy()
  })
})

describe('usage against the ceiling', () => {
  it('shows what has been used and what the cap is', async () => {
    mockApi()
    render(<AccountPage />)

    const meter = await screen.findByRole('meter', { name: /Datasets/ })
    expect(meter.getAttribute('aria-valuenow')).toBe('1')
    expect(meter.getAttribute('aria-valuemax')).toBe('3')
  })

  it('warns at four fifths, before the ceiling actually bites', async () => {
    /* Learning about a quota at the moment it refuses an upload is a bad
       way to find out what you bought. Four of five reports is amber. */
    mockApi({
      account: {
        ...account,
        limits: { ...account.limits, reports_per_month: 5 },
        used: { datasets: 1, reports: 4, models: 0 },
      },
    })
    render(<AccountPage />)

    const bar = (
      await screen.findByRole('meter', { name: /Reports this month/ })
    ).firstElementChild as HTMLElement

    expect(bar.className).toContain('bg-amber')
    expect(bar.className).not.toContain('bg-accent')
  })

  it('goes red once the ceiling is reached', async () => {
    mockApi({
      account: { ...account, used: { datasets: 1, reports: 3, models: 0 } },
    })
    render(<AccountPage />)

    const bar = (
      await screen.findByRole('meter', { name: /Reports this month/ })
    ).firstElementChild as HTMLElement

    expect(bar.className).toContain('bg-rose')
  })

  it('is calm well below the ceiling', async () => {
    mockApi({
      account: { ...account, used: { datasets: 1, reports: 1, models: 0 } },
    })
    render(<AccountPage />)

    const bar = (
      await screen.findByRole('meter', { name: /Reports this month/ })
    ).firstElementChild as HTMLElement

    expect(bar.className).toContain('bg-accent')
  })

  it('does not draw a bar for a plan with no ceiling', async () => {
    mockApi({
      account: {
        ...account,
        plan: 'unlimited',
        plan_label: 'Unlimited',
        limits: { ...account.limits, datasets: null },
      },
    })
    render(<AccountPage />)

    await screen.findByText('amy')
    expect(screen.queryByRole('meter', { name: /Datasets/ })).toBeNull()
  })

  it('says nothing is capped on a self-hosted deployment', async () => {
    /* Capping the container its own operator pays for serves nobody. */
    mockApi({ account: { ...account, metered: false } })
    render(<AccountPage />)

    expect(
      await screen.findByText(/Every ceiling is off/),
    ).toBeTruthy()
    expect(screen.queryByRole('meter')).toBeNull()
  })
})

describe('moving to a paid plan', () => {
  it('sends the plan key and follows the checkout url', async () => {
    mockApi()
    const post = vi
      .spyOn(client, 'apiPost')
      .mockResolvedValue({ url: 'https://checkout.stripe.test/s/1' } as never)

    // jsdom refuses a real navigation, so watch the assignment instead.
    const href = vi.fn()
    Object.defineProperty(window, 'location', {
      configurable: true,
      value: {
        get href() {
          return ''
        },
        set href(v: string) {
          href(v)
        },
      },
    })

    render(<AccountPage />)
    await userEvent.click(await screen.findByText(/Move to Solo/))

    await waitFor(() =>
      expect(post).toHaveBeenCalledWith('/api/billing/checkout', {
        plan: 'solo',
      }),
    )
    expect(href).toHaveBeenCalledWith('https://checkout.stripe.test/s/1')
  })

  it('says so rather than offering a dead button when payments are off', async () => {
    mockApi({ catalogue: { ...catalogue, checkout_available: false } })
    render(<AccountPage />)

    expect(
      await screen.findByText(/Payments are not set up on this deployment/),
    ).toBeTruthy()
    const btn = screen.getByText(/Move to Solo/).closest('button')
    expect(btn?.disabled).toBe(true)
  })

  it('surfaces a checkout failure instead of silently doing nothing', async () => {
    mockApi()
    vi.spyOn(client, 'apiPost').mockRejectedValue(
      new Error('Payments are not set up on this deployment.'),
    )

    render(<AccountPage />)
    await userEvent.click(await screen.findByText(/Move to Solo/))

    expect(
      await screen.findByText(/Payments are not set up on this deployment./),
    ).toBeTruthy()
  })
})

describe('when the account itself will not load', () => {
  it('reports the failure rather than showing an empty page', async () => {
    vi.spyOn(client, 'apiGet').mockRejectedValue(new Error('Not authenticated'))
    render(<AccountPage />)

    expect(await screen.findByText(/Not authenticated/)).toBeTruthy()
  })

  it('still shows the plan when only the catalogue is unreachable', async () => {
    /* The catalogue is the upgrade path, not the account. Losing it must
       not take away the answer to "what am I on?". */
    vi.spyOn(client, 'apiGet').mockImplementation((path: string) => {
      if (path === '/api/account') return Promise.resolve(account as never)
      return Promise.reject(new Error('billing is down'))
    })

    render(<AccountPage />)

    expect(await screen.findByText('amy')).toBeTruthy()
    expect(screen.getByText(/Loading plans/)).toBeTruthy()
  })
})

describe('changing a password', () => {
  it('sends both passwords and confirms', async () => {
    mockApi()
    const post = vi.spyOn(client, 'apiPost').mockResolvedValue({} as never)

    render(<AccountPage />)
    await screen.findByText('amy')

    await userEvent.type(
      screen.getByLabelText(/Current password/),
      'old-one-123',
    )
    await userEvent.type(screen.getByLabelText(/New password/), 'new-one-456')
    await userEvent.click(screen.getByText(/Change password/))

    await waitFor(() =>
      expect(post).toHaveBeenCalledWith('/api/account/password', {
        current_password: 'old-one-123',
        new_password: 'new-one-456',
      }),
    )
    expect(await screen.findByText(/Password changed/)).toBeTruthy()
  })

  it('will not submit half a form', async () => {
    mockApi()
    render(<AccountPage />)
    await screen.findByText('amy')

    const btn = screen.getByText(/Change password/).closest('button')
    expect(btn?.disabled).toBe(true)
  })
})
