/**
 * The delivery panel hands out a link that opens a report with no
 * sign-in. Two things it must get right, and both are here: the request
 * it sends, and what it tells the person about what they are handing
 * over. A share control that does not say when the link stops working,
 * or that offers no way back, is how a report ends up circulating a year
 * after the data behind it moved on.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { render } from '../test/render'
import Delivery from './Delivery'
import * as client from '../api/client'

const DAY = 86400

const artifact = {
  artifact_id: 'a1',
  kind: 'health-report',
  format: 'pdf',
  filename: 'health.pdf',
  title: 'Health report',
  size_bytes: 240_000,
  created_at: Date.now() / 1000 - 60,
  expires_at: Date.now() / 1000 + 30 * DAY,
  shared: false,
  share_views: 0,
  shares_revoked: false,
}

function mockGets(artifacts: unknown[], schedules: unknown[] = []) {
  return vi.spyOn(client, 'apiGet').mockImplementation((path: string) =>
    Promise.resolve(
      path.startsWith('/api/artifacts') ? { artifacts } : { schedules },
    ) as never,
  )
}

beforeEach(() => {
  vi.restoreAllMocks()
})

describe('reports that were already built', () => {
  it('says when each one will be removed, not just when it was made', async () => {
    mockGets([artifact])
    render(<Delivery datasetId="ds1" />)
    expect(await screen.findByText(/removed in 30 days/)).toBeTruthy()
  })

  it('offers somewhere to start rather than an empty box', async () => {
    mockGets([])
    render(<Delivery datasetId="ds1" />)
    expect(await screen.findByText(/Nothing built yet/)).toBeTruthy()
  })

  it('does not crash the page when the response is not what was expected', async () => {
    // A proxy error page, a half-finished deploy. The Reports page above
    // this panel is what the user actually came for.
    vi.spyOn(client, 'apiGet').mockResolvedValue({ oops: true } as never)
    render(<Delivery datasetId="ds1" />)
    expect(await screen.findByText(/Nothing built yet/)).toBeTruthy()
  })
})

describe('the share link', () => {
  it('asks for one, and says what it grants and when it lapses', async () => {
    mockGets([artifact])
    const post = vi.spyOn(client, 'apiPost').mockResolvedValue({
      url: 'http://x/api/shared/tok',
      expires_at: Date.now() / 1000 + 7 * DAY,
    } as never)

    const user = userEvent.setup()
    render(<Delivery datasetId="ds1" />)
    await user.click(await screen.findByRole('button', { name: /Get a link/ }))

    await waitFor(() => expect(post).toHaveBeenCalled())
    expect(post.mock.calls[0][0]).toBe('/api/artifacts/a1/share')

    expect(screen.getByText('http://x/api/shared/tok')).toBeTruthy()
    // The three things the person handing it over has to know.
    const note = screen.getByText(/no sign-in/)
    expect(note.textContent).toMatch(/stops working in 7 days/)
    expect(note.textContent).toMatch(/grants nothing else/)
    expect(note.textContent).toMatch(/withdraw it/)
  })

  it('offers withdrawing it on a report that is already shared', async () => {
    mockGets([{ ...artifact, shared: true, share_views: 3 }])
    const del = vi.spyOn(client, 'apiDelete').mockResolvedValue({} as never)

    const user = userEvent.setup()
    render(<Delivery datasetId="ds1" />)
    expect(await screen.findByText(/shared · 3 views/)).toBeTruthy()

    await user.click(screen.getByRole('button', { name: /Withdraw/ }))
    await waitFor(() =>
      expect(del).toHaveBeenCalledWith('/api/artifacts/a1/share'),
    )
  })

  it('does not offer to withdraw a link that was never made', async () => {
    mockGets([artifact])
    render(<Delivery datasetId="ds1" />)
    await screen.findByRole('button', { name: /Get a link/ })
    expect(screen.queryByRole('button', { name: /Withdraw/ })).toBeNull()
  })
})

describe('schedules', () => {
  it('sends the cadence, hour and recipients the user chose', async () => {
    mockGets([])
    const post = vi.spyOn(client, 'apiPost').mockResolvedValue({} as never)

    const user = userEvent.setup()
    render(<Delivery datasetId="ds1" />)

    await user.selectOptions(await screen.findByLabelText('How often'), 'monthly')
    await user.selectOptions(screen.getByLabelText('At'), '9')
    await user.type(screen.getByLabelText(/Email it to/), 'cfo@client.com')
    await user.click(screen.getByRole('button', { name: /Add schedule/ }))

    await waitFor(() => expect(post).toHaveBeenCalled())
    expect(post.mock.calls[0][1]).toMatchObject({
      dataset_id: 'ds1',
      cadence: 'monthly',
      hour: 9,
      recipients: ['cfo@client.com'],
    })
  })

  it('shows a schedule with no recipients as waiting here, not as sending', async () => {
    mockGets([], [
      {
        schedule_id: 's1',
        dataset_id: 'ds1',
        kind: 'health-report',
        cadence: 'weekly',
        hour: 7,
        recipients: [],
        enabled: true,
        last_run_at: 0,
        last_status: '',
        last_artifact_id: '',
        next_run_at: Date.now() / 1000 + 2 * DAY,
      },
    ])
    render(<Delivery datasetId="ds1" />)
    expect(await screen.findByText('waits here')).toBeTruthy()
  })

  it('leaves another dataset’s schedules off this dataset’s page', async () => {
    mockGets([], [
      {
        schedule_id: 's2',
        dataset_id: 'other',
        kind: 'report',
        cadence: 'daily',
        hour: 7,
        recipients: [],
        enabled: true,
        last_run_at: 0,
        last_status: '',
        last_artifact_id: '',
        next_run_at: Date.now() / 1000 + DAY,
      },
    ])
    render(<Delivery datasetId="ds1" />)
    await screen.findByRole('button', { name: /Add schedule/ })
    expect(screen.queryByText('waits here')).toBeNull()
  })
})
