/**
 * The System page exists so the person who added the keys can find out
 * whether they work, on the machine that actually holds them. That makes
 * two things load-bearing: a failure must say what failed and what to do
 * about it, and a key must never appear on screen.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { render } from '../test/render'
import SystemPage from './SystemPage'
import * as client from '../api/client'

const status = {
  providers: {
    groq: {
      name: 'groq',
      label: 'Groq',
      configured: true,
      model: 'llama-3.3-70b-versatile',
      free: true,
      local: false,
      missing: '',
    },
    gemini: {
      name: 'gemini',
      label: 'Google Gemini',
      configured: false,
      model: 'gemini-3.6-flash',
      free: false,
      local: false,
      missing: 'GEMINI_API_KEY is not set',
    },
  },
  configured: ['groq'],
  routing: { narrative: 'groq', executive_summary: 'gemini' },
  order: ['groq', 'gemini'],
  privacy_mode: false,
  any_available: true,
}

describe('SystemPage', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.spyOn(client, 'apiGet').mockResolvedValue(status)
  })

  it('says which providers are configured before anything is called', async () => {
    render(<SystemPage />)
    // "Groq" also names the routing rows, so anchor on the model line
    // that only the provider card carries.
    expect(await screen.findByText('llama-3.3-70b-versatile')).toBeInTheDocument()
    expect(screen.getByText('gemini-3.6-flash')).toBeInTheDocument()
    // The unconfigured one names the variable to set — "not configured"
    // on its own is a support ticket, not a status.
    expect(screen.getByText(/GEMINI_API_KEY is not set/)).toBeInTheDocument()
  })

  it('shows that reports still work when no model is available', async () => {
    vi.spyOn(client, 'apiGet').mockResolvedValue({
      ...status,
      configured: [],
      any_available: false,
    })
    render(<SystemPage />)
    expect(await screen.findByText('Engines')).toBeInTheDocument()
    expect(
      screen.getByText(/written by the analysis engines/),
    ).toBeInTheDocument()
  })

  it('reports a rejected key with the provider’s own error and a next step', async () => {
    const post = vi.spyOn(client, 'apiPost').mockResolvedValue({
      checked_at: '2026-09-02T10:00:00+00:00',
      working: [],
      any_working: false,
      summary: '2 provider(s) are configured but none answered.',
      privacy_mode: false,
      providers: [
        {
          ...status.providers.groq,
          ok: false,
          latency_ms: 240,
          reply: '',
          error: 'HTTP 401 — Invalid API Key',
          hint: 'The key in GROQ_API_KEY was rejected.',
        },
      ],
    })
    render(<SystemPage />)
    await screen.findByText('llama-3.3-70b-versatile')
    await userEvent.click(screen.getByRole('button', { name: /run live check/i }))

    await waitFor(() =>
      expect(screen.getByText(/HTTP 401 — Invalid API Key/)).toBeInTheDocument(),
    )
    expect(screen.getByText(/was rejected/)).toBeInTheDocument()
    expect(post).toHaveBeenCalledWith('/api/admin/llm-check')
  })

  it('confirms a working provider with its latency', async () => {
    vi.spyOn(client, 'apiPost').mockResolvedValue({
      checked_at: '2026-09-02T10:00:00+00:00',
      working: ['groq'],
      any_working: true,
      summary: 'Working: Groq.',
      privacy_mode: false,
      providers: [
        {
          ...status.providers.groq,
          ok: true,
          latency_ms: 412,
          reply: 'ready',
          error: '',
          hint: '',
        },
      ],
    })
    render(<SystemPage />)
    await screen.findByText('llama-3.3-70b-versatile')
    await userEvent.click(screen.getByRole('button', { name: /run live check/i }))

    expect(await screen.findByText('Working: Groq.')).toBeInTheDocument()
    expect(screen.getByText(/answered in 412 ms/)).toBeInTheDocument()
  })

  it('still names the variable to set after a check has run', async () => {
    // The check result carries `error`/`hint` rather than `missing`, so
    // this is where the advice used to disappear the moment the button
    // was pressed.
    vi.spyOn(client, 'apiPost').mockResolvedValue({
      checked_at: '2026-09-02T10:00:00+00:00',
      working: [],
      any_working: false,
      summary: 'No LLM provider is configured.',
      privacy_mode: false,
      providers: [
        {
          ...status.providers.gemini,
          ok: false,
          latency_ms: 0,
          reply: '',
          error: 'GEMINI_API_KEY is not set',
          hint: 'Add GEMINI_API_KEY to the environment.',
        },
      ],
    })
    render(<SystemPage />)
    await screen.findByText('llama-3.3-70b-versatile')
    await userEvent.click(screen.getByRole('button', { name: /run live check/i }))

    expect(
      await screen.findByText('GEMINI_API_KEY is not set'),
    ).toBeInTheDocument()
    expect(
      screen.getByText('Add GEMINI_API_KEY to the environment.'),
    ).toBeInTheDocument()
  })

  it('surfaces the fetch error rather than rendering an empty page', async () => {
    vi.spyOn(client, 'apiGet').mockRejectedValue(new Error('403 Forbidden'))
    render(<SystemPage />)
    expect(await screen.findByText(/403 Forbidden/)).toBeInTheDocument()
  })
})
