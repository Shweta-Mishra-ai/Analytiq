/**
 * The rule this table has to render correctly: a task can only be
 * pointed at a model that can do it. The dropdown must therefore offer
 * exactly what the server said is eligible — no more, and derived
 * nowhere else.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { render } from '../test/render'
import TaskRouting from './TaskRouting'
import * as client from '../api/client'

const models = [
  {
    id: 'groq/llama-3.1-8b-instant',
    provider: 'groq',
    label: 'Llama 3.1 8B Instant',
    capabilities: ['text', 'json'],
    tier: 'fast',
    context: 128000,
    free: true,
    declared: false,
    notes: '',
  },
  {
    id: 'gemini/gemini-3.6-flash',
    provider: 'gemini',
    label: 'Gemini 3.6 Flash',
    capabilities: ['text', 'json', 'vision'],
    tier: 'deep',
    context: 1000000,
    free: false,
    declared: false,
    notes: '',
  },
]

const base = {
  models,
  problems: [],
  deprecated: [],
  capabilities: {
    text: 'Writes prose',
    json: 'Returns structured JSON',
    vision: 'Reads images',
  },
  tasks: [
    {
      task: 'chart_caption',
      label: 'Chart captions',
      description: 'One sentence per chart.',
      requires: ['text'],
      min_context: 8000,
      degrades_to: 'The engine writes its own wording.',
      assigned: 'groq/llama-3.1-8b-instant',
      source: 'default',
      resolved: ['groq/llama-3.1-8b-instant', 'gemini/gemini-3.6-flash'],
      eligible: ['groq/llama-3.1-8b-instant', 'gemini/gemini-3.6-flash'],
      served: true,
    },
    {
      task: 'table_extraction',
      label: 'Tables from photos',
      description: 'Reads a photograph of a table.',
      requires: ['vision', 'json'],
      min_context: 0,
      degrades_to: 'The upload is refused, naming the missing capability.',
      assigned: 'gemini/gemini-3.6-flash',
      source: 'runtime',
      resolved: ['gemini/gemini-3.6-flash'],
      eligible: ['gemini/gemini-3.6-flash'],
      served: true,
    },
  ],
}

describe('TaskRouting', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.spyOn(client, 'apiGet').mockResolvedValue(base)
  })

  it('offers only the models the server said can do the job', async () => {
    render(<TaskRouting />)
    const select = await screen.findByLabelText(/model for tables from photos/i)
    const options = within(select as HTMLElement)
      .getAllByRole('option')
      .map((o) => (o as HTMLOptionElement).value)
      .filter(Boolean)
    // The text-only model is capable of the caption task and not this
    // one, so it must not appear here.
    expect(options).toEqual(['gemini/gemini-3.6-flash'])
  })

  it('shows what each task needs, in words rather than jargon', async () => {
    render(<TaskRouting />)
    expect(await screen.findByText('Reads images')).toBeInTheDocument()
    expect(screen.getAllByText('Returns structured JSON').length).toBeGreaterThan(0)
  })

  it('names the fallback so a rate limit is not a surprise', async () => {
    render(<TaskRouting />)
    expect(await screen.findByText(/falls back to Gemini 3.6 Flash/)).toBeInTheDocument()
  })

  it('says what ships when no model can serve a task', async () => {
    vi.spyOn(client, 'apiGet').mockResolvedValue({
      ...base,
      tasks: [{ ...base.tasks[1], assigned: '', eligible: [], resolved: [], served: false }],
    })
    render(<TaskRouting />)
    expect(
      await screen.findByText(/The upload is refused, naming the missing capability/),
    ).toBeInTheDocument()
    expect(screen.getByText(/no capable model configured/)).toBeInTheDocument()
  })

  it('surfaces a refused assignment with the reason', async () => {
    vi.spyOn(client, 'apiPost').mockRejectedValue(
      new Error('Llama 3.1 8B Instant cannot be used for Tables from photos: it cannot do what this task needs: reads images'),
    )
    render(<TaskRouting />)
    const select = await screen.findByLabelText(/model for chart captions/i)
    await userEvent.selectOptions(select, 'gemini/gemini-3.6-flash')
    expect(await screen.findByText(/cannot do what this task needs/)).toBeInTheDocument()
  })

  it('saves an assignment and re-reads the server’s answer', async () => {
    const post = vi.spyOn(client, 'apiPost').mockResolvedValue({
      ...base,
      tasks: [{ ...base.tasks[0], assigned: 'gemini/gemini-3.6-flash', source: 'runtime' }],
    })
    render(<TaskRouting />)
    const select = await screen.findByLabelText(/model for chart captions/i)
    await userEvent.selectOptions(select, 'gemini/gemini-3.6-flash')
    await waitFor(() =>
      expect(post).toHaveBeenCalledWith('/api/admin/routing', {
        task: 'chart_caption',
        model_id: 'gemini/gemini-3.6-flash',
      }),
    )
  })

  it('reports a misconfiguration rather than applying it quietly', async () => {
    vi.spyOn(client, 'apiGet').mockResolvedValue({
      ...base,
      problems: [
        {
          task: 'table_extraction',
          model_id: 'groq/llama-3.1-8b-instant',
          kind: 'incapable',
          detail:
            'Llama 3.1 8B Instant is assigned to Tables from photos, but it cannot do what this task needs: reads images.',
          source: 'env',
        },
      ],
    })
    render(<TaskRouting />)
    expect(
      await screen.findByText(/it cannot do what this task needs: reads images/),
    ).toBeInTheDocument()
  })

  it('says a stale environment name still works', async () => {
    vi.spyOn(client, 'apiGet').mockResolvedValue({
      ...base,
      deprecated: [{ from: 'chart_analysis', to: 'chart_caption' }],
    })
    render(<TaskRouting />)
    expect(await screen.findByText(/still names/)).toBeInTheDocument()
  })

  it('survives a server that answers with the wrong shape', async () => {
    vi.spyOn(client, 'apiGet').mockResolvedValue({ unexpected: true })
    render(<TaskRouting />)
    expect(
      await screen.findByText(/did not report any routable tasks/),
    ).toBeInTheDocument()
  })
})
