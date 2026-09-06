/**
 * Segment health is a verdict — "this segment is a 43, that one a 69" —
 * and a verdict needs to know which way better runs. It did not: health
 * was scored as bigger-numbers-win, so the strongest region in a test
 * dataset (highest revenue, half the churn, 40% lower support cost)
 * scored lowest of four and was advised to raise its churn rate to the
 * dataset average. These tests pin what the page shows once the engine
 * knows the difference, including the case where it cannot know.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { screen } from '@testing-library/react'
import { render } from '../test/render'
import BiPage from './BiPage'
import { useApp } from '../store/app'
import * as client from '../api/client'
import type { DatasetMeta } from '../api/client'

const meta: DatasetMeta = {
  dataset_id: 'ds1',
  filename: 'sales.csv',
  size_mb: 0.4,
  uploaded_at: 0,
  rows: 600,
  cols: 4,
  warnings: [],
}

const segment = {
  segment_name: 'A',
  segment_col: 'region',
  n: 141,
  health_score: 69.1,
  strengths: ['revenue', 'churn_rate', 'support_cost'],
  weaknesses: [],
  opportunity: "Already leading in 'churn_rate' — leverage this advantage.",
  scored: true,
}

const report = {
  root_causes: [],
  cohorts: [],
  pareto: [],
  segments: [segment],
  key_insights: [],
  executive_brief: 'Region A leads on every measure.',
}

function stub(over: Record<string, unknown> = {}) {
  vi.spyOn(client, 'apiGet').mockResolvedValue({ ...report, ...over })
}

beforeEach(() => {
  useApp.setState({ dataset: meta, filters: [] })
})

describe('segment health', () => {
  it('shows the score and the metrics that earned it', async () => {
    stub()
    render(<BiPage />)
    expect(await screen.findByText('69')).toBeInTheDocument()
    expect(
      screen.getByText(/revenue · churn_rate · support_cost/),
    ).toBeInTheDocument()
  })

  it('shows a dash, not a middling 50, when nothing could be judged', async () => {
    stub({
      segments: [
        {
          ...segment,
          health_score: 50,
          strengths: [],
          opportunity:
            'No metric here has a direction its name makes clear, so this segment cannot be called healthier or weaker than another.',
          scored: false,
        },
      ],
    })
    render(<BiPage />)
    expect(await screen.findByText('—')).toBeInTheDocument()
    expect(screen.queryByText('50')).toBeNull()
    expect(
      screen.getByText(/cannot be called healthier or weaker/),
    ).toBeInTheDocument()
  })
})

describe('without a dataset', () => {
  it('asks for data rather than running an analysis of nothing', () => {
    useApp.setState({ dataset: null })
    render(<BiPage />)
    expect(screen.queryByText('Segment health')).toBeNull()
  })
})
