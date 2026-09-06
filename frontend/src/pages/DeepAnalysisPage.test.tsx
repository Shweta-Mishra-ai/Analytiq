/**
 * This page is where the rigour layer became visible. Everything it
 * renders — decision bands, model selection, leakage, intervals,
 * interactions — was already computed by the API and shown nowhere, so
 * these tests are the guard against it going quiet again.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { render } from '../test/render'
import DeepAnalysisPage from './DeepAnalysisPage'
import { useApp } from '../store/app'
import * as client from '../api/client'
import type { DatasetMeta } from '../api/client'

const meta: DatasetMeta = {
  dataset_id: 'ds1',
  filename: 'hr.csv',
  size_mb: 1.2,
  uploaded_at: 0,
  rows: 1470,
  cols: 35,
  warnings: [],
}

const drivers = {
  target: 'attrition',
  auc: 0.781,
  accuracy: 0.748,
  n_rows: 1470,
  n_features: 28,
  top_drivers: [
    ['overtime', 0.21],
    ['tenure_years', 0.14],
  ],
  high_risk_profile: 'Overtime = Yes and tenure under 2 years',
  high_risk_rate: 41.2,
  base_rate: 16.1,
  high_risk_n: 147,
  model_name: 'Gradient Boosting',
  decision_bands: [
    {
      budget_pct: 10,
      n_targeted: 147,
      n_events_caught: 61,
      total_events: 237,
      precision: 41,
      recall: 26,
      lift: 2.4,
    },
  ],
  model_choice: {
    name: 'Gradient Boosting',
    auc: 0.781,
    candidates: [
      ['Logistic Regression', 0.741],
      ['Gradient Boosting', 0.781],
    ],
    threshold: 0.31,
    threshold_basis: 'maximises F1 on held-out data',
    calibrated: true,
    calibration_before: 14,
    calibration_after: 3,
    excluded_high_cardinality: [],
  },
  verdict: {
    usable: true,
    baseline_score: 0.807,
    baseline_strategy: 'always predict No',
    model_score: 0.748,
    metric: 'accuracy',
    lift: -0.059,
    auc: 0.781,
    minority_recall: 0.81,
    reason: 'ranks well',
    verdict: 'The model ranks risk well.',
  },
  leakage: [
    {
      column: 'exit_date',
      separation: 0.99,
      reason: 'is only recorded after the outcome it predicts',
    },
  ],
}

const eda = {
  estimates: [
    {
      column: 'monthly_income',
      statistic: 'mean',
      value: 6502.93,
      ci_low: 6265.11,
      ci_high: 6740.75,
      n: 1470,
    },
  ],
  interactions: [
    {
      metric: 'satisfaction',
      factor: 'overtime',
      moderator: 'department',
      effect_by_level: { Sales: -0.8, Research: 0.6 },
      reverses: true,
      ratio: 2.1,
      description:
        'Overtime lowers satisfaction in Sales but raises it in Research.',
      effect_sd: 0.7,
    },
  ],
  rare_categories: [
    { column: 'department', level: 'Legal', n: 4, share: 0.003 },
  ],
  imbalance_notes: [],
}

function stub(over: Record<string, unknown> = {}) {
  const responses: Record<string, unknown> = {
    benchmarks: { benchmarks: [] },
    'industry-benchmarks': { domain: 'hr', benchmarks: [] },
    eda,
    drivers,
    'scenario/fields': { numeric_columns: ['salary', 'satisfaction'] },
    ...over,
  }
  vi.spyOn(client, 'apiGet').mockImplementation(async (path: string) => {
    for (const [key, value] of Object.entries(responses)) {
      if (path.includes(key)) {
        if (value instanceof Error) throw value
        return value
      }
    }
    throw new Error(`unexpected ${path}`)
  })
}

beforeEach(() => {
  useApp.setState({ dataset: meta, filters: [] })
})

describe('the model that was fitted', () => {
  it('shows where to act, not only how good the model is', async () => {
    stub()
    render(<DeepAnalysisPage />)
    expect(await screen.findByText('Where to act')).toBeInTheDocument()
    expect(screen.getByText('top 10%')).toBeInTheDocument()
  })

  it('shows how the model was chosen and against what', async () => {
    stub()
    render(<DeepAnalysisPage />)
    expect(
      await screen.findByText('How this model was selected'),
    ).toBeInTheDocument()
    expect(screen.getByText('0.741')).toBeInTheDocument()
  })

  it('names the fields it refused to use', async () => {
    stub()
    render(<DeepAnalysisPage />)
    expect(await screen.findByText('exit_date')).toBeInTheDocument()
  })

  it('does not warn about signal when the model is usable', async () => {
    stub()
    render(<DeepAnalysisPage />)
    await screen.findByText('Where to act')
    expect(screen.queryByText(/No predictive signal found/)).toBeNull()
  })
})

describe('a model with nothing to say', () => {
  it('reports it as a finding rather than showing importances alone', async () => {
    stub({
      drivers: {
        ...drivers,
        verdict: {
          ...drivers.verdict,
          usable: false,
          auc: 0.51,
          verdict: 'Nothing recorded here separates the outcomes.',
        },
      },
    })
    render(<DeepAnalysisPage />)
    expect(
      await screen.findByText(/No predictive signal found/),
    ).toBeInTheDocument()
    expect(
      screen.getByText('Nothing recorded here separates the outcomes.'),
    ).toBeInTheDocument()
  })
})

describe('the EDA depth layer', () => {
  it('shows headline measures with their intervals', async () => {
    stub()
    render(<DeepAnalysisPage />)
    expect(
      await screen.findByText('Headline measures, with uncertainty'),
    ).toBeInTheDocument()
    expect(screen.getByText('6,265.11 – 6,740.75')).toBeInTheDocument()
  })

  it('surfaces an effect that reverses between groups', async () => {
    stub()
    render(<DeepAnalysisPage />)
    expect(
      await screen.findByText('Effects that differ by group'),
    ).toBeInTheDocument()
    expect(screen.getByText('direction reverses')).toBeInTheDocument()
  })

  it('names the groups too thin to rank', async () => {
    stub()
    render(<DeepAnalysisPage />)
    expect(await screen.findByText(/category level/)).toBeInTheDocument()
    expect(screen.getByText('Legal')).toBeInTheDocument()
  })
})

describe('when a model could not be fitted', () => {
  it('says why instead of spinning forever', async () => {
    stub({ drivers: new Error('No binary outcome column found') })
    render(<DeepAnalysisPage />)
    expect(
      await screen.findByText('No binary outcome column found'),
    ).toBeInTheDocument()
  })

  it('still renders the rest of the analysis', async () => {
    stub({ drivers: new Error('No binary outcome column found') })
    render(<DeepAnalysisPage />)
    // A missing model must not take the intervals down with it.
    expect(
      await screen.findByText('Headline measures, with uncertainty'),
    ).toBeInTheDocument()
  })
})

/**
 * A projection the data cannot support must say which of the three
 * reasons applies. All three look identical to a reader who sees only
 * "not reliable", and the one that matters most — the driver being the
 * target rewritten — produces the most convincing numbers on the page:
 * a perfect fit, a vanishing p-value, and exactly the change requested.
 */
describe('a projection that cannot be supported', () => {
  const baseScenario = {
    driver_col: 'salary',
    target_col: 'satisfaction',
    change_pct: 10,
    current_driver_mean: 60000,
    current_target_mean: 3.2,
    projected_target_mean: 3.5,
    projected_change_pct: 9.4,
    r_squared: 0.42,
    p_value: 0.0001,
    reliable: true,
    interpretation: 'If salary increases by 10% …',
    caveat: 'Association, not causation.',
    projected_driver_value: 66000,
    driver_observed_min: 30000,
    driver_observed_max: 120000,
    within_observed_range: true,
    driver_restates_target: false,
  }

  function renderScenario(over: Record<string, unknown>) {
    stub()
    vi.spyOn(client, 'apiPost').mockResolvedValue({ ...baseScenario, ...over })
    render(<DeepAnalysisPage />)
  }

  it('names a driver that is the target rewritten', async () => {
    renderScenario({
      driver_col: 'revenue_k',
      target_col: 'revenue',
      reliable: false,
      driver_restates_target: true,
      r_squared: 1,
      p_value: 0,
    })
    const user = userEvent.setup()
    await screen.findByRole('button', { name: /Project/ })
    await user.click(screen.getByRole('button', { name: /Project/ }))

    expect(
      await screen.findByText(/revenue_k is revenue rewritten, not a lever on it/),
    ).toBeInTheDocument()
    expect(screen.queryByText('Too weak to rely on')).toBeNull()
  })

  it('still says "outside what this data covers" when that is the reason', async () => {
    renderScenario({
      reliable: false,
      within_observed_range: false,
      projected_driver_value: 400000,
    })
    const user = userEvent.setup()
    await screen.findByRole('button', { name: /Project/ })
    await user.click(screen.getByRole('button', { name: /Project/ }))

    expect(
      await screen.findByText(/Outside what this data covers/),
    ).toBeInTheDocument()
  })

  it('calls a genuinely weak relationship weak', async () => {
    renderScenario({ reliable: false, r_squared: 0.02, p_value: 0.6 })
    const user = userEvent.setup()
    await screen.findByRole('button', { name: /Project/ })
    await user.click(screen.getByRole('button', { name: /Project/ }))

    expect(await screen.findByText('Too weak to rely on')).toBeInTheDocument()
  })

  it('leaves a sound projection alone', async () => {
    renderScenario({})
    const user = userEvent.setup()
    await screen.findByRole('button', { name: /Project/ })
    await user.click(screen.getByRole('button', { name: /Project/ }))

    expect(
      await screen.findByText('Relationship is strong enough to project'),
    ).toBeInTheDocument()
  })
})
