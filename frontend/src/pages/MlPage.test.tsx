/**
 * The ML page is where a model either earns trust or should lose it.
 * The verdict and the leakage notes exist so a model that beat nothing
 * cannot be read as a result — these tests hold that in place.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { render } from '../test/render'
import MlPage from './MlPage'
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

const targets = {
  targets: [
    { column: 'attrition', task: 'classification', reason: 'two balanced-enough classes' },
    { column: 'monthly_income', task: 'regression', reason: 'continuous' },
  ],
}

const goodReport = {
  task: 'classification',
  target_col: 'attrition',
  n_rows_used: 1470,
  n_features: 28,
  models: [
    { name: 'Logistic Regression', task: 'classification', cv_score: 0.741, cv_std: 0.021, test_score: 0.752, overfit_label: 'None', metric_name: 'ROC AUC', roc_auc: 0.752, is_best: false },
    { name: 'Gradient Boosting', task: 'classification', cv_score: 0.784, cv_std: 0.018, test_score: 0.781, overfit_label: 'Mild', metric_name: 'ROC AUC', roc_auc: 0.781, is_best: true },
  ],
  best_model: { name: 'Gradient Boosting', task: 'classification', cv_score: 0.784, cv_std: 0.018, test_score: 0.781, overfit_label: 'Mild', metric_name: 'ROC AUC', roc_auc: 0.781, is_best: true },
  feature_importance: [
    { feature: 'overtime', importance: 0.21, rank: 1, direction: 'up', explanation: 'raises the modelled risk' },
    { feature: 'tenure_years', importance: 0.14, rank: 2, direction: 'down', explanation: 'lowers the modelled risk' },
  ],
  warnings: [],
  insights: ['Overtime is the strongest single driver.'],
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
  leakage: [],
}

beforeEach(() => {
  useApp.setState({ dataset: meta, filters: [] })
})

function stubApi(report: unknown | null) {
  vi.spyOn(client, 'apiGet').mockImplementation(async (path: string) => {
    if (path.endsWith('/targets')) return targets
    if (path.endsWith('/report')) {
      if (report === null) throw new Error('No trained model yet')
      return report
    }
    throw new Error(`unexpected ${path}`)
  })
}

/** The model name and its ★ are separate nodes, so the row is found by
 *  its whole text rather than by a single matching element. */
async function modelRow(name: string) {
  return (await screen.findByRole('row', {
    name: (_content, el) => (el.textContent ?? '').includes(name),
  })) as HTMLElement
}

describe('target selection', () => {
  it('offers each suggested target with its reason', async () => {
    stubApi(null)
    render(<MlPage />)
    await waitFor(() =>
      expect(screen.getByRole('combobox')).toHaveValue('attrition'),
    )
    expect(
      screen.getByText('two balanced-enough classes'),
    ).toBeInTheDocument()
    expect(
      screen.getByRole('option', { name: /monthly_income — regression/ }),
    ).toBeInTheDocument()
  })

  it('trains the target the user picked, not the default', async () => {
    stubApi(null)
    const post = vi.spyOn(client, 'apiPost').mockResolvedValue(goodReport)
    const user = userEvent.setup()
    render(<MlPage />)
    await waitFor(() => expect(screen.getByRole('combobox')).toBeEnabled())

    await user.selectOptions(screen.getByRole('combobox'), 'monthly_income')
    await user.click(screen.getByRole('button', { name: /Train models/ }))

    await waitFor(() =>
      expect(post).toHaveBeenCalledWith('/api/ml/ds1/train', {
        target: 'monthly_income',
      }),
    )
  })
})

describe('a model that worked', () => {
  it('shows the leaderboard with the winner marked', async () => {
    stubApi(goodReport)
    render(<MlPage />)
    const row = await modelRow('Gradient Boosting')
    expect(within(row).getByText('★')).toBeInTheDocument()
    expect(within(row).getByText('78.1%')).toBeInTheDocument()
    expect(within(row).getByText('Mild')).toBeInTheDocument()

    // The model it beat is listed too, and is not marked as selected.
    const runnerUp = await modelRow('Logistic Regression')
    expect(within(runnerUp).queryByText('★')).toBeNull()
  })

  it('explains each driver rather than listing bare importances', async () => {
    stubApi(goodReport)
    render(<MlPage />)
    expect(await screen.findByText('overtime')).toBeInTheDocument()
    expect(
      screen.getByText('raises the modelled risk'),
    ).toBeInTheDocument()
  })

  it('does not show a no-signal warning on a usable model', async () => {
    stubApi(goodReport)
    render(<MlPage />)
    await modelRow('Gradient Boosting')
    expect(screen.queryByText(/No predictive signal found/)).toBeNull()
  })
})

describe('a model that did not work', () => {
  const noSignal = {
    ...goodReport,
    verdict: {
      ...goodReport.verdict,
      usable: false,
      auc: 0.51,
      minority_recall: 0.02,
      verdict:
        'No combination of the recorded fields separates the outcomes; the importances below describe noise.',
    },
  }

  it('states the verdict where the importances are, not in a log', async () => {
    stubApi(noSignal)
    render(<MlPage />)
    expect(
      await screen.findByText(/No predictive signal found/),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/the importances below describe noise/),
    ).toBeInTheDocument()
  })
})

describe('leakage', () => {
  it('names fields that were excluded and why', async () => {
    stubApi({
      ...goodReport,
      leakage: [
        {
          column: 'exit_date',
          separation: 0.99,
          reason: 'is only recorded after the outcome it predicts',
        },
      ],
    })
    render(<MlPage />)
    expect(await screen.findByText('exit_date')).toBeInTheDocument()
    expect(
      screen.getByText(/only recorded after the outcome it predicts/),
    ).toBeInTheDocument()
  })
})

describe('failures', () => {
  it('shows a training error instead of leaving the page blank', async () => {
    stubApi(null)
    vi.spyOn(client, 'apiPost').mockRejectedValue(
      new Error('Training failed: only one class present'),
    )
    const user = userEvent.setup()
    render(<MlPage />)
    await waitFor(() => expect(screen.getByRole('combobox')).toBeEnabled())
    await user.click(screen.getByRole('button', { name: /Train models/ }))
    expect(
      await screen.findByText(/only one class present/),
    ).toBeInTheDocument()
  })

  it('shows no stale report when there is no trained model yet', async () => {
    stubApi(null)
    render(<MlPage />)
    await waitFor(() => expect(screen.getByRole('combobox')).toBeEnabled())
    expect(screen.queryByText('Model leaderboard')).toBeNull()
  })
})
