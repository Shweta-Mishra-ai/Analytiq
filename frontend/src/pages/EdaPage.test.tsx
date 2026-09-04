/**
 * Deep EDA serves two readers at once.
 *
 * The page stays technical — every statistic it computed is still on it.
 * What it gained is a second wording for the same results, so a director
 * can read the findings without knowing what a variance inflation factor
 * is, and the analyst beside them can still check the working.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { render } from '../test/render'
import EdaPage from './EdaPage'
import { useApp } from '../store/app'
import * as client from '../api/client'
import type { DatasetMeta } from '../api/client'

const meta: DatasetMeta = {
  dataset_id: 'ds1',
  filename: 'sales.csv',
  size_mb: 3.4,
  uploaded_at: 0,
  rows: 40000,
  cols: 15,
  warnings: [],
}

const report = {
  n_rows: 40000,
  n_cols: 15,
  numeric_cols: ['revenue', 'profit'],
  categorical_cols: ['region'],
  datetime_cols: [],
  univariate: {
    revenue: {
      column: 'revenue',
      mean: 523,
      median: 404,
      std: 410,
      skewness: 2.2,
      skew_label: 'heavily right-skewed',
      normality_verdict: 'Not normal',
      outliers_iqr: 1700,
      outlier_pct: 4.3,
      best_fit_dist: 'lognorm',
      interpretation: 'Heavily right-skewed. Mean 523 vs median 404.',
      plain: 'Revenue is pulled by a long tail of unusually high values.',
    },
  },
  correlations: [
    {
      col_a: 'revenue',
      col_b: 'profit',
      test_name: 'Spearman',
      statistic: 0.96,
      p_value: 0.0,
      is_significant: true,
      effect_size: 0.96,
      effect_label: 'large',
      interpretation: 'Strong positive correlation (r=0.96, p<0.001).',
      plain: 'Revenue and Profit move up and down together, almost perfectly.',
    },
  ],
  group_comparisons: [],
  multicollinearity: [
    {
      feature: 'revenue',
      vif: 14.0,
      verdict: 'High',
      interpretation: 'High multicollinearity — consider removing or combining.',
      plain: 'Revenue repeats information already carried by other columns.',
    },
  ],
  identifier_cols: ['order_id'],
  time_series: [],
  key_findings: ['1 feature(s) have high VIF (multicollinearity): revenue.'],
  plain_findings: ['Revenue repeats information already carried by other columns.'],
  warnings: [],
}

beforeEach(() => {
  vi.restoreAllMocks()
  useApp.setState({ dataset: meta, filters: [] })
  vi.spyOn(client, 'apiGet').mockResolvedValue(report as never)
})

describe('Deep EDA wording', () => {
  it('leads with the plain reading', async () => {
    /* The same sentence legitimately appears twice — once as a headline
       finding and once against the VIF row it came from. */
    render(<EdaPage />)
    const shown = await screen.findAllByText(/repeats information already carried/i)
    expect(shown.length).toBeGreaterThan(0)
  })

  it('shows the statistical wording when asked for it', async () => {
    render(<EdaPage />)
    await screen.findAllByText(/repeats information already carried/i)

    await userEvent.click(screen.getByRole('button', { name: /statistical/i }))

    expect(
      await screen.findByText(/high VIF \(multicollinearity\)/i),
    ).toBeInTheDocument()
  })

  it('goes back to plain wording', async () => {
    render(<EdaPage />)
    await screen.findAllByText(/repeats information already carried/i)

    await userEvent.click(screen.getByRole('button', { name: /statistical/i }))
    await screen.findByText(/high VIF \(multicollinearity\)/i)
    await userEvent.click(screen.getByRole('button', { name: /in plain terms/i }))

    const back = await screen.findAllByText(/repeats information already carried/i)
    expect(back.length).toBeGreaterThan(0)
  })

  it('keeps every statistic on the page in either wording', async () => {
    /* The toggle changes the prose, not the numbers. A reader who
       switches to plain wording must not lose the VIF value itself. */
    render(<EdaPage />)
    await screen.findAllByText(/repeats information already carried/i)
    expect(await screen.findByText(/VIF 14\.0/)).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: /statistical/i }))
    expect(await screen.findByText(/VIF 14\.0/)).toBeInTheDocument()
  })

  it('explains what a VIF row means, in either wording', async () => {
    /* The panel used to read "VIF 14.0 — High" and nothing else: the
       most jargon-dense element on the page carried no explanation. */
    render(<EdaPage />)
    expect(
      (await screen.findAllByText(/repeats information already carried/i)).length,
    ).toBeGreaterThan(0)

    await userEvent.click(screen.getByRole('button', { name: /statistical/i }))
    expect(
      await screen.findByText(/consider removing or combining/i),
    ).toBeInTheDocument()
  })
})
