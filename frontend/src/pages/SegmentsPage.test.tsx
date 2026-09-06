/**
 * When the app cannot guess which column is the customer, it should ask
 * — not stop.
 *
 * Detection reads column names, so a file whose customer column is
 * called `rep` fell through it, and the page said "this dataset doesn't
 * look like transaction data" with nothing to do about it. The endpoint
 * had always accepted explicit columns.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { render } from '../test/render'
import SegmentsPage from './SegmentsPage'
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

const candidates = {
  customer: [
    { column: 'rep', note: '120 distinct, 333.3 rows each', distinct: 120 },
    { column: 'region', note: '5 distinct, 8000.0 rows each', distinct: 5 },
  ],
  date: [{ column: 'order_date', note: 'already a date' }],
  monetary: [
    { column: 'revenue', note: 'totals 20,915,751' },
    { column: 'units', note: 'totals 801,433' },
  ],
}

const report = {
  customer_table: {
    columns: ['customer', 'monetary'],
    dtypes: { customer: 'str', monetary: 'float64' },
    records: [{ customer: 'rep_001', monetary: 421.5 }],
    total_rows: 120,
    truncated: false,
  },
  segment_summary: [],
  columns_used: {
    customer_id: 'rep', date_col: 'order_date',
    monetary_col: 'revenue', quantity_col: null, price_col: null,
  },
  n_customers: 120,
  n_transactions: 40000,
  total_revenue: 20915751,
  top_segment: 'Champions',
  at_risk_pct: 34.2,
  at_risk_revenue: 7201764,
  champions_count: 23,
  warnings: [],
}

beforeEach(() => {
  vi.restoreAllMocks()
  useApp.setState({ dataset: meta, filters: [] })
})

function stub(detected: unknown | null) {
  return vi.spyOn(client, 'apiGet').mockImplementation(async (path: string) => {
    if (path.includes('/rfm/columns')) return { detected, candidates } as never
    if (path.includes('/rfm')) return report as never
    throw new Error(`unexpected ${path}`)
  })
}

describe('when the customer column cannot be guessed', () => {
  it('asks instead of stopping', async () => {
    stub(null)
    render(<SegmentsPage />)
    expect(
      await screen.findByText(/which columns describe your customers/i),
    ).toBeInTheDocument()
  })

  it('fills in its best guess so the choice is a confirmation', async () => {
    stub(null)
    render(<SegmentsPage />)
    const who = await screen.findByLabelText(/who the customer is/i)
    expect((who as HTMLSelectElement).value).toBe('rep')
    expect(
      (await screen.findByLabelText(/what it was worth/i) as HTMLSelectElement)
        .value,
    ).toBe('revenue')
  })

  it('shows what the data says about each option', async () => {
    stub(null)
    render(<SegmentsPage />)
    expect(
      await screen.findByRole('option', { name: /rep · 120 distinct/i }),
    ).toBeInTheDocument()
  })

  it('sends the chosen columns when it runs', async () => {
    const api = stub(null)
    render(<SegmentsPage />)
    await screen.findByLabelText(/who the customer is/i)

    await userEvent.selectOptions(
      screen.getByLabelText(/who the customer is/i), 'region')
    await userEvent.click(
      screen.getByRole('button', { name: /run segmentation/i }))

    await waitFor(() => {
      const call = api.mock.calls
        .map((c) => String(c[0]))
        .find((p) => p.includes('/rfm?'))
      expect(call).toContain('customer_col=region')
      expect(call).toContain('date_col=order_date')
    })
  })

  it('cannot run until a customer and a date are chosen', async () => {
    vi.spyOn(client, 'apiGet').mockImplementation(async (path: string) => {
      if (path.includes('/rfm/columns'))
        return { detected: null,
                 candidates: { customer: [], date: [], monetary: [] } } as never
      throw new Error(`unexpected ${path}`)
    })
    render(<SegmentsPage />)
    await screen.findByText(/nothing to score/i)
    expect(
      screen.getByRole('button', { name: /run segmentation/i }),
    ).toBeDisabled()
  })
})

describe('when detection succeeds', () => {
  it('does not ask', async () => {
    stub({
      customer_id: 'customer_id', date_col: 'order_date',
      monetary_col: 'amount', quantity_col: null, price_col: null,
    })
    render(<SegmentsPage />)
    await screen.findByText(/detected columns/i)
    expect(
      screen.queryByText(/which columns describe your customers/i),
    ).not.toBeInTheDocument()
  })
})
