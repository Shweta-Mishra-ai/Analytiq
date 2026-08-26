/**
 * The reports page is the product's delivery surface: whatever it sends
 * is what reaches the client. These tests pin the request it builds and
 * the file it names, because a wrong flag here silently ships a report
 * missing a section the user asked for.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import { render } from '../test/render'
import userEvent from '@testing-library/user-event'
import ReportsPage from './ReportsPage'
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

const health = {
  domain: 'hr',
  health: {
    score: 82,
    grade: 'B',
    label: 'Good',
    color: '#2dd4bf',
    missing_pct: 1.2,
    dup_pct: 0,
    outlier_pct: 8,
    rows: 1470,
    cols: 35,
  },
  insights: [
    { tag: 'RISK', title: 'Overtime drives attrition', body: '', action: '', severity: 'critical' },
  ],
  executive_summary: 'Attrition is concentrated in two departments.',
  key_findings: ['a', 'b', 'c'],
  risks: ['r'],
  opportunities: [],
  actions: ['x', 'y'],
}

let blobSpy: ReturnType<typeof vi.spyOn>

beforeEach(() => {
  useApp.setState({ dataset: meta, filters: [] })
  vi.spyOn(client, 'apiGet').mockResolvedValue(health)
  blobSpy = vi.spyOn(client, 'apiBlob').mockResolvedValue(new Blob(['%PDF']))
  vi.spyOn(client, 'downloadBlob').mockImplementation(() => {})
})

describe('without a dataset', () => {
  it('asks for data instead of offering a report of nothing', () => {
    useApp.setState({ dataset: null })
    render(<ReportsPage />)
    expect(screen.queryByRole('button', { name: /Generate report/ })).toBeNull()
  })
})

describe('the PDF request', () => {
  it('carries the title, client and section choices the user set', async () => {
    const user = userEvent.setup()
    render(<ReportsPage />)

    const title = screen.getByDisplayValue('Data Analysis Report')
    await user.clear(title)
    await user.type(title, 'Attrition Review')
    await user.type(
      screen.getByPlaceholderText(/S. Mishra/),
      'S. Mishra, Data Analytics',
    )
    // ML is off by default; a user who trained a model wants it in.
    await user.click(screen.getByLabelText('ML results (if trained)'))
    await user.click(screen.getByRole('button', { name: /Generate report/ }))

    await waitFor(() => expect(blobSpy).toHaveBeenCalled())
    const [path, method, body] = blobSpy.mock.calls[0]
    expect(path).toBe('/api/reports/ds1/pdf')
    expect(method).toBe('POST')
    expect(body).toMatchObject({
      title: 'Attrition Review',
      prepared_by: 'S. Mishra, Data Analytics',
      include_stats: true,
      include_bi: true,
      include_ml: true,
      format: 'pdf',
    })
  })

  it('asks for a deck on the deck button and names the file .pptx', async () => {
    const user = userEvent.setup()
    render(<ReportsPage />)
    await user.click(screen.getByRole('button', { name: /Generate deck/ }))

    await waitFor(() => expect(blobSpy).toHaveBeenCalled())
    expect(blobSpy.mock.calls[0][2]).toMatchObject({ format: 'pptx' })
    expect(client.downloadBlob).toHaveBeenCalledWith(
      expect.any(Blob),
      'analytiq_report.pptx',
    )
  })

  it('leaves attribution empty rather than inventing a name', async () => {
    const user = userEvent.setup()
    render(<ReportsPage />)
    await user.click(screen.getByRole('button', { name: /Generate report/ }))
    await waitFor(() => expect(blobSpy).toHaveBeenCalled())
    expect(blobSpy.mock.calls[0][2]).toMatchObject({ prepared_by: '' })
  })

  it('shows the failure instead of a silent no-op', async () => {
    blobSpy.mockRejectedValue(new Error('PDF build failed: no numeric columns'))
    const user = userEvent.setup()
    render(<ReportsPage />)
    await user.click(screen.getByRole('button', { name: /Generate report/ }))
    expect(
      await screen.findByText(/PDF build failed: no numeric columns/),
    ).toBeInTheDocument()
  })
})

describe('the health panel', () => {
  it('shows the grade and what the report will contain', async () => {
    render(<ReportsPage />)
    expect(await screen.findByText(/Grade B — Good/)).toBeInTheDocument()
    expect(
      screen.getByText('Attrition is concentrated in two departments.'),
    ).toBeInTheDocument()
    expect(screen.getByText('3 findings')).toBeInTheDocument()
    expect(screen.getByText('1 risk')).toBeInTheDocument()
    expect(screen.getByText('1 insight')).toBeInTheDocument()
  })

  it('omits a count of zero rather than showing "0 opportunities"', async () => {
    render(<ReportsPage />)
    await screen.findByText(/Grade B/)
    expect(screen.queryByText(/0 opportunities/)).toBeNull()
  })

  it('stays usable when the health call fails', async () => {
    vi.spyOn(client, 'apiGet').mockRejectedValue(new Error('down'))
    render(<ReportsPage />)
    // The report buttons must not depend on the scorecard loading.
    expect(
      screen.getByRole('button', { name: /Generate report/ }),
    ).toBeEnabled()
    await waitFor(() =>
      expect(screen.getByText('Computing data health…')).toBeInTheDocument(),
    )
  })
})

describe('data exports', () => {
  it('names the cleaned CSV and the workbook distinctly', async () => {
    const user = userEvent.setup()
    render(<ReportsPage />)
    await user.click(screen.getByText('Cleaned CSV'))
    await waitFor(() =>
      expect(client.downloadBlob).toHaveBeenCalledWith(
        expect.any(Blob),
        'analytiq_cleaned_data.csv',
      ),
    )
    await user.click(screen.getByText('Excel workbook'))
    await waitFor(() =>
      expect(client.downloadBlob).toHaveBeenCalledWith(
        expect.any(Blob),
        'analytiq_export.xlsx',
      ),
    )
  })
})
