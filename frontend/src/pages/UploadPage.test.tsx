/**
 * Upload is the first thing every user does and the only place the
 * backend's warnings about their data reach them. A warning that gets
 * dropped here — "this table is cut off, so this is a portion of it" —
 * is a client analysing a fragment without knowing it.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { render } from '../test/render'
import UploadPage from './UploadPage'
import { useApp } from '../store/app'
import * as client from '../api/client'
import type { DatasetMeta, TableData } from '../api/client'

const meta = (over: Partial<DatasetMeta> = {}): DatasetMeta => ({
  dataset_id: 'ds1',
  filename: 'hr.csv',
  size_mb: 1.2,
  uploaded_at: 0,
  rows: 1470,
  cols: 35,
  warnings: [],
  ...over,
})

const preview: TableData = {
  columns: ['department', 'salary'],
  dtypes: { department: 'str', salary: 'int64' },
  records: [{ department: 'Sales', salary: 61000 }],
  total_rows: 1470,
  truncated: true,
}

/** The page makes two different GETs — the recent list and the preview
 *  for whichever dataset is selected — so the stub answers by path. */
function stubGet(datasets: DatasetMeta[] = []) {
  return vi.spyOn(client, 'apiGet').mockImplementation(async (path: string) => {
    if (path.includes('/preview')) return preview
    return { datasets }
  })
}

beforeEach(() => {
  useApp.setState({ dataset: null, filters: [] })
  stubGet()
})

const csv = () =>
  new File(['department,salary\nSales,61000'], 'hr.csv', { type: 'text/csv' })

/** The file inputs are visually hidden, so they are addressed by the
 *  accept list rather than by a label. */
function inputFor(accept: string) {
  return document.querySelector<HTMLInputElement>(`input[accept*="${accept}"]`)!
}

describe('uploading a file', () => {
  it('selects the new dataset and shows its shape', async () => {
    vi.spyOn(client, 'apiUpload').mockResolvedValue({ meta: meta(), preview })
    render(<UploadPage />)
    await userEvent.upload(inputFor('.csv'), csv())

    expect(await screen.findByText('hr.csv')).toBeInTheDocument()
    expect(screen.getByText(/1,470 rows × 35 columns/)).toBeInTheDocument()
    expect(useApp.getState().dataset?.dataset_id).toBe('ds1')
  })

  it('posts to the tabular endpoint, not an extraction one', async () => {
    const upload = vi
      .spyOn(client, 'apiUpload')
      .mockResolvedValue({ meta: meta(), preview })
    render(<UploadPage />)
    await userEvent.upload(inputFor('.csv'), csv())
    await waitFor(() =>
      expect(upload).toHaveBeenCalledWith(
        '/api/datasets/upload',
        expect.any(File),
      ),
    )
  })

  it('routes an image to the extraction endpoint instead', async () => {
    const upload = vi
      .spyOn(client, 'apiUpload')
      .mockResolvedValue({ meta: meta({ filename: 'table.png' }), preview })
    render(<UploadPage />)
    await userEvent.upload(
      inputFor('.png'),
      new File(['x'], 'table.png', { type: 'image/png' }),
    )
    await waitFor(() =>
      expect(upload).toHaveBeenCalledWith(
        '/api/datasets/extract-from-image',
        expect.any(File),
      ),
    )
  })

  it('shows the reason an upload was rejected', async () => {
    vi.spyOn(client, 'apiUpload').mockRejectedValue(
      new Error('File too large — the limit is 200 MB'),
    )
    render(<UploadPage />)
    await userEvent.upload(inputFor('.csv'), csv())
    expect(
      await screen.findByText(/the limit is 200 MB/),
    ).toBeInTheDocument()
    expect(useApp.getState().dataset).toBeNull()
  })
})

describe('warnings about the data', () => {
  it('shows every warning on its own line', async () => {
    vi.spyOn(client, 'apiUpload').mockResolvedValue({
      meta: meta({
        warnings: [
          'The table is cut off, so this is a portion of it.',
          '2 column(s) mix numbers and text and are stored as text: amount, code',
        ],
      }),
      preview,
    })
    render(<UploadPage />)
    await userEvent.upload(inputFor('.csv'), csv())

    const first = await screen.findByText(/The table is cut off/)
    const second = screen.getByText(/mix numbers and text/)
    // Run together on one line they read as noise and get skipped.
    expect(first.closest('li')).not.toBe(second.closest('li'))
  })

  it('shows no warning strip when the upload was clean', async () => {
    vi.spyOn(client, 'apiUpload').mockResolvedValue({ meta: meta(), preview })
    render(<UploadPage />)
    await userEvent.upload(inputFor('.csv'), csv())
    await screen.findByText('hr.csv')
    expect(screen.queryByText('⚠')).toBeNull()
  })
})

describe('recent datasets', () => {
  it('lists earlier uploads so work can be resumed', async () => {
    stubGet([meta({ dataset_id: 'ds9', filename: 'sales_2024.csv' })])
    render(<UploadPage />)
    expect(await screen.findByText('sales_2024.csv')).toBeInTheDocument()
  })

  it('does not show the panel when there is nothing to resume', async () => {
    render(<UploadPage />)
    await waitFor(() =>
      expect(screen.queryByText('Recent datasets')).toBeNull(),
    )
  })
})
