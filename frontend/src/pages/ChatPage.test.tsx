/**
 * An assistant that has read your file should ask about your file.
 *
 * The starter questions were four fixed strings — "Show top 10 rows by
 * the main numeric column", "Which category has the highest average
 * value?" Someone who does not know their schema cannot tell what
 * either would do, and someone who does is being asked to translate.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { screen } from '@testing-library/react'
import { render } from '../test/render'
import ChatPage from './ChatPage'
import { useApp } from '../store/app'
import * as client from '../api/client'
import type { DatasetMeta, Field } from '../api/client'

const meta: DatasetMeta = {
  dataset_id: 'ds1',
  filename: 'sales.csv',
  size_mb: 3.4,
  uploaded_at: 0,
  rows: 40000,
  cols: 15,
  warnings: [],
}

const field = (
  name: string,
  kind: Field['kind'],
  unique = 10,
  missing_pct = 0,
): Field => ({ name, kind, unique, missing_pct })

function stubFields(fields: Field[]) {
  vi.spyOn(client, 'apiGet').mockResolvedValue({ fields } as never)
}

beforeEach(() => {
  vi.restoreAllMocks()
  useApp.setState({ dataset: meta, filters: [] })
})

describe('starter questions', () => {
  it('name the columns actually in the file', async () => {
    stubFields([
      field('units', 'numeric', 39),
      field('revenue', 'numeric', 33623),
      field('region', 'categorical', 5),
      field('order_date', 'datetime', 40000),
    ])
    render(<ChatPage />)
    expect(
      await screen.findByRole('button', {
        name: /which region has the highest average revenue/i,
      }),
    ).toBeInTheDocument()
  })

  it('pick the headline measure, not the leftmost column', async () => {
    /* `units` comes first in the file. The question a director wants
       asked is about revenue. Column order is not importance. */
    stubFields([
      field('units', 'numeric', 39),
      field('revenue', 'numeric', 33623),
      field('region', 'categorical', 5),
    ])
    render(<ChatPage />)
    const asked = await screen.findByRole('button', {
      name: /highest average/i,
    })
    expect(asked.textContent).toMatch(/Revenue/)
    expect(asked.textContent).not.toMatch(/Units/)
  })

  it('never offer an identifier as a grouping', async () => {
    /* 40,000 order IDs are not a segmentation. */
    stubFields([
      field('order_id', 'categorical', 40000),
      field('region', 'categorical', 5),
      field('revenue', 'numeric', 33623),
    ])
    render(<ChatPage />)
    await screen.findByRole('button', { name: /highest average/i })
    expect(screen.queryByText(/order id/i)).not.toBeInTheDocument()
  })

  it('fall back to generic questions when the file is too thin', async () => {
    stubFields([field('note', 'categorical', 40000)])
    render(<ChatPage />)
    expect(
      await screen.findByRole('button', { name: /how many rows and columns/i }),
    ).toBeInTheDocument()
  })

  it('leave the page usable when the lookup fails', async () => {
    vi.spyOn(client, 'apiGet').mockRejectedValue(new Error('nope'))
    render(<ChatPage />)
    expect(
      await screen.findByPlaceholderText(/average salary by department/i),
    ).toBeInTheDocument()
  })
})
