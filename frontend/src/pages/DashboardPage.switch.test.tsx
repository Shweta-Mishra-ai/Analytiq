/**
 * Switching datasets.
 *
 * The tiles a dashboard is showing belong to one dataset. When the user
 * picked a different file, `ds` changed immediately but `tiles` still
 * held the previous file's specs, and the effect that reloads tiles ran
 * once in between — so a finance dataset was asked for a histogram of
 * MonthlyIncome. Plotly raised inside make_figure and the client got a
 * 500 with nothing usable in it; one tile stayed broken until reload.
 *
 * A ref compared against `ds` does not fix this, because by then both
 * sides are already the new id. The guard has to name the dataset the
 * tiles in state were actually built for.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { act, waitFor } from '@testing-library/react'

// plotly.js does not load under jsdom, and nothing here inspects a
// rendered figure — only which requests the page makes.
vi.mock('../components/PlotlyChart', () => ({ default: () => null }))
import { render } from '../test/render'
import DashboardPage from './DashboardPage'
import { useApp } from '../store/app'
import * as client from '../api/client'
import type { DatasetMeta } from '../api/client'

const meta = (id: string, filename: string): DatasetMeta => ({
  dataset_id: id, filename, size_mb: 1, uploaded_at: 0,
  rows: 2000, cols: 6, warnings: [],
})

const TILES: Record<string, { id: string; title: string; type: string; x?: string; y?: string; agg?: string }[]> = {
  hr: [{ id: 't1', title: 'Monthly Income by Department', type: 'bar', x: 'Department', y: 'MonthlyIncome', agg: 'auto' }],
  fin: [{ id: 't1', title: 'Revenue by Account', type: 'bar', x: 'account', y: 'revenue', agg: 'auto' }],
}

const COLUMNS: Record<string, string[]> = {
  hr: ['Department', 'MonthlyIncome'],
  fin: ['account', 'revenue'],
}

function stub() {
  const builds: { ds: string; x?: string; y?: string }[] = []
  vi.spyOn(client, 'apiGet').mockImplementation(async (path: string) => {
    const ds = path.split('/')[3]
    return { fields: COLUMNS[ds].map((name) => ({ name, kind: 'categorical', unique: 4 })) }
  })
  vi.spyOn(client, 'apiPost').mockImplementation(async (path: string, body: unknown) => {
    const ds = path.split('/')[3]
    if (path.endsWith('/recommend-tiles')) return { tiles: TILES[ds] }
    if (path.endsWith('/kpis')) return { kpis: [], data_quality: [] }
    if (path.endsWith('/build')) {
      const b = body as { x?: string; y?: string }
      builds.push({ ds, x: b.x, y: b.y })
      // The server rejects a column the dataset does not have.
      for (const col of [b.x, b.y].filter(Boolean) as string[])
        if (!COLUMNS[ds].includes(col))
          throw new Error(`Column '${col}' is not in this dataset.`)
      return { figure: { data: [], layout: {} } }
    }
    return {}
  })
  return builds
}

// jsdom has no ResizeObserver, and the grid measures its container.
class NoopResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

beforeEach(() => {
  vi.restoreAllMocks()
  vi.stubGlobal('ResizeObserver', NoopResizeObserver)
  useApp.setState({ dataset: meta('hr', 'hr.csv'), filters: [] })
})

describe('switching datasets', () => {
  it('never asks a dataset for a column it does not have', async () => {
    const builds = stub()
    render(<DashboardPage />)
    await waitFor(() => expect(builds.length).toBeGreaterThan(0))

    await act(async () => {
      useApp.setState({ dataset: meta('fin', 'finance.csv'), filters: [] })
    })
    await waitFor(() =>
      expect(builds.some((b) => b.ds === 'fin')).toBe(true),
    )

    const wrong = builds.filter(
      (b) => ![b.x, b.y].filter(Boolean).every((c) => COLUMNS[b.ds].includes(c as string)),
    )
    expect(wrong).toEqual([])
  })

  it('loads the new dataset’s own tiles', async () => {
    const builds = stub()
    render(<DashboardPage />)
    await waitFor(() => expect(builds.length).toBeGreaterThan(0))

    await act(async () => {
      useApp.setState({ dataset: meta('fin', 'finance.csv'), filters: [] })
    })
    await waitFor(() =>
      expect(builds.filter((b) => b.ds === 'fin').map((b) => b.y)).toContain(
        'revenue',
      ),
    )
  })
})
