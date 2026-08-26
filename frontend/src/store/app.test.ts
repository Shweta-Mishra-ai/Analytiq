/**
 * The filter store decides what every chart and table on the page is
 * looking at, so its rules are worth pinning down.
 */
import { beforeEach, describe, expect, it } from 'vitest'
import { useApp } from './app'
import type { DatasetMeta } from '../api/client'

const meta = (id: string): DatasetMeta => ({
  dataset_id: id,
  filename: `${id}.csv`,
  size_mb: 1,
  uploaded_at: 0,
  rows: 10,
  cols: 3,
  warnings: [],
})

beforeEach(() => {
  useApp.setState({ dataset: null, filters: [] })
})

describe('dataset selection', () => {
  it('clears filters when the dataset changes', () => {
    // A filter on `department` means nothing against a different upload,
    // and silently carrying it over would show an empty page.
    useApp.getState().setDataset(meta('a'))
    useApp.getState().addFilter({ column: 'department', op: '==', value: 'Sales' })
    useApp.getState().setDataset(meta('b'))
    expect(useApp.getState().filters).toEqual([])
  })
})

describe('filters', () => {
  it('replaces rather than stacks a repeat click on the same column', () => {
    const { addFilter } = useApp.getState()
    addFilter({ column: 'department', op: '==', value: 'Sales' })
    addFilter({ column: 'department', op: '==', value: 'Engineering' })
    expect(useApp.getState().filters).toEqual([
      { column: 'department', op: '==', value: 'Engineering' },
    ])
  })

  it('keeps filters on different columns side by side', () => {
    const { addFilter } = useApp.getState()
    addFilter({ column: 'department', op: '==', value: 'Sales' })
    addFilter({ column: 'salary', op: '>', value: 50000 })
    expect(useApp.getState().filters).toHaveLength(2)
  })

  it('treats a different operator on the same column as a separate filter', () => {
    // `salary > 40000` and `salary < 90000` are a range, not a conflict.
    const { addFilter } = useApp.getState()
    addFilter({ column: 'salary', op: '>', value: 40000 })
    addFilter({ column: 'salary', op: '<', value: 90000 })
    expect(useApp.getState().filters).toHaveLength(2)
  })

  it('removes by index and clears all', () => {
    const { addFilter } = useApp.getState()
    addFilter({ column: 'a', op: '==', value: 1 })
    addFilter({ column: 'b', op: '==', value: 2 })
    useApp.getState().removeFilter(0)
    expect(useApp.getState().filters.map((f) => f.column)).toEqual(['b'])
    useApp.getState().clearFilters()
    expect(useApp.getState().filters).toEqual([])
  })
})
