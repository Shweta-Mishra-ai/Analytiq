import { beforeEach, describe, expect, it } from 'vitest'
import { useApp } from '../store/app'
import type { DatasetMeta } from '../api/client'

beforeEach(() => useApp.setState({ dataset: null, filters: [] }))

describe('app store', () => {
  it('replaces a filter on the same column and operator instead of stacking it', () => {
    const { addFilter } = useApp.getState()
    addFilter({ column: 'Country', op: 'eq', value: 'France' })
    addFilter({ column: 'Country', op: 'eq', value: 'EIRE' })
    expect(useApp.getState().filters).toEqual([{ column: 'Country', op: 'eq', value: 'EIRE' }])
  })

  it('clears filters when a different dataset is chosen', () => {
    useApp.getState().addFilter({ column: 'Country', op: 'eq', value: 'France' })
    useApp.getState().setDataset({ dataset_id: 'x' } as DatasetMeta)
    expect(useApp.getState().filters).toEqual([])
  })

  it('removes one filter by position', () => {
    const s = useApp.getState()
    s.addFilter({ column: 'a', op: 'eq', value: 1 })
    s.addFilter({ column: 'b', op: 'eq', value: 2 })
    useApp.getState().removeFilter(0)
    expect(useApp.getState().filters.map((f) => f.column)).toEqual(['b'])
  })
})
