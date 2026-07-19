import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { DatasetMeta, Filter } from '../api/client'

interface AppState {
  dataset: DatasetMeta | null
  filters: Filter[]
  setDataset: (d: DatasetMeta | null) => void
  setFilters: (f: Filter[]) => void
  addFilter: (f: Filter) => void
  removeFilter: (index: number) => void
  clearFilters: () => void
}

export const useApp = create<AppState>()(
  persist(
    (set) => ({
      dataset: null,
      filters: [],
      setDataset: (dataset) => set({ dataset, filters: [] }),
      setFilters: (filters) => set({ filters }),
      addFilter: (f) =>
        set((s) => {
          // clicking the same category again replaces (not stacks) the filter
          const rest = s.filters.filter(
            (x) => !(x.column === f.column && x.op === f.op),
          )
          return { filters: [...rest, f] }
        }),
      removeFilter: (index) =>
        set((s) => ({ filters: s.filters.filter((_, i) => i !== index) })),
      clearFilters: () => set({ filters: [] }),
    }),
    { name: 'analytiq-app' },
  ),
)
