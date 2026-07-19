import { X, FilterX } from 'lucide-react'
import { useApp } from '../store/app'

function fmtVal(v: unknown): string {
  if (Array.isArray(v)) return v.join(' – ')
  return String(v)
}

const OP_LABEL: Record<string, string> = {
  eq: '=',
  ne: '≠',
  gt: '>',
  lt: '<',
  gte: '≥',
  lte: '≤',
  in: 'in',
  contains: '⊃',
  between: 'between',
}

export default function FilterBar() {
  const { filters, removeFilter, clearFilters } = useApp()
  if (!filters.length) return null
  return (
    <div className="flex flex-wrap items-center gap-2">
      {filters.map((f, i) => (
        <span
          key={i}
          className="flex items-center gap-1.5 rounded-full border border-accent/40 bg-accent/10 px-3 py-1 text-xs text-accent"
        >
          <b>{f.column}</b> {OP_LABEL[f.op] ?? f.op} {fmtVal(f.value)}
          <button
            onClick={() => removeFilter(i)}
            className="hover:text-rose"
            title="Remove filter"
          >
            <X className="h-3 w-3" />
          </button>
        </span>
      ))}
      <button
        onClick={clearFilters}
        className="flex items-center gap-1 rounded-full border border-edge px-3 py-1 text-xs text-mute hover:text-rose"
      >
        <FilterX className="h-3 w-3" /> Clear all
      </button>
    </div>
  )
}
