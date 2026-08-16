import { useMemo, useState } from 'react'
import { ChevronDown, FilterX, Search } from 'lucide-react'
import type { Field } from '../api/client'
import { useApp } from '../store/app'

/**
 * The slicer rail — the control surface a BI report is actually driven
 * from.
 *
 * The dashboard's only filter control was a row of chips showing filters
 * that were already applied. There was no way to apply one except by
 * clicking a bar, so a reader could not ask "show me EMEA" without first
 * finding a chart that happened to have EMEA on it. Every value in every
 * categorical field is now one click away, with the active ones marked,
 * which is what makes a dashboard explorable rather than a picture.
 */

function Slicer({ field }: { field: Field }) {
  const { filters, addFilter, removeFilter } = useApp()
  const [open, setOpen] = useState(true)
  const [query, setQuery] = useState('')

  const activeIndex = filters.findIndex(
    (f) => f.column === field.name && f.op === 'eq',
  )
  const active = activeIndex >= 0 ? String(filters[activeIndex].value) : null

  const values = useMemo(() => {
    const all = field.values ?? []
    if (!query) return all.slice(0, 40)
    const q = query.toLowerCase()
    return all.filter((v) => String(v).toLowerCase().includes(q)).slice(0, 40)
  }, [field.values, query])

  if (!field.values?.length) return null

  return (
    <div className="border-b border-edge/70 last:border-0">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-3 py-2.5 text-left hover:bg-panel2/60"
      >
        <span className="min-w-0">
          <span className="block truncate text-xs font-semibold text-ink2">
            {field.name}
          </span>
          {active && (
            <span className="block truncate font-data text-[11px] text-accent">
              {active}
            </span>
          )}
        </span>
        <ChevronDown
          className={`h-3.5 w-3.5 shrink-0 text-faint transition-transform ${
            open ? '' : '-rotate-90'
          }`}
        />
      </button>

      {open && (
        <div className="px-2 pb-2.5">
          {(field.values?.length ?? 0) > 12 && (
            <div className="relative mb-1.5">
              <Search className="pointer-events-none absolute top-1.5 left-2 h-3 w-3 text-faint" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Find…"
                className="w-full rounded-md border border-edge bg-panel2 py-1 pr-2 pl-6 text-[11px] text-ink placeholder:text-faint focus:border-accent focus:outline-none"
              />
            </div>
          )}
          <div className="max-h-44 space-y-0.5 overflow-y-auto">
            {values.map((v) => {
              const on = active === String(v)
              return (
                <button
                  key={String(v)}
                  onClick={() => {
                    if (on) removeFilter(activeIndex)
                    else
                      addFilter({ column: field.name, op: 'eq', value: String(v) })
                  }}
                  className={`block w-full truncate rounded-md px-2 py-1 text-left text-[11px] transition-colors ${
                    on
                      ? 'bg-accent/15 font-medium text-accent'
                      : 'text-mute hover:bg-panel2 hover:text-ink2'
                  }`}
                  title={String(v)}
                >
                  {String(v)}
                </button>
              )
            })}
            {!values.length && (
              <p className="px-2 py-1 text-[11px] text-faint">No match</p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default function SlicerRail({ fields }: { fields: Field[] }) {
  const { filters, clearFilters } = useApp()
  const slicers = useMemo(
    () =>
      fields.filter(
        (f) => f.kind === 'categorical' && f.unique > 1 && f.unique <= 200,
      ),
    [fields],
  )
  if (!slicers.length) return null

  return (
    <aside className="w-52 shrink-0 self-start overflow-hidden rounded-xl border border-edge bg-panel">
      <div className="flex items-center justify-between border-b border-edge px-3 py-2">
        <span className="text-[11px] font-semibold tracking-wide text-mute uppercase">
          Filters
        </span>
        {filters.length > 0 && (
          <button
            onClick={clearFilters}
            className="flex items-center gap-1 text-[11px] text-faint hover:text-rose"
            title="Clear all filters"
          >
            <FilterX className="h-3 w-3" />
            {filters.length}
          </button>
        )}
      </div>
      <div className="max-h-[calc(100vh-14rem)] overflow-y-auto">
        {slicers.map((f) => (
          <Slicer key={f.name} field={f} />
        ))}
      </div>
    </aside>
  )
}
