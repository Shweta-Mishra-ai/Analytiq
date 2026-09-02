import { useEffect, useRef, useState } from 'react'
import { Check, ChevronDown, Database, Plus } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { apiGet, type DatasetMeta } from '../api/client'
import { useApp } from '../store/app'
import * as fmt from '../lib/format'

/**
 * Which dataset every page on screen is describing, and how to change it.
 *
 * This used to be four words of grey text at the bottom of the sidebar,
 * below the fold on a short window, with no way to switch without
 * navigating back to Upload. In a tool where every number on every page
 * depends on which file is loaded, that is the one piece of state that
 * has to be visible at all times.
 */
export default function DatasetSwitcher() {
  const dataset = useApp((s) => s.dataset)
  const setDataset = useApp((s) => s.setDataset)
  const [open, setOpen] = useState(false)
  const [items, setItems] = useState<DatasetMeta[]>([])
  const nav = useNavigate()
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    apiGet<{ datasets: DatasetMeta[] }>('/api/datasets')
      .then((r) => setItems(r.datasets))
      .catch(() => setItems([]))
  }, [open])

  useEffect(() => {
    if (!open) return
    const onAway = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    const onEsc = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false)
    document.addEventListener('mousedown', onAway)
    document.addEventListener('keydown', onEsc)
    return () => {
      document.removeEventListener('mousedown', onAway)
      document.removeEventListener('keydown', onEsc)
    }
  }, [open])

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="flex max-w-80 items-center gap-2.5 rounded-lg border border-edge bg-panel2 px-3 py-1.5 text-left transition hover:border-edge2"
      >
        <Database className="h-3.5 w-3.5 shrink-0 text-accent" />
        <span className="min-w-0">
          <span className="block truncate text-[13px] font-medium text-ink">
            {dataset ? dataset.filename : 'No dataset selected'}
          </span>
          {dataset && (
            <span className="block font-data text-[11px] text-faint">
              {fmt.count(dataset.rows)} rows · {dataset.cols} columns
            </span>
          )}
        </span>
        <ChevronDown className="h-3.5 w-3.5 shrink-0 text-mute" />
      </button>

      {open && (
        <div
          role="listbox"
          className="absolute top-full left-0 z-40 mt-1.5 w-80 overflow-hidden rounded-xl border border-edge bg-panel shadow-2xl shadow-black/40"
        >
          <div className="max-h-72 overflow-y-auto py-1">
            {items.length === 0 && (
              <p className="px-3 py-4 text-center text-xs text-mute">
                Nothing uploaded yet.
              </p>
            )}
            {items.map((d) => {
              const active = d.dataset_id === dataset?.dataset_id
              return (
                <button
                  key={d.dataset_id}
                  role="option"
                  aria-selected={active}
                  onClick={() => {
                    setDataset(d)
                    setOpen(false)
                  }}
                  className={`flex w-full items-center gap-2.5 px-3 py-2 text-left transition ${
                    active ? 'bg-accent/10' : 'hover:bg-panel2'
                  }`}
                >
                  <Check
                    className={`h-3.5 w-3.5 shrink-0 ${
                      active ? 'text-accent' : 'opacity-0'
                    }`}
                  />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[13px] text-ink">
                      {d.filename}
                    </span>
                    <span className="block font-data text-[11px] text-faint">
                      {fmt.count(d.rows)} × {d.cols} · {d.size_mb} MB
                    </span>
                  </span>
                </button>
              )
            })}
          </div>
          <button
            onClick={() => {
              setOpen(false)
              nav('/')
            }}
            className="flex w-full items-center gap-2 border-t border-edge px-3 py-2.5 text-[13px] text-mute transition hover:bg-panel2 hover:text-ink"
          >
            <Plus className="h-3.5 w-3.5" /> Upload another file
          </button>
        </div>
      )}
    </div>
  )
}
