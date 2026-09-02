import { useEffect, useMemo, useState } from 'react'
import { CornerDownLeft, Search } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { apiGet, type DatasetMeta } from '../api/client'
import { useApp } from '../store/app'
import * as fmt from '../lib/format'

export interface Command {
  id: string
  label: string
  group: string
  run: () => void
  hint?: string
}

/**
 * Every destination and every dataset, one keystroke away.
 *
 * With fifteen analysis pages in four sidebar groups, finding the right
 * one is a scan. A palette is what turns that into typing three letters,
 * and it is the difference between a tool people learn and one they
 * navigate.
 */
export default function CommandPalette({
  pages,
}: {
  pages: { to: string; label: string; group: string }[]
}) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [cursor, setCursor] = useState(0)
  const [datasets, setDatasets] = useState<DatasetMeta[]>([])
  const setDataset = useApp((s) => s.setDataset)
  const nav = useNavigate()

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setOpen((v) => !v)
      }
      if (e.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => {
    if (!open) {
      setQuery('')
      setCursor(0)
      return
    }
    apiGet<{ datasets: DatasetMeta[] }>('/api/datasets')
      .then((r) => setDatasets(r.datasets))
      .catch(() => setDatasets([]))
  }, [open])

  const commands: Command[] = useMemo(() => {
    const go: Command[] = pages.map((p) => ({
      id: 'go:' + p.to,
      label: p.label,
      group: p.group,
      run: () => nav(p.to),
    }))
    const pick: Command[] = datasets.map((d) => ({
      id: 'ds:' + d.dataset_id,
      label: d.filename,
      group: 'Switch dataset',
      hint: `${fmt.count(d.rows)} × ${d.cols}`,
      run: () => setDataset(d),
    }))
    return [...go, ...pick]
  }, [pages, datasets, nav, setDataset])

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return commands.slice(0, 12)
    // Subsequence matching, so "dq" finds "Data Quality" — the way
    // people actually type into a palette.
    const score = (text: string) => {
      const t = text.toLowerCase()
      if (t.startsWith(q)) return 0
      if (t.includes(q)) return 1
      let i = 0
      for (const ch of t) if (ch === q[i]) i++
      return i === q.length ? 2 : 99
    }
    return commands
      .map((c) => ({ c, s: Math.min(score(c.label), score(c.group)) }))
      .filter((x) => x.s < 99)
      .sort((a, b) => a.s - b.s)
      .slice(0, 12)
      .map((x) => x.c)
  }, [query, commands])

  if (!open) return null

  const runAt = (i: number) => {
    const cmd = matches[i]
    if (!cmd) return
    cmd.run()
    setOpen(false)
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 pt-[14vh] backdrop-blur-sm"
      onClick={() => setOpen(false)}
    >
      <div
        role="dialog"
        aria-label="Command palette"
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-lg overflow-hidden rounded-2xl border border-edge2 bg-panel shadow-2xl shadow-black/60"
      >
        <div className="flex items-center gap-2.5 border-b border-edge px-4 py-3">
          <Search className="h-4 w-4 shrink-0 text-mute" />
          <input
            autoFocus
            value={query}
            onChange={(e) => {
              setQuery(e.target.value)
              setCursor(0)
            }}
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') {
                e.preventDefault()
                setCursor((c) => Math.min(c + 1, matches.length - 1))
              } else if (e.key === 'ArrowUp') {
                e.preventDefault()
                setCursor((c) => Math.max(c - 1, 0))
              } else if (e.key === 'Enter') {
                e.preventDefault()
                runAt(cursor)
              }
            }}
            placeholder="Go to a page, or switch dataset…"
            className="w-full bg-transparent text-sm text-ink placeholder:text-faint focus:outline-none"
          />
          <kbd className="rounded border border-edge px-1.5 py-0.5 font-data text-[10px] text-faint">
            esc
          </kbd>
        </div>
        <div className="max-h-80 overflow-y-auto py-1.5">
          {matches.length === 0 && (
            <p className="px-4 py-6 text-center text-xs text-mute">
              Nothing matches “{query}”.
            </p>
          )}
          {matches.map((cmd, i) => (
            <button
              key={cmd.id}
              onMouseEnter={() => setCursor(i)}
              onClick={() => runAt(i)}
              className={`flex w-full items-center gap-3 px-4 py-2 text-left transition ${
                i === cursor ? 'bg-accent/12' : ''
              }`}
            >
              <span
                className={`flex-1 truncate text-[13px] ${
                  i === cursor ? 'text-ink' : 'text-ink2'
                }`}
              >
                {cmd.label}
              </span>
              {cmd.hint && (
                <span className="font-data text-[11px] text-faint">
                  {cmd.hint}
                </span>
              )}
              <span className="text-[11px] text-faint">{cmd.group}</span>
              {i === cursor && (
                <CornerDownLeft className="h-3 w-3 shrink-0 text-accent" />
              )}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
