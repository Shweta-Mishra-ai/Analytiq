/**
 * The dashboard.
 *
 * What was here worked — draggable tiles, cross-filtering, a tile
 * builder — but read as a chart grid rather than a report:
 *
 *  - **It chose its own tiles in the browser**, taking `nums[0]` and
 *    `cats[0]` from the field list. On a sales export that is `order_id`
 *    and revenue never appeared. The starting layout now comes from
 *    `/layout`, which uses the same measure ranking as the PDF export and
 *    the recommended charts, so the three cannot disagree about what the
 *    business measure is.
 *  - **Every tile was the same 6×5 rectangle**, so nothing said what to
 *    read first. Tile sizes now come from the server with the lead
 *    measure on a full-width tile.
 *  - **Filters could only be applied by clicking a bar.** A reader who
 *    wanted EMEA had to find a chart that happened to show EMEA. There is
 *    now a slicer rail.
 *  - **Tiles were titled with their axes.** Each carries its finding, and
 *    the axis names sit underneath it.
 */
import { useCallback, useEffect, useMemo, useState } from 'react'
import GridLayout, { type LayoutItem } from 'react-grid-layout'
import { GripVertical, Plus, RefreshCw, Trash2 } from 'lucide-react'
import { apiGet, apiPost, type Field, type Kpi } from '../api/client'
import type { Figure } from '../types'
import { useApp } from '../store/app'
import FilterBar from '../components/FilterBar'
import KpiCard from '../components/KpiCard'
import PlotlyChart from '../components/PlotlyChart'
import SlicerRail from '../components/SlicerRail'
import { Btn, ErrorBox, NeedData, PageHeader, Spinner } from '../components/Ui'

interface TileSpec {
  id: string
  title: string
  type: string
  x?: string
  y?: string
  agg?: string
}

interface TileState {
  figure?: Figure
  error?: string
  loading: boolean
}

/** What `/layout` returns: a tile plus where it sits on the grid. */
interface ServerTile extends TileSpec {
  w: number
  h: number
  gx: number
  gy: number
}

export default function DashboardPage() {
  const { dataset, filters, addFilter } = useApp()
  const [fields, setFields] = useState<Field[]>([])
  const [kpis, setKpis] = useState<Kpi[]>([])
  const [tiles, setTiles] = useState<TileSpec[]>([])
  const [tileState, setTileState] = useState<Record<string, TileState>>({})
  const [layout, setLayout] = useState<LayoutItem[]>([])
  const [showBuilder, setShowBuilder] = useState(false)
  const [error, setError] = useState('')
  const [width, setWidth] = useState(1200)

  const ds = dataset?.dataset_id

  useEffect(() => {
    const el = document.getElementById('dash-grid-wrap')
    if (!el) return
    const obs = new ResizeObserver((e) => setWidth(e[0].contentRect.width))
    obs.observe(el)
    return () => obs.disconnect()
  }, [ds, fields.length])

  // ── fields for the slicer rail and the tile builder ────
  useEffect(() => {
    if (!ds) return
    apiGet<{ fields: Field[] }>(`/api/charts/${ds}/fields`)
      .then((r) => setFields(r.fields))
      .catch((e) => setError(e.message))
  }, [ds])

  // ── the starting layout, decided server-side ───────────
  useEffect(() => {
    if (!ds) return
    setError('')
    apiGet<{ tiles: ServerTile[] }>(`/api/charts/${ds}/layout`)
      .then((r) => {
        const t = r.tiles ?? []
        setTiles(
          t.map((s) => ({
            id: s.id,
            title: s.title,
            type: s.type,
            x: s.x,
            y: s.y,
            agg: s.agg,
          })),
        )
        setLayout(
          t.map((s) => ({
            i: s.id,
            x: s.gx ?? 0,
            y: s.gy ?? 0,
            w: s.w,
            h: s.h,
            minW: 3,
            minH: 3,
          })),
        )
      })
      .catch((e) => setError(e.message))
  }, [ds])

  const loadKpis = useCallback(() => {
    if (!ds) return
    apiPost<{ kpis: Kpi[] }>(`/api/charts/${ds}/kpis`, { filters })
      .then((r) => setKpis(r.kpis))
      .catch(() => {})
  }, [ds, filters])

  const loadTile = useCallback(
    (t: TileSpec) => {
      if (!ds) return
      setTileState((s) => ({ ...s, [t.id]: { ...s[t.id], loading: true } }))
      apiPost<{ figure: Figure }>(`/api/charts/${ds}/build`, {
        type: t.type,
        x: t.x,
        y: t.y,
        agg: t.agg ?? 'sum',
        title: '',
        filters,
      })
        .then((r) =>
          setTileState((s) => ({
            ...s,
            [t.id]: { figure: r.figure, loading: false },
          })),
        )
        .catch((e) =>
          setTileState((s) => ({
            ...s,
            [t.id]: { error: e.message, loading: false },
          })),
        )
    },
    [ds, filters],
  )

  useEffect(() => {
    loadKpis()
    tiles.forEach(loadTile)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters, tiles.length, ds])

  const catFields = useMemo(
    () => new Set(fields.filter((f) => f.kind === 'categorical').map((f) => f.name)),
    [fields],
  )

  const handlePointClick = (t: TileSpec) => (pt: Record<string, unknown>) => {
    const col = t.x
    if (!col || !catFields.has(col)) return
    const value = (pt.label ?? pt.x) as string | undefined
    if (value !== undefined)
      addFilter({ column: col, op: 'eq', value: String(value) })
  }

  if (!dataset) return <NeedData />

  const filtered = filters.length > 0

  return (
    <div className="p-6">
      <PageHeader
        title="Dashboard"
        subtitle={`${dataset.filename} — pick a value on the left, or click any bar or slice, to filter every tile`}
        right={
          <div className="flex gap-2">
            <Btn variant="ghost" onClick={() => tiles.forEach(loadTile)}>
              <RefreshCw className="h-4 w-4" />
            </Btn>
            <Btn onClick={() => setShowBuilder(true)}>
              <span className="flex items-center gap-1.5">
                <Plus className="h-4 w-4" /> Add tile
              </span>
            </Btn>
          </div>
        }
      />
      {error && (
        <div className="mb-4">
          <ErrorBox message={error} />
        </div>
      )}

      <div className="flex gap-5">
        <SlicerRail fields={fields} />

        <div className="min-w-0 flex-1">
          {filtered && (
            <div className="mb-4">
              <FilterBar />
            </div>
          )}

          <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-8">
            {kpis.map((k) => (
              <KpiCard key={k.label} kpi={k} />
            ))}
          </div>

          <div id="dash-grid-wrap">
            <GridLayout
              className="layout"
              layout={layout}
              gridConfig={{ cols: 12, rowHeight: 64 }}
              width={width}
              onLayoutChange={(l) => setLayout([...l])}
              dragConfig={{ handle: '.tile-drag' }}
            >
              {tiles.map((t) => {
                const st = tileState[t.id]
                return (
                  <div
                    key={t.id}
                    className="group flex flex-col overflow-hidden rounded-xl border border-edge bg-panel"
                  >
                    <div className="flex shrink-0 items-center justify-between gap-2 px-3 pt-2.5">
                      <div className="tile-drag flex min-w-0 cursor-move items-center gap-1.5">
                        <GripVertical className="h-3.5 w-3.5 shrink-0 text-faint opacity-0 transition-opacity group-hover:opacity-100" />
                        <span className="truncate text-[11px] tracking-wide text-mute uppercase">
                          {t.title}
                        </span>
                      </div>
                      <button
                        onClick={() => {
                          setTiles((ts) => ts.filter((x) => x.id !== t.id))
                          setLayout((l) => l.filter((x) => x.i !== t.id))
                        }}
                        className="shrink-0 text-faint opacity-0 transition-opacity group-hover:opacity-100 hover:text-rose"
                        title="Remove tile"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                    <div className="min-h-0 flex-1 px-1 pb-1">
                      {st?.loading && !st.figure && <Spinner label="" />}
                      {st?.error && (
                        <div className="p-3 text-xs text-rose">{st.error}</div>
                      )}
                      {st?.figure && (
                        <PlotlyChart
                          figure={st.figure}
                          onClickPoint={handlePointClick(t)}
                        />
                      )}
                    </div>
                  </div>
                )
              })}
            </GridLayout>
          </div>

          {!tiles.length && !error && (
            <p className="rounded-xl border border-dashed border-edge px-4 py-8 text-center text-sm text-mute">
              Nothing in this file charts as a business measure — every
              numeric column reads as an identifier. Add a tile to plot one
              anyway.
            </p>
          )}
        </div>
      </div>

      {showBuilder && (
        <TileBuilder
          fields={fields}
          onClose={() => setShowBuilder(false)}
          onAdd={(t) => {
            setTiles((ts) => [...ts, t])
            setLayout((l) => [
              ...l,
              { i: t.id, x: 0, y: Infinity, w: 6, h: 5, minW: 3, minH: 3 },
            ])
            setShowBuilder(false)
          }}
        />
      )}
    </div>
  )
}

// ── tile builder modal ───────────────────────────────────
function TileBuilder({
  fields,
  onAdd,
  onClose,
}: {
  fields: Field[]
  onAdd: (t: TileSpec) => void
  onClose: () => void
}) {
  const [type, setType] = useState('bar')
  const [x, setX] = useState('')
  const [y, setY] = useState('')
  const [agg, setAgg] = useState('sum')

  const nums = fields.filter((f) => f.kind === 'numeric')
  const xOptions =
    type === 'histogram' || type === 'scatter'
      ? nums
      : fields.filter((f) => f.kind !== 'numeric')

  const needsY = !['histogram', 'heatmap'].includes(type)
  const valid = type === 'heatmap' || (x && (!needsY || y))

  const sel =
    'w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink focus:border-accent focus:outline-none'

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      onClick={onClose}
    >
      <div
        className="w-96 rounded-2xl border border-edge bg-panel p-5"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 className="mb-4 font-semibold text-ink">Add tile</h3>
        <div className="space-y-3">
          <div>
            <label className="mb-1 block text-xs text-mute">Chart type</label>
            <select
              value={type}
              onChange={(e) => setType(e.target.value)}
              className={sel}
            >
              {['bar', 'line', 'area', 'pie', 'scatter', 'histogram', 'heatmap'].map(
                (t) => (
                  <option key={t}>{t}</option>
                ),
              )}
            </select>
          </div>
          {type !== 'heatmap' && (
            <div>
              <label className="mb-1 block text-xs text-mute">
                {type === 'histogram' ? 'Column' : 'X axis / category'}
              </label>
              <select value={x} onChange={(e) => setX(e.target.value)} className={sel}>
                <option value="">— select —</option>
                {xOptions.map((f) => (
                  <option key={f.name}>{f.name}</option>
                ))}
              </select>
            </div>
          )}
          {needsY && (
            <div>
              <label className="mb-1 block text-xs text-mute">Value (numeric)</label>
              <select value={y} onChange={(e) => setY(e.target.value)} className={sel}>
                <option value="">— select —</option>
                {nums.map((f) => (
                  <option key={f.name}>{f.name}</option>
                ))}
              </select>
            </div>
          )}
          {needsY && type !== 'scatter' && (
            <div>
              <label className="mb-1 block text-xs text-mute">Aggregation</label>
              <select
                value={agg}
                onChange={(e) => setAgg(e.target.value)}
                className={sel}
              >
                {['sum', 'mean', 'count', 'median', 'min', 'max'].map((a) => (
                  <option key={a}>{a}</option>
                ))}
              </select>
            </div>
          )}
          <div className="flex gap-2 pt-1">
            <Btn variant="ghost" className="flex-1" onClick={onClose}>
              Cancel
            </Btn>
            <Btn
              disabled={!valid}
              className="flex-1"
              onClick={() =>
                onAdd({
                  id: `t${Date.now()}`,
                  title:
                    type === 'heatmap'
                      ? 'Correlation matrix'
                      : type === 'histogram'
                        ? `Distribution of ${x}`
                        : `${agg} of ${y} by ${x}`,
                  type,
                  x: x || undefined,
                  y: y || undefined,
                  agg,
                })
              }
            >
              Add
            </Btn>
          </div>
        </div>
      </div>
    </div>
  )
}
