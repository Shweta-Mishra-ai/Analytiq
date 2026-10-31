/**
 * Power BI-style dashboard:
 *  - opening tiles chosen by the server with the same rules as the report's
 *    charts (an outcome as a rate, totals only for measures that add up)
 *  - draggable / resizable tiles (react-grid-layout); stacked on narrow screens
 *  - every tile re-queries the server with the shared slicer state
 *  - clicking a bar / pie slice cross-filters every other tile
 *  - add-tile builder (chart type, x, y, aggregation)
 */
import { useCallback, useEffect, useMemo, useState } from 'react'
import GridLayout, { type LayoutItem } from 'react-grid-layout'
import { GripVertical, Plus, RefreshCw, X } from 'lucide-react'
import { apiGet, apiPost, type Field, type Kpi } from '../api/client'
import type { Figure } from '../types'
import { useApp } from '../store/app'
import FilterBar from '../components/FilterBar'
import KpiCard from '../components/KpiCard'
import PlotlyChart from '../components/PlotlyChart'
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

const DEFAULT_LAYOUT_H = 5

const TYPE_LABEL: Record<string, string> = {
  bar: 'Bar',
  rate: 'Outcome rate by group',
  line: 'Line (over time)',
  area: 'Area (over time)',
  pie: 'Pie (shares of a total)',
  scatter: 'Scatter',
  histogram: 'Distribution',
  heatmap: 'Correlation grid',
}

const AGG_LABEL: Record<string, string> = {
  mean: 'Average',
  sum: 'Total',
  median: 'Median',
  count: 'Count of',
  min: 'Minimum',
  max: 'Maximum',
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
  }, [ds])

  // ── initial: fields + the server's opening tiles ───────
  // The tiles were chosen here from the first numeric and first text
  // column, which summed employee_id by department and drew a pie of it.
  useEffect(() => {
    if (!ds) return
    setError('')
    Promise.all([
      apiGet<{ fields: Field[] }>(`/api/charts/${ds}/fields`),
      apiGet<{ tiles: TileSpec[] }>(`/api/charts/${ds}/auto-tiles`),
    ])
      .then(([f, t]) => {
        setFields(f.fields)
        setTiles(t.tiles)
        setLayout(
          t.tiles.map((tile, i) => ({
            i: tile.id,
            x: (i % 2) * 6,
            y: Math.floor(i / 2) * DEFAULT_LAYOUT_H,
            w: 6,
            h: DEFAULT_LAYOUT_H,
            minW: 3,
            minH: 3,
          })),
        )
      })
      .catch((e) => setError(e.message))
  }, [ds])

  // ── refetch KPIs + all tiles when filters change ──────
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

  // clicking a categorical point cross-filters all tiles
  const handlePointClick = (t: TileSpec) => (pt: Record<string, unknown>) => {
    const col = t.x
    if (!col || !catFields.has(col)) return
    const value = (pt.label ?? pt.x) as string | undefined
    if (value !== undefined)
      addFilter({ column: col, op: 'eq', value: String(value) })
  }

  const removeTile = (id: string) => {
    setTiles((ts) => ts.filter((x) => x.id !== id))
    setLayout((l) => l.filter((x) => x.i !== id))
  }

  if (!dataset) return <NeedData />

  return (
    <div className="page">
      <PageHeader
        title="Dashboard"
        subtitle={`${dataset.filename} — click any bar or slice to cross-filter`}
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
      <div className="mb-4">
        <FilterBar />
      </div>

      <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
        {kpis.map((k) => (
          <KpiCard key={k.label} kpi={k} />
        ))}
      </div>

      {tiles.length === 0 && !error && (
        <div className="surface p-8 text-center text-sm text-mute">
          No tiles yet — use <span className="font-semibold text-ink">Add tile</span> to build one.
        </div>
      )}

      <div id="dash-grid-wrap">
        {width < 768 ? (
          // A 12-column grid on a phone made every tile 150px wide.
          <div className="space-y-3">
            {tiles.map((t) => (
              <div key={t.id} className="h-80">
                <Tile t={t} st={tileState[t.id]} onRemove={removeTile}
                  onClickPoint={handlePointClick(t)} />
              </div>
            ))}
          </div>
        ) : (
          <GridLayout
            className="layout"
            layout={layout}
            gridConfig={{ cols: 12, rowHeight: 64 }}
            width={width}
            onLayoutChange={(l) => setLayout([...l])}
            dragConfig={{ handle: '.tile-drag' }}
          >
            {tiles.map((t) => (
              <div key={t.id}>
                <Tile t={t} st={tileState[t.id]} onRemove={removeTile}
                  onClickPoint={handlePointClick(t)} />
              </div>
            ))}
          </GridLayout>
        )}
      </div>

      {showBuilder && (
        <TileBuilder
          fields={fields}
          onClose={() => setShowBuilder(false)}
          onAdd={(t) => {
            setTiles((ts) => [...ts, t])
            setLayout((l) => [
              ...l,
              {
                i: t.id,
                x: 0,
                y: Infinity,
                w: 6,
                h: DEFAULT_LAYOUT_H,
                minW: 3,
                minH: 3,
              },
            ])
            setShowBuilder(false)
          }}
        />
      )}
    </div>
  )
}

// ── one tile ─────────────────────────────────────────────
function Tile({
  t,
  st,
  onRemove,
  onClickPoint,
}: {
  t: TileSpec
  st?: TileState
  onRemove: (id: string) => void
  onClickPoint: (pt: Record<string, unknown>) => void
}) {
  return (
    <div className="surface flex h-full flex-col overflow-hidden">
      <div className="flex items-center justify-between border-b border-edge px-3 py-2">
        <div className="tile-drag flex min-w-0 cursor-grab items-center gap-1.5">
          <GripVertical className="h-3.5 w-3.5 shrink-0 text-faint" />
          <span className="truncate text-[13px] font-semibold text-ink">{t.title}</span>
        </div>
        <button
          onClick={() => onRemove(t.id)}
          className="rounded p-0.5 text-faint hover:bg-panel2 hover:text-rose"
          title="Remove tile"
          aria-label={`Remove ${t.title}`}
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="min-h-0 flex-1 p-1.5">
        {st?.loading && !st.figure && <Spinner label="" />}
        {st?.error && <div className="p-3 text-xs text-rose">{st.error}</div>}
        {st?.figure && <PlotlyChart figure={st.figure} onClickPoint={onClickPoint} />}
      </div>
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
  // Average by default: a total is right only for measures that add up
  // (revenue, units), and summing salary or age gives a meaningless number.
  const [agg, setAgg] = useState('mean')

  const nums = fields.filter((f) => f.kind === 'numeric')
  const twoValue = fields.filter((f) => f.unique === 2)
  const xOptions =
    type === 'histogram'
      ? nums
      : type === 'scatter'
        ? nums
        : fields.filter((f) => f.kind !== 'numeric' || type === 'scatter')
  const yOptions = type === 'rate' ? twoValue : nums

  const needsY = !['histogram', 'heatmap'].includes(type)
  const valid =
    type === 'heatmap' || (x && (!needsY || y))

  const sel =
    'w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink'

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="w-96 rounded-2xl border border-edge bg-panel p-5">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="font-semibold text-ink">Add tile</h3>
          <button onClick={onClose} className="text-mute hover:text-ink">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="space-y-3">
          <div>
            <label className="mb-1 block text-xs text-mute">Chart type</label>
            <select value={type} onChange={(e) => setType(e.target.value)} className={sel}>
              {Object.entries(TYPE_LABEL).map(([t, label]) => (
                <option key={t} value={t}>
                  {label}
                </option>
              ))}
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
              <label className="mb-1 block text-xs text-mute">
                {type === 'rate' ? 'Outcome (two values, e.g. Yes/No)' : 'Value (numeric)'}
              </label>
              <select value={y} onChange={(e) => setY(e.target.value)} className={sel}>
                <option value="">— select —</option>
                {yOptions.map((f) => (
                  <option key={f.name}>{f.name}</option>
                ))}
              </select>
            </div>
          )}
          {needsY && type !== 'scatter' && type !== 'rate' && (
            <div>
              <label className="mb-1 block text-xs text-mute">Aggregation</label>
              <select value={agg} onChange={(e) => setAgg(e.target.value)} className={sel}>
                {Object.entries(AGG_LABEL).map(([a, label]) => (
                  <option key={a} value={a}>
                    {label}
                  </option>
                ))}
              </select>
            </div>
          )}
          <Btn
            disabled={!valid}
            className="w-full"
            onClick={() =>
              onAdd({
                id: `t${Date.now()}`,
                title:
                  type === 'heatmap'
                    ? 'How the measures relate'
                    : type === 'histogram'
                      ? `Distribution of ${x}`
                      : type === 'rate'
                        ? `${y} rate by ${x}`
                        : type === 'scatter'
                          ? `${y} against ${x}`
                          : `${AGG_LABEL[agg]} ${y} by ${x}`,
                type,
                x: x || undefined,
                y: y || undefined,
                agg,
              })
            }
          >
            Add to dashboard
          </Btn>
        </div>
      </div>
    </div>
  )
}
