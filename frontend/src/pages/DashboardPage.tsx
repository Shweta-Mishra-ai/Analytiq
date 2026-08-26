/**
 * Power BI-style dashboard:
 *  - draggable / resizable tiles (react-grid-layout)
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

export default function DashboardPage() {
  const { dataset, filters, addFilter } = useApp()
  const [fields, setFields] = useState<Field[]>([])
  const [kpis, setKpis] = useState<Kpi[]>([])
  // Record counts and completeness describe the file, not the
  // business. Shown, but not mixed in with the KPIs.
  const [quality, setQuality] = useState<Kpi[]>([])
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

  // ── initial: fields + auto-recommended tiles ──────────
  useEffect(() => {
    if (!ds) return
    setError('')
    apiGet<{ fields: Field[] }>(`/api/charts/${ds}/fields`)
      .then((r) => {
        setFields(r.fields)
        const nums = r.fields.filter((f) => f.kind === 'numeric')
        const cats = r.fields.filter(
          (f) => f.kind === 'categorical' && f.unique > 1 && f.unique <= 30,
        )
        const dates = r.fields.filter((f) => f.kind === 'datetime')
        const auto: TileSpec[] = []
        if (cats[0] && nums[0])
          auto.push({
            id: 't1',
            title: `${nums[0].name} by ${cats[0].name}`,
            type: 'bar',
            x: cats[0].name,
            y: nums[0].name,
            agg: 'sum',
          })
        if (dates[0] && nums[0])
          auto.push({
            id: 't2',
            title: `${nums[0].name} over time`,
            type: 'line',
            x: dates[0].name,
            y: nums[0].name,
            agg: 'sum',
          })
        if (cats[1] && nums[0])
          auto.push({
            id: 't3',
            title: `${nums[0].name} share by ${cats[1].name}`,
            type: 'pie',
            x: cats[1].name,
            y: nums[0].name,
            agg: 'sum',
          })
        if (nums[1])
          auto.push({
            id: 't4',
            title: `Distribution of ${nums[1].name}`,
            type: 'histogram',
            x: nums[1].name,
          })
        if (nums.length >= 2)
          auto.push({ id: 't5', title: 'Correlation', type: 'heatmap' })
        setTiles(auto)
        setLayout(
          auto.map((t, i) => ({
            i: t.id,
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
    apiPost<{ kpis: Kpi[]; data_quality?: Kpi[] }>(
      `/api/charts/${ds}/kpis`, { filters })
      .then((r) => {
        setKpis(r.kpis)
        setQuality(r.data_quality ?? [])
      })
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

  if (!dataset) return <NeedData />

  return (
    <div className="p-6">
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

      <div className="mb-3 grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        {kpis.map((k) => (
          <KpiCard key={k.label} kpi={k} />
        ))}
      </div>

      {quality.length > 0 && (
        <div className="mb-4 flex flex-wrap items-center gap-x-5 gap-y-1
                        text-[11px] text-mute">
          {quality.map((q) => (
            <span key={q.label} className="font-data">
              {q.label}:{' '}
              <span className="text-ink">
                {q.format === 'pct'
                  ? `${q.value}%`
                  : q.value.toLocaleString()}
              </span>
            </span>
          ))}
        </div>
      )}

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
                className="overflow-hidden rounded-xl border border-edge bg-panel"
              >
                <div className="flex items-center justify-between border-b border-edge px-3 py-1.5">
                  <div className="tile-drag flex min-w-0 items-center gap-1.5">
                    <GripVertical className="h-3.5 w-3.5 shrink-0 text-mute" />
                    <span className="truncate text-xs font-semibold text-ink">
                      {t.title}
                    </span>
                  </div>
                  <button
                    onClick={() => {
                      setTiles((ts) => ts.filter((x) => x.id !== t.id))
                      setLayout((l) => l.filter((x) => x.i !== t.id))
                    }}
                    className="text-mute hover:text-rose"
                    title="Remove tile"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
                <div className="h-[calc(100%-2rem)] p-1">
                  {st?.loading && <Spinner label="" />}
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
    type === 'histogram'
      ? nums
      : type === 'scatter'
        ? nums
        : fields.filter((f) => f.kind !== 'numeric' || type === 'scatter')

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
              <select value={agg} onChange={(e) => setAgg(e.target.value)} className={sel}>
                {['sum', 'mean', 'count', 'median', 'min', 'max'].map((a) => (
                  <option key={a}>{a}</option>
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
                    ? 'Correlation'
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
            Add to dashboard
          </Btn>
        </div>
      </div>
    </div>
  )
}
