import type { Kpi } from '../api/client'

function fmt(k: Kpi): string {
  if (k.format === 'pct') return `${k.value}%`
  if (k.format === 'int') return k.value.toLocaleString()
  const v = k.value
  if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`
  if (Math.abs(v) >= 10_000) return `${(v / 1_000).toFixed(1)}K`
  return v.toLocaleString(undefined, { maximumFractionDigits: 2 })
}

export default function KpiCard({ kpi }: { kpi: Kpi }) {
  return (
    <div className="surface min-w-0 px-4 py-3">
      <div className="t-label truncate" title={kpi.label}>
        {kpi.label}
      </div>
      <div className="t-metric mt-1.5 text-2xl text-ink">{fmt(kpi)}</div>
      {kpi.sub && (
        <div className="mt-1 truncate text-[11px] text-faint" title={kpi.sub}>
          {kpi.sub}
        </div>
      )}
    </div>
  )
}
