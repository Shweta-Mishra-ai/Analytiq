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
      <div className="t-label line-clamp-2" title={kpi.label}>
        {kpi.label}
      </div>
      <div className="mt-1.5 flex items-baseline gap-2">
        <span className="t-metric text-2xl text-ink">{fmt(kpi)}</span>
        {kpi.delta !== undefined && kpi.delta !== null && (
          <span
            className={`text-xs font-semibold ${
              kpi.delta_good === true
                ? 'text-teal'
                : kpi.delta_good === false
                  ? 'text-rose'
                  : 'text-mute'
            }`}
            title="Change on the prior period"
          >
            {kpi.delta > 0 ? '▲' : kpi.delta < 0 ? '▼' : '■'}{' '}
            {kpi.delta > 0 ? '+' : ''}
            {kpi.delta.toFixed(1)}
            {kpi.delta_unit === 'pp' ? ' pp' : '%'}
          </span>
        )}
      </div>
      {kpi.sub && (
        <div className="mt-1 truncate text-[11px] text-faint" title={kpi.sub}>
          {kpi.sub}
        </div>
      )}
    </div>
  )
}
