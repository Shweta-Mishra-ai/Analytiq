import type { Kpi } from '../api/client'

function fmt(k: Kpi): string {
  if (k.format === 'pct') return `${k.value}%`
  if (k.format === 'int') return k.value.toLocaleString()
  const v = k.value
  if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`
  if (Math.abs(v) >= 1_000) return `${(v / 1_000).toFixed(1)}K`
  return v.toLocaleString(undefined, { maximumFractionDigits: 2 })
}

export default function KpiCard({ kpi }: { kpi: Kpi }) {
  return (
    <div className="rounded-xl border border-edge bg-panel px-4 py-3">
      <div className="text-[11px] font-medium tracking-wide text-mute uppercase">
        {kpi.label}
      </div>
      <div className="mt-1 font-data text-2xl font-semibold text-ink">
        {fmt(kpi)}
      </div>
      {kpi.mean !== undefined && (
        <div className="font-data text-[11px] text-mute">
          avg {kpi.mean.toLocaleString(undefined, { maximumFractionDigits: 1 })}
        </div>
      )}
    </div>
  )
}
