import type { Kpi } from '../api/client'
import * as format from '../lib/format'

function display(k: Kpi): string {
  // One decimal, matching every other place the same figure appears.
  // The dashboard tile read "19.86%" beside a report and an insights
  // page both saying 19.9%.
  if (k.format === 'pct') return format.pct(k.value)
  if (k.format === 'int') return format.count(k.value)
  return format.num(k.value)
}

export default function KpiCard({ kpi }: { kpi: Kpi }) {
  const context = kpi.benchmark || kpi.note
  return (
    <div className="rounded-xl border border-edge bg-panel px-4 py-3">
      <div
        className="text-[11px] font-medium tracking-wide text-mute uppercase"
        title={kpi.source_column ? `From ${kpi.source_column}` : undefined}
      >
        {kpi.label}
      </div>
      <div className="mt-1 font-data text-2xl font-semibold text-ink">
        {display(kpi)}
      </div>
      {kpi.mean !== undefined && (
        <div className="font-data text-[11px] text-mute">
          avg {kpi.mean.toLocaleString(undefined, { maximumFractionDigits: 1 })}
        </div>
      )}
      {context && (
        <div className="mt-1 text-[10px] leading-snug text-mute">{context}</div>
      )}
    </div>
  )
}
