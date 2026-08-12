import { useEffect, useState } from 'react'
import { Gauge, Sparkles, Target } from 'lucide-react'
import { apiGet, apiPost } from '../api/client'
import { useApp } from '../store/app'
import { Btn, ErrorBox, NeedData, PageHeader, Panel, Spinner } from '../components/Ui'

interface BenchmarkContext {
  metric: string
  value: number
  reference: number
  reference_kind: string
  direction: number
  gap: number
  headroom_pct: number
  meets: boolean
  interpretation: string
}

interface IndustryEntry {
  column: string
  benchmark: { low: number; high: number; unit: string; note: string }
  context: string
}

interface DriverResult {
  target: string
  auc: number
  accuracy: number
  n_rows: number
  n_features: number
  top_drivers: [string, number][]
  high_risk_profile: string
  high_risk_rate: number
  base_rate: number
  high_risk_n: number
  model_name: string
}

interface ScenarioResult {
  driver_col: string
  target_col: string
  change_pct: number
  current_driver_mean: number
  current_target_mean: number
  projected_target_mean: number
  projected_change_pct: number
  r_squared: number
  p_value: number
  reliable: boolean
  interpretation: string
  caveat: string
}

const num = (v: number, d = 2) =>
  v.toLocaleString(undefined, { maximumFractionDigits: d })

export default function DeepAnalysisPage() {
  const dataset = useApp((s) => s.dataset)
  const ds = dataset?.dataset_id

  const [benchmarks, setBenchmarks] = useState<BenchmarkContext[]>([])
  const [industry, setIndustry] = useState<IndustryEntry[]>([])
  const [domain, setDomain] = useState('')
  const [drivers, setDrivers] = useState<DriverResult | null>(null)
  const [driversMsg, setDriversMsg] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // what-if
  const [numericCols, setNumericCols] = useState<string[]>([])
  const [driverCol, setDriverCol] = useState('')
  const [targetCol, setTargetCol] = useState('')
  const [changePct, setChangePct] = useState(10)
  const [scenario, setScenario] = useState<ScenarioResult | null>(null)
  const [scenarioBusy, setScenarioBusy] = useState(false)
  const [scenarioErr, setScenarioErr] = useState('')

  useEffect(() => {
    if (!ds) return
    setLoading(true)
    setError('')
    setDrivers(null)
    setDriversMsg('')
    setScenario(null)

    apiGet<{ benchmarks: BenchmarkContext[] }>(`/api/analytics/${ds}/benchmarks`)
      .then((r) => setBenchmarks(r.benchmarks))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))

    apiGet<{ domain: string; benchmarks: IndustryEntry[] }>(
      `/api/analytics/${ds}/industry-benchmarks`,
    )
      .then((r) => {
        setDomain(r.domain)
        setIndustry(r.benchmarks)
      })
      .catch(() => setIndustry([]))

    apiGet<DriverResult>(`/api/analytics/${ds}/drivers`)
      .then(setDrivers)
      .catch((e) =>
        setDriversMsg(e instanceof Error ? e.message : String(e)),
      )

    apiGet<{ numeric_columns: string[] }>(`/api/analytics/${ds}/scenario/fields`)
      .then((r) => {
        setNumericCols(r.numeric_columns)
        setDriverCol(r.numeric_columns[0] ?? '')
        setTargetCol(r.numeric_columns[1] ?? r.numeric_columns[0] ?? '')
      })
      .catch(() => setNumericCols([]))
  }, [ds])

  if (!dataset) return <NeedData />

  const runScenario = async () => {
    setScenarioBusy(true)
    setScenarioErr('')
    try {
      setScenario(
        await apiPost<ScenarioResult>(`/api/analytics/${ds}/scenario`, {
          driver_col: driverCol,
          target_col: targetCol,
          change_pct: changePct,
        }),
      )
    } catch (e) {
      setScenarioErr(e instanceof Error ? e.message : String(e))
      setScenario(null)
    } finally {
      setScenarioBusy(false)
    }
  }

  return (
    <div className="p-8">
      <PageHeader
        title="Deep Analysis"
        subtitle="Benchmarks, what predicts your outcome, and what-if projections"
      />

      {error && (
        <div className="mb-4">
          <ErrorBox message={error} />
        </div>
      )}
      {loading && <Spinner label="Computing benchmarks…" />}

      <div className="space-y-5">
        {/* ── What-if simulator ─────────────────────────────── */}
        <Panel title="What-if simulator">
          <div className="flex flex-wrap items-end gap-3">
            <Select
              label="If this changes"
              value={driverCol}
              onChange={setDriverCol}
              options={numericCols}
            />
            <div>
              <label className="mb-1 block text-[11px] text-mute uppercase">
                By
              </label>
              <div className="flex items-center gap-1">
                <input
                  type="number"
                  value={changePct}
                  onChange={(e) => setChangePct(Number(e.target.value))}
                  className="w-20 rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink"
                />
                <span className="text-sm text-mute">%</span>
              </div>
            </div>
            <Select
              label="What happens to"
              value={targetCol}
              onChange={setTargetCol}
              options={numericCols}
            />
            <Btn
              onClick={runScenario}
              disabled={scenarioBusy || !driverCol || !targetCol}
            >
              <span className="flex items-center gap-1.5">
                <Sparkles className="h-4 w-4" /> Project
              </span>
            </Btn>
          </div>

          {scenarioErr && (
            <div className="mt-3">
              <ErrorBox message={scenarioErr} />
            </div>
          )}
          {scenarioBusy && <Spinner label="Fitting the relationship…" />}

          {scenario && !scenarioBusy && (
            <div className="mt-4 space-y-3">
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <MiniStat
                  label={`${scenario.target_col} now`}
                  value={num(scenario.current_target_mean)}
                />
                <MiniStat
                  label="Projected"
                  value={num(scenario.projected_target_mean)}
                  tone={scenario.projected_change_pct >= 0 ? 'teal' : 'rose'}
                />
                <MiniStat
                  label="Change"
                  value={`${scenario.projected_change_pct >= 0 ? '+' : ''}${scenario.projected_change_pct.toFixed(1)}%`}
                  tone={scenario.projected_change_pct >= 0 ? 'teal' : 'rose'}
                />
                <MiniStat
                  label="R² / p"
                  value={`${scenario.r_squared.toFixed(2)} / ${scenario.p_value.toFixed(3)}`}
                  tone={scenario.reliable ? 'teal' : 'amber'}
                />
              </div>

              <div
                className={`rounded-lg border px-4 py-3 text-sm ${
                  scenario.reliable
                    ? 'border-teal/40 bg-teal/5 text-ink'
                    : 'border-amber/40 bg-amber/5 text-ink'
                }`}
              >
                <div className="mb-1 font-semibold">
                  {scenario.reliable
                    ? 'Relationship is strong enough to project'
                    : 'Too weak to rely on'}
                </div>
                {scenario.interpretation}
              </div>
              <p className="text-xs text-mute">⚠ {scenario.caveat}</p>
            </div>
          )}
        </Panel>

        {/* ── Internal benchmarks ───────────────────────────── */}
        {benchmarks.length > 0 && (
          <Panel title="Each metric vs its own best quartile">
            <div className="space-y-3">
              {benchmarks.map((b) => (
                <div key={b.metric}>
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-ink">
                      {b.metric}
                      <span className="ml-2 font-normal text-mute">
                        {b.direction < 0 ? '(lower is better)' : '(higher is better)'}
                      </span>
                    </span>
                    <span className={b.meets ? 'text-teal' : 'text-amber'}>
                      {b.meets
                        ? 'at or above its own benchmark'
                        : `${Math.abs(b.headroom_pct).toFixed(1)}% headroom`}
                    </span>
                  </div>
                  <div className="mt-1 flex items-center gap-3 text-xs">
                    <span className="w-24 text-mute">now {num(b.value)}</span>
                    <div className="relative h-2.5 flex-1 overflow-hidden rounded-full bg-panel2">
                      <div
                        className={`h-full rounded-full ${b.meets ? 'bg-teal' : 'bg-accent'}`}
                        style={{
                          width: `${Math.min(100, Math.max(4, b.reference === 0 ? 0 : (b.value / b.reference) * 100))}%`,
                        }}
                      />
                    </div>
                    <span className="w-32 text-right text-mute">
                      target {num(b.reference)}
                    </span>
                  </div>
                  <p className="mt-1 text-[11px] text-mute">{b.interpretation}</p>
                </div>
              ))}
            </div>
          </Panel>
        )}

        {/* ── Industry benchmarks ───────────────────────────── */}
        {industry.length > 0 && (
          <Panel
            title={`Industry reference ranges${domain ? ` — ${domain}` : ''}`}
          >
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="text-mute">
                  <th className="px-2 py-2">Metric</th>
                  <th className="px-2 py-2">Typical range</th>
                  <th className="px-2 py-2">Context</th>
                </tr>
              </thead>
              <tbody>
                {industry.map((e) => (
                  <tr key={e.column} className="border-t border-edge/60">
                    <td className="px-2 py-2 font-semibold text-ink">{e.column}</td>
                    <td className="px-2 py-2 whitespace-nowrap text-accent">
                      {e.benchmark.low}–{e.benchmark.high}
                      {e.benchmark.unit}
                    </td>
                    <td className="px-2 py-2 text-mute">{e.context}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
        )}

        {/* ── Predictive drivers ────────────────────────────── */}
        <Panel
          title={
            drivers
              ? `What predicts ${drivers.target}`
              : 'What predicts your outcome'
          }
        >
          {!drivers && driversMsg && (
            <p className="text-sm text-mute">{driversMsg}</p>
          )}
          {!drivers && !driversMsg && <Spinner label="Fitting model…" />}

          {drivers && (
            <>
              <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
                <MiniStat label="Model" value={drivers.model_name} />
                <MiniStat
                  label="AUC"
                  value={drivers.auc.toFixed(3)}
                  tone={drivers.auc >= 0.7 ? 'teal' : 'amber'}
                />
                {/* base_rate / high_risk_rate arrive already expressed as
                    percentages (predictive.py multiplies by 100), so they
                    must not be scaled again here. */}
                <MiniStat
                  label="Base rate"
                  value={`${drivers.base_rate.toFixed(1)}%`}
                />
                <MiniStat
                  label="High-risk rate"
                  value={`${drivers.high_risk_rate.toFixed(1)}%`}
                  tone="rose"
                />
              </div>

              <div className="space-y-1.5">
                {drivers.top_drivers.map(([feature, importance]) => (
                  <div key={feature} className="flex items-center gap-3 text-xs">
                    <span className="w-44 truncate font-semibold text-ink">
                      {feature}
                    </span>
                    <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-panel2">
                      <div
                        className="h-full rounded-full bg-violet"
                        style={{
                          width: `${(importance / (drivers.top_drivers[0]?.[1] || 1)) * 100}%`,
                        }}
                      />
                    </div>
                    <span className="w-14 text-right text-mute">
                      {importance.toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>

              {drivers.high_risk_profile && (
                <div className="mt-4 rounded-lg border border-rose/40 bg-rose/5 px-4 py-3 text-sm text-ink">
                  <div className="mb-1 flex items-center gap-1.5 font-semibold text-rose">
                    <Target className="h-4 w-4" /> Highest-risk profile
                    <span className="font-normal text-mute">
                      ({drivers.high_risk_n} records)
                    </span>
                  </div>
                  {drivers.high_risk_profile}
                </div>
              )}
            </>
          )}
        </Panel>

        {benchmarks.length === 0 && !loading && (
          <Panel>
            <p className="flex items-center gap-2 text-sm text-mute">
              <Gauge className="h-4 w-4" />
              No directional numeric metrics found to benchmark in this dataset.
            </p>
          </Panel>
        )}
      </div>
    </div>
  )
}

function Select({
  label,
  value,
  onChange,
  options,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  options: string[]
}) {
  return (
    <div className="min-w-44">
      <label className="mb-1 block text-[11px] text-mute uppercase">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </div>
  )
}

function MiniStat({
  label,
  value,
  tone = 'ink',
}: {
  label: string
  value: string
  tone?: 'ink' | 'teal' | 'amber' | 'rose'
}) {
  const tones = {
    ink: 'text-ink',
    teal: 'text-teal',
    amber: 'text-amber',
    rose: 'text-rose',
  }
  return (
    <div className="rounded-lg border border-edge bg-panel2 px-3 py-2">
      <div className="text-[11px] text-mute uppercase">{label}</div>
      <div className={`text-base font-bold ${tones[tone]}`}>{value}</div>
    </div>
  )
}
