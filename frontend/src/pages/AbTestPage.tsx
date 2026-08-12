import { useEffect, useState } from 'react'
import { FlaskRound } from 'lucide-react'
import { apiGet, apiPost } from '../api/client'
import { useApp } from '../store/app'
import { Btn, ErrorBox, NeedData, PageHeader, Panel, Spinner } from '../components/Ui'

interface Fields {
  group_columns: string[]
  metric_columns: string[]
}

interface AbResult {
  test_type: 'conversion' | 'continuous'
  metric_name: string
  variant_a_name: string
  variant_b_name: string
  n_a: number
  n_b: number
  conversions_a: number | null
  conversions_b: number | null
  rate_a: number | null
  rate_b: number | null
  mean_a: number | null
  mean_b: number | null
  median_a: number | null
  median_b: number | null
  test_used: string
  p_value: number
  is_significant: boolean
  confidence_level: number
  relative_uplift: number
  absolute_diff: number
  ci_lower: number
  ci_upper: number
  power: number | null
  sample_size_adequate: boolean
  verdict: string
  recommendation: string
  warnings: string[]
}

const num = (v: number | null, digits = 2) =>
  v == null ? '—' : v.toLocaleString(undefined, { maximumFractionDigits: digits })

export default function AbTestPage() {
  const dataset = useApp((s) => s.dataset)
  const ds = dataset?.dataset_id

  const [fields, setFields] = useState<Fields | null>(null)
  const [groupCol, setGroupCol] = useState('')
  const [metricCol, setMetricCol] = useState('')
  const [confidence, setConfidence] = useState(0.95)
  const [result, setResult] = useState<AbResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!ds) return
    setResult(null)
    setError('')
    apiGet<Fields>(`/api/analytics/${ds}/ab-test/fields`)
      .then((f) => {
        setFields(f)
        setGroupCol(f.group_columns[0] ?? '')
        setMetricCol(f.metric_columns[0] ?? '')
      })
      .catch((e) => setError(e.message))
  }, [ds])

  if (!dataset) return <NeedData />

  const run = async () => {
    setBusy(true)
    setError('')
    try {
      setResult(
        await apiPost<AbResult>(`/api/analytics/${ds}/ab-test`, {
          group_col: groupCol,
          metric_col: metricCol,
          confidence_level: confidence,
        }),
      )
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setResult(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="p-8">
      <PageHeader
        title="A/B Test"
        subtitle="Is the difference between two groups real, or just noise?"
      />

      {error && (
        <div className="mb-4">
          <ErrorBox message={error} />
        </div>
      )}

      <Panel title="Set up the comparison">
        <div className="flex flex-wrap items-end gap-3">
          <Select
            label="Split by"
            value={groupCol}
            onChange={setGroupCol}
            options={fields?.group_columns ?? []}
          />
          <Select
            label="Compare this metric"
            value={metricCol}
            onChange={setMetricCol}
            options={fields?.metric_columns ?? []}
          />
          <div>
            <label className="mb-1 block text-[11px] text-mute uppercase">
              Confidence
            </label>
            <select
              value={confidence}
              onChange={(e) => setConfidence(Number(e.target.value))}
              className="rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink"
            >
              <option value={0.9}>90%</option>
              <option value={0.95}>95%</option>
              <option value={0.99}>99%</option>
            </select>
          </div>
          <Btn onClick={run} disabled={busy || !groupCol || !metricCol}>
            <span className="flex items-center gap-1.5">
              <FlaskRound className="h-4 w-4" /> Run test
            </span>
          </Btn>
        </div>
        <p className="mt-2 text-xs text-mute">
          The two largest groups in the chosen column are compared. A two-value
          metric is treated as a conversion rate; anything else is compared on
          its average.
        </p>
      </Panel>

      {busy && <Spinner label="Running significance test…" />}

      {result && !busy && (
        <div className="mt-5 space-y-5">
          <Panel
            className={
              result.is_significant
                ? 'border-teal/40 bg-teal/5'
                : 'border-amber/40 bg-amber/5'
            }
          >
            <div className="text-xs uppercase text-mute">
              {result.test_used}
            </div>
            <div
              className={`mt-1 text-lg font-bold ${
                result.is_significant ? 'text-teal' : 'text-amber'
              }`}
            >
              {result.is_significant
                ? 'Significant difference'
                : 'No significant difference'}
            </div>
            <p className="mt-2 text-sm text-ink">{result.verdict}</p>
            <p className="mt-2 text-sm font-semibold text-ink">
              → {result.recommendation}
            </p>
          </Panel>

          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="p-value" value={result.p_value.toFixed(4)} />
            <Stat
              label="Difference"
              value={
                result.test_type === 'conversion'
                  ? `${(result.absolute_diff * 100).toFixed(2)} pp`
                  : num(result.absolute_diff)
              }
            />
            <Stat
              label="95% CI on difference"
              value={`${num(result.ci_lower, 3)} … ${num(result.ci_upper, 3)}`}
            />
            <Stat
              label="Power"
              value={result.power == null ? '—' : `${(result.power * 100).toFixed(0)}%`}
              tone={
                result.power != null && result.power < 0.8 ? 'amber' : 'teal'
              }
            />
          </div>

          <Panel title="Side by side">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="text-mute">
                  <th className="px-2 py-2">Variant</th>
                  <th className="px-2 py-2">n</th>
                  {result.test_type === 'conversion' ? (
                    <>
                      <th className="px-2 py-2">Conversions</th>
                      <th className="px-2 py-2">Rate</th>
                    </>
                  ) : (
                    <>
                      <th className="px-2 py-2">Mean</th>
                      <th className="px-2 py-2">Median</th>
                    </>
                  )}
                </tr>
              </thead>
              <tbody>
                {[
                  {
                    name: result.variant_a_name,
                    n: result.n_a,
                    conv: result.conversions_a,
                    rate: result.rate_a,
                    mean: result.mean_a,
                    median: result.median_a,
                  },
                  {
                    name: result.variant_b_name,
                    n: result.n_b,
                    conv: result.conversions_b,
                    rate: result.rate_b,
                    mean: result.mean_b,
                    median: result.median_b,
                  },
                ].map((v) => (
                  <tr key={v.name} className="border-t border-edge/60">
                    <td className="px-2 py-2 font-semibold text-ink">{v.name}</td>
                    <td className="px-2 py-2 text-mute">{v.n.toLocaleString()}</td>
                    {result.test_type === 'conversion' ? (
                      <>
                        <td className="px-2 py-2 text-mute">{v.conv ?? '—'}</td>
                        <td className="px-2 py-2 text-mute">
                          {v.rate == null ? '—' : `${(v.rate * 100).toFixed(2)}%`}
                        </td>
                      </>
                    ) : (
                      <>
                        <td className="px-2 py-2 text-mute">{num(v.mean)}</td>
                        <td className="px-2 py-2 text-mute">{num(v.median)}</td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>

          {result.warnings.length > 0 && (
            <Panel title="Read this before deciding">
              <ul className="space-y-1 text-xs text-amber">
                {result.warnings.map((w, i) => (
                  <li key={i}>⚠ {w}</li>
                ))}
              </ul>
            </Panel>
          )}
        </div>
      )}
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
    <div className="min-w-48">
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

function Stat({
  label,
  value,
  tone = 'ink',
}: {
  label: string
  value: string
  tone?: 'ink' | 'teal' | 'amber'
}) {
  const tones = { ink: 'text-ink', teal: 'text-teal', amber: 'text-amber' }
  return (
    <Panel>
      <div className="text-xs text-mute uppercase">{label}</div>
      <div className={`text-lg font-bold ${tones[tone]}`}>{value}</div>
    </Panel>
  )
}
