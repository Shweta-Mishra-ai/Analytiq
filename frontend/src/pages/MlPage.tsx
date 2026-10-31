import { useEffect, useState } from 'react'
import { Brain, FlaskConical } from 'lucide-react'
import { apiGet, apiPost } from '../api/client'
import { useApp } from '../store/app'
import { Btn, ErrorBox, NeedData, PageHeader, Panel, Spinner } from '../components/Ui'

interface Target {
  column: string
  task: string
  reason: string
}

interface ModelResult {
  name: string
  task: string
  cv_score: number
  cv_std: number
  test_score: number
  overfit_label: string
  metric_name: string
  mae?: number
  rmse?: number
  roc_auc?: number
  is_best: boolean
}

interface FeatureImp {
  feature: string
  importance: number
  rank: number
  direction: string
  explanation: string
}

interface FeatureInput {
  name: string
  kind: 'category' | 'date' | 'number'
  options?: string[]
  min?: number | string | null
  max?: number | string | null
  default?: number | string | null
}

interface WhatIfResult {
  prediction: number
  prediction_label?: string
  probabilities?: Record<string, number>
  confidence?: number | null
  lower?: number
  upper?: number
  confidence_note?: string
  assumed_at_training_median?: string[]
}

interface MlReport {
  task: string
  target_col: string
  n_rows_used: number
  n_features: number
  class_balance?: Record<string, number>
  models: ModelResult[]
  best_model?: ModelResult
  feature_importance: FeatureImp[]
  warnings: string[]
  insights: string[]
  /** Set when the best model is no better than a trivial guess. */
  no_signal?: string
  feature_inputs?: FeatureInput[]
}

const field =
  'w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink'

/**
 * Score one profile with the trained model. Inputs start at the training
 * data's typical values, so changing one field shows that field's effect.
 */
function WhatIfPanel({ ds, report }: { ds: string; report: MlReport }) {
  const specs = report.feature_inputs ?? []
  const rank = new Map(report.feature_importance.map((f) => [f.feature, f.rank]))
  const ordered = [...specs].sort((a, b) => (rank.get(a.name) ?? 999) - (rank.get(b.name) ?? 999))
  const defaults = () =>
    Object.fromEntries(specs.map((f) => [f.name, f.default == null ? '' : String(f.default)]))
  const [inputs, setInputs] = useState<Record<string, string>>(defaults)
  const [result, setResult] = useState<WhatIfResult | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  if (!specs.length || !report.best_model) return null

  const run = async () => {
    setBusy(true)
    setError('')
    try {
      setResult(
        await apiPost<WhatIfResult>(`/api/ml/${ds}/what-if`, {
          target: report.target_col,
          inputs,
        }),
      )
    } catch (e) {
      setResult(null)
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const num = (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 2 })

  return (
    <Panel title={`What-if: predict ${report.target_col} for one profile`}>
      <p className="mb-3 text-xs text-mute">
        Starts from a typical row of the training data. Change any field and run — the result
        is what this model expects for that profile, a pattern in this data rather than proof
        that changing the field would change the outcome.
      </p>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        {ordered.map((f) => (
          <div key={f.name}>
            <label className="mb-1 block text-xs text-mute" htmlFor={`wi-${f.name}`}>
              {f.name}
            </label>
            {f.kind === 'category' ? (
              <select
                id={`wi-${f.name}`}
                value={inputs[f.name] ?? ''}
                onChange={(e) => setInputs({ ...inputs, [f.name]: e.target.value })}
                className={field}
              >
                {(f.options ?? []).map((o) => (
                  <option key={o}>{o}</option>
                ))}
              </select>
            ) : (
              <input
                id={`wi-${f.name}`}
                type={f.kind === 'date' ? 'date' : 'number'}
                step="any"
                min={f.min ?? undefined}
                max={f.max ?? undefined}
                value={inputs[f.name] ?? ''}
                onChange={(e) => setInputs({ ...inputs, [f.name]: e.target.value })}
                className={field}
              />
            )}
          </div>
        ))}
      </div>
      <div className="mt-4 flex items-center gap-3">
        <Btn onClick={run} disabled={busy}>
          <span className="flex items-center gap-1.5">
            <FlaskConical className="h-4 w-4" /> {busy ? 'Predicting…' : 'Predict'}
          </span>
        </Btn>
        <Btn variant="ghost" onClick={() => { setInputs(defaults()); setResult(null) }} disabled={busy}>
          Reset to typical
        </Btn>
      </div>
      {error && (
        <div className="mt-3">
          <ErrorBox message={error} />
        </div>
      )}
      {result && (
        <div className="mt-4 rounded-lg border border-edge bg-panel2 p-4" data-testid="what-if-result">
          {report.no_signal && (
            <p className="mb-2 text-xs text-amber">⚠ {report.no_signal}</p>
          )}
          {report.task === 'classification' ? (
            <>
              <div className="text-xs text-mute uppercase">Predicted {report.target_col}</div>
              <div className="text-2xl font-bold text-accent">
                {result.prediction_label ?? num(result.prediction)}
                {result.confidence != null && (
                  <span className="ml-2 text-sm font-normal text-mute">
                    {result.confidence}% model probability
                  </span>
                )}
              </div>
              {result.probabilities && (
                <div className="mt-2 flex flex-wrap gap-3 text-xs text-mute">
                  {Object.entries(result.probabilities).map(([k, v]) => (
                    <span key={k}>{k}: {(v * 100).toFixed(1)}%</span>
                  ))}
                </div>
              )}
            </>
          ) : (
            <>
              <div className="text-xs text-mute uppercase">Predicted {report.target_col}</div>
              <div className="text-2xl font-bold text-accent">{num(result.prediction)}</div>
              {result.lower != null && result.upper != null && (
                <div className="mt-1 text-xs text-mute">
                  Typical error range {num(result.lower)} – {num(result.upper)} ({result.confidence_note})
                </div>
              )}
            </>
          )}
          {result.assumed_at_training_median && result.assumed_at_training_median.length > 0 && (
            <p className="mt-2 text-xs text-amber">
              Left blank, so assumed typical: {result.assumed_at_training_median.join(', ')}
            </p>
          )}
        </div>
      )}
    </Panel>
  )
}

export default function MlPage() {
  const dataset = useApp((s) => s.dataset)
  const [targets, setTargets] = useState<Target[]>([])
  const [selected, setSelected] = useState('')
  const [report, setReport] = useState<MlReport | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const ds = dataset?.dataset_id

  useEffect(() => {
    if (!ds) return
    setReport(null)
    setTargets([])
    setSelected('')
    apiGet<{ targets: Target[] }>(`/api/ml/${ds}/targets`)
      .then((r) => {
        setTargets(r.targets)
        if (r.targets[0]) setSelected(r.targets[0].column)
      })
      .catch((e) => setError(e.message))
    // load a previously trained report if it exists, and point the target
    // picker at it — it showed the first suggestion over another target's model
    apiGet<MlReport>(`/api/ml/${ds}/report`)
      .then((r) => {
        setReport(r)
        setSelected(r.target_col)
      })
      .catch(() => {})
  }, [ds])

  if (!dataset) return <NeedData />

  const train = async () => {
    setBusy(true)
    setError('')
    try {
      setReport(await apiPost<MlReport>(`/api/ml/${ds}/train`, { target: selected }))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const fmtScore = (v: number) => (v * 100).toFixed(1) + '%'
  const isClf = report?.task === 'classification'
  const baseline = report?.class_balance
    ? Object.entries(report.class_balance).sort((a, b) => b[1] - a[1])[0]
    : undefined
  const scoreHint = !report?.best_model
    ? ''
    : isClf && baseline
      ? `vs ${fmtScore(baseline[1])} from always answering '${baseline[0]}'`
      : 'share of the variation explained on unseen rows'

  return (
    <div className="page">
      <PageHeader
        title="ML Predictions"
        subtitle="Auto model selection with cross-validation and feature importance"
      />
      {error && (
        <div className="mb-4">
          <ErrorBox message={error} />
        </div>
      )}

      <Panel title="Pick a prediction target">
        <div className="flex flex-wrap items-end gap-3">
          <div className="min-w-64">
            <select
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
              className="w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink"
            >
              {targets.map((t) => (
                <option key={t.column} value={t.column}>
                  {t.column} — {t.task}
                </option>
              ))}
            </select>
            {targets.find((t) => t.column === selected)?.reason && (
              <p className="mt-1 text-xs text-mute">
                {targets.find((t) => t.column === selected)?.reason}
              </p>
            )}
          </div>
          <Btn onClick={train} disabled={busy || !selected}>
            <span className="flex items-center gap-1.5">
              <Brain className="h-4 w-4" /> Train models
            </span>
          </Btn>
        </div>
      </Panel>

      {busy && (
        <Spinner label="Training four models with cross-validation — a few seconds on small files, up to a minute on 50,000 rows…" />
      )}

      {report && !busy && (
        <div className="mt-5 space-y-5">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Panel>
              <div className="text-xs text-mute uppercase">Task</div>
              <div className="text-xl font-bold text-ink capitalize">{report.task}</div>
            </Panel>
            <Panel>
              <div className="text-xs text-mute uppercase">Best model</div>
              <div className="text-xl font-bold text-accent">
                {report.best_model?.name ?? '—'}
              </div>
            </Panel>
            <Panel>
              <div className="text-xs text-mute uppercase">
                {isClf ? 'Accuracy on unseen rows' : 'R² on unseen rows'}
              </div>
              <div className={`text-xl font-bold ${report.no_signal ? 'text-amber' : 'text-teal'}`}>
                {report.best_model ? fmtScore(report.best_model.test_score) : '—'}
              </div>
              {scoreHint && <div className="mt-0.5 text-[11px] text-faint">{scoreHint}</div>}
            </Panel>
            <Panel>
              <div className="text-xs text-mute uppercase">Rows / features</div>
              <div className="text-xl font-bold text-ink">
                {report.n_rows_used.toLocaleString()} / {report.n_features}
              </div>
            </Panel>
          </div>

          <Panel
            title="Model leaderboard"
            subtitle={
              isClf
                ? 'Picked on cross-validated log-loss, which penalises confident wrong answers; scores shown are accuracy.'
                : 'Picked on cross-validated R²; the test score is on rows no model saw in training.'
            }
          >
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="text-mute">
                    {['Model', isClf ? 'CV accuracy' : 'CV R²', 'Test score', 'Overfit', 'Extra'].map((h) => (
                      <th key={h} className="px-2 py-2">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {report.models.map((m) => (
                    <tr key={m.name}
                      className={`border-t border-edge/60 ${m.is_best ? 'bg-accent/5' : ''}`}>
                      <td className="px-2 py-2 font-semibold text-ink">
                        {m.name} {m.is_best && <span className="text-accent">★</span>}
                      </td>
                      <td className="px-2 py-2 text-mute">
                        {fmtScore(m.cv_score)} ± {(m.cv_std * 100).toFixed(1)}
                      </td>
                      <td className="px-2 py-2 text-mute">{fmtScore(m.test_score)}</td>
                      <td className={`px-2 py-2 ${
                        m.overfit_label === 'None' ? 'text-teal'
                          : m.overfit_label === 'Mild' ? 'text-amber' : 'text-rose'
                      }`}>
                        {m.overfit_label}
                      </td>
                      <td className="px-2 py-2 text-mute">
                        {m.mae != null && `MAE ${m.mae.toLocaleString(undefined, { maximumFractionDigits: 2 })}`}
                        {m.roc_auc != null && ` AUC ${m.roc_auc.toFixed(3)}`}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>

          <Panel title="What drives the prediction">
            {report.no_signal && (
              <p className="mb-3 rounded-lg border border-amber/40 bg-amber/5 px-4 py-3 text-sm text-ink">
                ⚠ {report.no_signal}
              </p>
            )}
            <div className={`space-y-3 ${report.no_signal ? 'opacity-60' : ''}`}>
              {report.feature_importance.slice(0, 12).map((f) => (
                <div key={f.feature} className="text-xs">
                  <div className="flex items-center gap-3">
                    <span className="w-32 shrink-0 truncate font-semibold text-ink sm:w-40" title={f.feature}>
                      {f.feature}
                    </span>
                    <div className="h-2 flex-1 overflow-hidden rounded-full bg-panel2">
                      <div
                        className="h-full rounded-full bg-accent"
                        style={{
                          width: `${(f.importance / (report.feature_importance[0]?.importance || 1)) * 100}%`,
                        }}
                      />
                    </div>
                    <span className="w-10 shrink-0 text-right font-data text-mute">
                      {(f.importance * 100).toFixed(0)}%
                    </span>
                  </div>
                  {/* the full sentence: truncated at 56 characters it read
                      "In the model, 'tenure' at 15 rather than…" */}
                  <p className="mt-1 text-mute sm:pl-43">{f.explanation}</p>
                </div>
              ))}
              {report.feature_importance.length === 0 && (
                <p className="text-sm text-mute">
                  No column changes the model's accuracy on unseen rows, so there is nothing to rank.
                </p>
              )}
            </div>
          </Panel>

          {ds && <WhatIfPanel key={`${ds}:${report.target_col}`} ds={ds} report={report} />}

          {report.insights.length > 0 && (
            <Panel title="Model insights">
              <ul className="space-y-1.5 text-sm text-mute">
                {report.insights.map((x, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="text-accent">▸</span> {x}
                  </li>
                ))}
              </ul>
            </Panel>
          )}

          {report.warnings.length > 0 && (
            <Panel title="Warnings">
              <ul className="space-y-1 text-xs text-amber">
                {report.warnings.map((w, i) => (
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
