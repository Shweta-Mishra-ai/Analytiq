import { useEffect, useState } from 'react'
import { Brain } from 'lucide-react'
import { apiGet, apiPost } from '../api/client'
import { LeakageNotes, NoSignal } from '../components/Analysis'
import type { LeakageFinding, ModelVerdict } from '../components/Analysis'
import { useApp } from '../store/app'
import { Btn, ErrorBox, NeedData, PageHeader, Panel, Spinner } from '../components/Ui'
import * as fmt from '../lib/format'

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
  /** Whether the model beat the obvious guess. When it did not, its
   *  feature importances describe the noise it was fitted to. */
  verdict?: ModelVerdict | null
  leakage?: LeakageFinding[]
  /** {feature: {min, max}} — the span the model was fitted over. */
  feature_ranges?: Record<string, { min: number; max: number }>
}

interface WhatIf {
  prediction: number
  prediction_label?: string
  confidence?: number | null
  lower?: number
  upper?: number
  confidence_note?: string
  /** False when an input sits outside what the model was fitted on. A
   *  model answers any question put to it; this says whether it had any
   *  standing to. */
  within_training_range: boolean
  out_of_range: string[]
  range_note?: string
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
    // Load a previously trained report if there is one. "Not trained
    // yet" is an answer, not a failure — the endpoint returns
    // { report: null } for it rather than a 404, so an ordinary first
    // visit no longer logs an error in the user's console.
    apiGet<MlReport | { report: null }>(`/api/ml/${ds}/report`)
      .then((r) => {
        if (r && 'report' in r && r.report === null) return
        setReport(r as MlReport)
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

  return (
    <div className="p-8">
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
                  {fmt.label(t.column)} — {t.task}
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

      {busy && <Spinner label="Training and cross-validating models…" />}

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
                {report.best_model?.metric_name ?? 'Score'}
              </div>
              <div className="text-xl font-bold text-teal">
                {report.best_model ? fmtScore(report.best_model.test_score) : '—'}
              </div>
            </Panel>
            <Panel>
              <div className="text-xs text-mute uppercase">Rows / features</div>
              <div className="text-xl font-bold text-ink">
                {report.n_rows_used.toLocaleString()} / {report.n_features}
              </div>
            </Panel>
          </div>

          <Panel title="Model leaderboard">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="text-mute">
                    {['Model', 'CV score', 'Test score', 'Overfit', 'Extra'].map((h) => (
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
            <div className="space-y-1.5">
              {report.feature_importance.slice(0, 12).map((f) => (
                <div key={f.feature} className="flex items-center gap-3 text-xs">
                  <span className="w-44 truncate font-semibold text-ink" title={f.feature}>
                    {fmt.label(f.feature)}
                  </span>
                  <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-panel2">
                    <div
                      className="h-full rounded-full bg-accent"
                      style={{
                        width: `${(f.importance / (report.feature_importance[0]?.importance || 1)) * 100}%`,
                      }}
                    />
                  </div>
                  <span className="w-56 truncate text-mute">{f.explanation}</span>
                </div>
              ))}
            </div>
          </Panel>

          {report.verdict && !report.verdict.usable && (
            <Panel title="Model verdict">
              <NoSignal verdict={report.verdict} />
            </Panel>
          )}

          {report.leakage && report.leakage.length > 0 && (
            <Panel title="Excluded fields">
              <LeakageNotes findings={report.leakage} />
            </Panel>
          )}

          {/* Ask the model a question.
              The endpoint existed and no page called it, so a user could
              train a model and never use it for the one thing a model is
              for. Only numeric features with a known span are offered —
              those are the ones a prediction can be sanity-checked
              against. */}
          {report.feature_ranges &&
            Object.keys(report.feature_ranges).length > 0 && (
              <WhatIfPanel ds={ds!} report={report} />
            )}

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


/**
 * "What would the model predict if…" — the interactive half of a trained
 * model, and the place a user is most likely to walk past the edge of
 * the evidence, because typing a big number is the obvious thing to try.
 *
 * The prediction is always shown. What changes when an input is out of
 * range is that the figure is presented as an extrapolation rather than
 * a finding, and the panel says which input and what the data covers —
 * a model fitted on salaries of 30k–120k answered a question about
 * 5,000,000 with a confidence interval and no caveat at all.
 */
function WhatIfPanel({ ds, report }: { ds: string; report: MlReport }) {
  const ranges = report.feature_ranges ?? {}
  const features = Object.keys(ranges).slice(0, 6)

  const [values, setValues] = useState<Record<string, string>>(() =>
    Object.fromEntries(
      features.map((f) => [
        f,
        String(Math.round(((ranges[f].min + ranges[f].max) / 2) * 100) / 100),
      ]),
    ),
  )
  const [result, setResult] = useState<WhatIf | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  async function run() {
    setBusy(true)
    setErr('')
    try {
      const inputs: Record<string, number> = {}
      for (const f of features) {
        const n = Number(values[f])
        if (Number.isFinite(n)) inputs[f] = n
      }
      setResult(
        await apiPost<WhatIf>(`/api/ml/${ds}/what-if`, {
          target: report.target_col,
          inputs,
        }),
      )
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
      setResult(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel title={`What would ${fmt.label(report.target_col)} be?`}>
      <p className="mb-3 text-xs text-mute">
        Set the inputs and the trained model will predict{' '}
        {fmt.label(report.target_col)}. Each field shows the range the model
        was actually fitted on — a value far outside it is guesswork, and the
        answer will say so.
      </p>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {features.map((f) => (
          <div key={f}>
            <label className="mb-1 block text-[11px] text-mute uppercase">
              {fmt.label(f)}
            </label>
            <input
              type="number"
              value={values[f] ?? ''}
              onChange={(e) =>
                setValues((v) => ({ ...v, [f]: e.target.value }))
              }
              className="w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink"
            />
            <div className="mt-1 text-[10px] text-faint">
              data covers {fmt.num(ranges[f].min)} – {fmt.num(ranges[f].max)}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4">
        <Btn onClick={run} disabled={busy}>
          {busy ? 'Predicting…' : 'Predict'}
        </Btn>
      </div>

      {err && (
        <div className="mt-3">
          <ErrorBox message={err} />
        </div>
      )}

      {result && !busy && (
        <div
          className={`mt-4 rounded-lg border px-4 py-3 ${
            result.within_training_range
              ? 'border-teal/40 bg-teal/5'
              : 'border-amber/40 bg-amber/5'
          }`}
        >
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-ink">
              {result.prediction_label ?? fmt.num(result.prediction)}
            </span>
            <span className="text-xs text-mute">
              {fmt.label(report.target_col)}
            </span>
          </div>
          {result.confidence != null && (
            <div className="mt-1 text-xs text-mute">
              {result.confidence}% confident
            </div>
          )}
          {result.lower != null && result.upper != null && (
            <div className="mt-1 text-xs text-mute">
              likely between {fmt.num(result.lower)} and {fmt.num(result.upper)}
            </div>
          )}
          {result.range_note && (
            <p className="mt-2 text-xs leading-relaxed text-ink2">
              <span className="font-semibold text-amber">
                Outside what the model has seen:{' '}
              </span>
              {result.range_note.replace(
                'This prediction is outside what the model has seen: ',
                '',
              )}
            </p>
          )}
        </div>
      )}
    </Panel>
  )
}
