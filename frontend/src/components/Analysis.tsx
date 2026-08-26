/**
 * Displays for the analytical depth the backend computes.
 *
 * All of this — decision bands, model selection, calibration, interactions,
 * confidence intervals, thin groups, class imbalance — was already being
 * calculated and serialised by the API, and rendered nowhere. It reached
 * the reader only if they generated a PDF first.
 */
import {
  AlertTriangle,
  CheckCircle2,
  Info,
  Layers,
  Scale,
  Target,
} from 'lucide-react'

// ── shapes the API already returns ────────────────────────

export interface DecisionBand {
  budget_pct: number
  n_targeted: number
  n_events_caught: number
  total_events: number
  precision: number
  recall: number
  lift: number
}

export interface ModelChoice {
  name: string
  auc: number
  candidates: [string, number][]
  threshold: number
  threshold_basis: string
  calibrated: boolean
  calibration_before: number | null
  calibration_after: number | null
  excluded_high_cardinality: [string, number][]
}

export interface ModelVerdict {
  usable: boolean
  baseline_score: number
  baseline_strategy: string
  model_score: number
  metric: string
  lift: number
  auc: number | null
  minority_recall: number | null
  reason: string
  verdict: string
}

export interface LeakageFinding {
  column: string
  separation: number
  reason: string
}

export interface Estimate {
  column: string
  statistic: string
  value: number
  ci_low: number
  ci_high: number
  n: number
}

export interface Interaction {
  metric: string
  factor: string
  moderator: string
  effect_by_level: Record<string, number>
  reverses: boolean
  ratio: number
  description: string
  effect_sd: number
}

export interface RareCategory {
  column: string
  level: string
  n: number
  share: number
}

export interface ImbalanceNote {
  column: string
  majority_level: string
  majority_share: number
  note: string
}

const num = (v: number, digits = 0) =>
  v.toLocaleString(undefined, { maximumFractionDigits: digits })

// ── no signal ─────────────────────────────────────────────

/**
 * What the report says when a model cannot beat the obvious guess.
 * Shown as a finding rather than an error: "we looked and found nothing"
 * is a real result, and a blank panel reads as a broken feature.
 */
export function NoSignal({ verdict }: { verdict: ModelVerdict }) {
  return (
    <div className="rounded-lg border border-amber/40 bg-amber/5 px-4 py-3">
      <div className="mb-1.5 flex items-center gap-1.5 text-sm font-semibold text-amber">
        <AlertTriangle className="h-4 w-4" />
        No predictive signal found
      </div>
      <p className="text-sm leading-relaxed text-ink-2">{verdict.verdict}</p>
      <p className="mt-2 text-xs text-mute">
        This is a finding, not a gap in the analysis. Collecting the factors
        thought to drive this outcome — or recording them earlier in the
        process — is what a usable model would need.
      </p>
    </div>
  )
}

// ── where to act ──────────────────────────────────────────

/**
 * The question a manager actually asks: we can contact 200 people this
 * month — which 200, and how many were going to leave anyway?
 */
export function DecisionBands({ bands }: { bands: DecisionBand[] }) {
  if (!bands?.length) return null
  const best = bands.reduce((a, b) => (b.lift > a.lift ? b : a), bands[0])
  return (
    <div>
      <div className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-ink">
        <Target className="h-4 w-4 text-teal" />
        Where to act
      </div>
      <div className="overflow-x-auto rounded-lg border border-edge">
        <table className="w-full min-w-[34rem] text-sm">
          <thead>
            <tr className="bg-panel-2 text-[11px] uppercase tracking-wide text-mute">
              <th className="px-3 py-2 text-left font-medium">If you act on</th>
              <th className="px-3 py-2 text-right font-medium">Records</th>
              <th className="px-3 py-2 text-right font-medium">Events reached</th>
              <th className="px-3 py-2 text-right font-medium">Hit rate</th>
              <th className="px-3 py-2 text-right font-medium">Share of events</th>
              <th className="px-3 py-2 text-right font-medium">vs random</th>
            </tr>
          </thead>
          <tbody className="font-data">
            {bands.map((b) => (
              <tr
                key={b.budget_pct}
                className="border-t border-edge/60 tabular-nums"
              >
                <td className="px-3 py-2 text-left text-ink">
                  top {b.budget_pct}%
                </td>
                <td className="px-3 py-2 text-right text-ink-2">
                  {num(b.n_targeted)}
                </td>
                <td className="px-3 py-2 text-right text-ink-2">
                  {num(b.n_events_caught)} of {num(b.total_events)}
                </td>
                <td className="px-3 py-2 text-right text-ink">
                  {b.precision.toFixed(0)}%
                </td>
                <td className="px-3 py-2 text-right text-ink-2">
                  {b.recall.toFixed(0)}%
                </td>
                <td
                  className={`px-3 py-2 text-right font-semibold ${
                    b.lift >= 1.5 ? 'text-teal' : 'text-mute'
                  }`}
                >
                  {b.lift.toFixed(1)}x
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-mute">
        Targeting the top {best.budget_pct}% reaches{' '}
        {num(best.n_events_caught)} of {num(best.total_events)} events, with{' '}
        {best.precision.toFixed(0)}% of those contacted recording it —{' '}
        {best.lift.toFixed(1)} times better than choosing at random. Which row
        to choose is a budget decision, not a modelling one.
      </p>
    </div>
  )
}

// ── how the model was chosen ──────────────────────────────

export function ModelSelection({ choice }: { choice: ModelChoice }) {
  return (
    <div>
      <div className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-ink">
        <Scale className="h-4 w-4 text-teal" />
        How this model was selected
      </div>
      <div className="flex flex-col gap-1">
        {choice.candidates.map(([name, auc]) => {
          const won = name === choice.name
          return (
            <div key={name} className="flex items-center gap-3 text-sm">
              <span
                className={`w-44 shrink-0 truncate ${
                  won ? 'font-semibold text-ink' : 'text-mute'
                }`}
              >
                {name}
                {won && <span className="ml-1.5 text-[11px] text-teal">selected</span>}
              </span>
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-panel-2">
                <div
                  className={`h-full rounded-full ${won ? 'bg-teal' : 'bg-edge'}`}
                  style={{ width: `${Math.max((auc - 0.5) * 200, 2)}%` }}
                />
              </div>
              <span className="w-12 shrink-0 text-right font-data tabular-nums text-ink-2">
                {auc.toFixed(3)}
              </span>
            </div>
          )
        })}
      </div>

      <div className="mt-3 flex flex-col gap-1.5 text-xs leading-relaxed text-mute">
        <p>
          Selected on cross-validated ranking quality, not by preference — on a
          different dataset a different one wins.
        </p>
        {choice.calibrated && choice.calibration_before != null && (
          <p>
            <span className="text-ink-2">Scores are calibrated.</span> They were{' '}
            {choice.calibration_before.toFixed(0)} percentage points from
            observed risk in the top band; they are now{' '}
            {(choice.calibration_after ?? 0).toFixed(0)}. A score of 0.30 now
            genuinely means roughly a 30% chance, so it can be quoted and not
            only ranked.
          </p>
        )}
        <p>
          <span className="text-ink-2">
            Operating threshold {choice.threshold.toFixed(2)}
          </span>{' '}
          rather than the conventional 0.50, because it {choice.threshold_basis}.
        </p>
        {choice.excluded_high_cardinality?.length > 0 && (
          <p>
            Excluded for having too many distinct values to learn from:{' '}
            {choice.excluded_high_cardinality
              .slice(0, 3)
              .map(([c, n]) => `${c} (${num(n)} values)`)
              .join(', ')}
            . These need grouping into fewer categories before a model can use
            them.
          </p>
        )}
      </div>
    </div>
  )
}

// ── leakage ───────────────────────────────────────────────

export function LeakageNotes({ findings }: { findings: LeakageFinding[] }) {
  if (!findings?.length) return null
  return (
    <div className="rounded-lg border border-rose/40 bg-rose/5 px-4 py-3">
      <div className="mb-1.5 flex items-center gap-1.5 text-sm font-semibold text-rose">
        <AlertTriangle className="h-4 w-4" />
        Fields excluded as outcome leakage
      </div>
      <div className="flex flex-col gap-2">
        {findings.slice(0, 3).map((f) => (
          <p key={f.column} className="text-sm leading-relaxed text-ink-2">
            <span className="font-data text-ink">{f.column}</span> — {f.reason}
          </p>
        ))}
      </div>
    </div>
  )
}

// ── estimates with intervals ──────────────────────────────

/**
 * A point estimate printed alone invites the reader to treat sampling
 * noise as a change worth acting on.
 */
export function Estimates({ estimates }: { estimates: Estimate[] }) {
  if (!estimates?.length) return null
  return (
    <div>
      <div className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-ink">
        <Info className="h-4 w-4 text-teal" />
        Headline measures, with uncertainty
      </div>
      <div className="overflow-x-auto rounded-lg border border-edge">
        <table className="w-full min-w-[30rem] text-sm">
          <thead>
            <tr className="bg-panel-2 text-[11px] uppercase tracking-wide text-mute">
              <th className="px-3 py-2 text-left font-medium">Measure</th>
              <th className="px-3 py-2 text-right font-medium">Mean</th>
              <th className="px-3 py-2 text-right font-medium">95% interval</th>
              <th className="px-3 py-2 text-right font-medium">Records</th>
            </tr>
          </thead>
          <tbody className="font-data">
            {estimates.map((e) => (
              <tr key={e.column} className="border-t border-edge/60 tabular-nums">
                <td className="px-3 py-2 text-left text-ink">
                  {e.column.replace(/_/g, ' ')}
                </td>
                <td className="px-3 py-2 text-right text-ink">
                  {num(e.value, 2)}
                </td>
                <td className="px-3 py-2 text-right text-ink-2">
                  {num(e.ci_low, 2)} – {num(e.ci_high, 2)}
                </td>
                <td className="px-3 py-2 text-right text-mute">{num(e.n)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-mute">
        A difference smaller than the interval is not evidence of a change.
      </p>
    </div>
  )
}

// ── interactions ──────────────────────────────────────────

/**
 * The finding a main-effects summary reports as "no effect", because it
 * averages a large positive and a large negative into nothing.
 */
export function Interactions({ interactions }: { interactions: Interaction[] }) {
  if (!interactions?.length) return null
  return (
    <div>
      <div className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-ink">
        <Layers className="h-4 w-4 text-teal" />
        Effects that differ by group
      </div>
      <div className="flex flex-col gap-2">
        {interactions.map((i, idx) => (
          <div
            key={`${i.metric}-${i.factor}-${i.moderator}-${idx}`}
            className="rounded-lg border border-edge bg-panel-2/40 px-3 py-2.5"
          >
            <div className="mb-1 flex flex-wrap items-center gap-1.5 text-xs">
              {i.reverses && (
                <span className="rounded bg-rose/15 px-1.5 py-0.5 font-data text-[10px] uppercase tracking-wide text-rose">
                  direction reverses
                </span>
              )}
              <span className="font-data text-[10px] uppercase tracking-wide text-mute">
                {i.effect_sd.toFixed(1)} SD
              </span>
            </div>
            <p className="text-sm leading-relaxed text-ink-2">{i.description}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── caveats ───────────────────────────────────────────────

export function DataCaveats({
  rare,
  imbalance,
}: {
  rare: RareCategory[]
  imbalance: ImbalanceNote[]
}) {
  if (!rare?.length && !imbalance?.length) return null
  return (
    <div className="flex flex-col gap-2">
      {imbalance?.slice(0, 2).map((n) => (
        <div
          key={n.column}
          className="rounded-lg border border-amber/40 bg-amber/5 px-3 py-2.5"
        >
          <p className="text-sm leading-relaxed text-ink-2">{n.note}</p>
        </div>
      ))}
      {rare?.length > 0 && (
        <div className="rounded-lg border border-edge bg-panel-2/40 px-3 py-2.5">
          <div className="mb-1 flex items-center gap-1.5 text-xs font-semibold text-mute">
            <CheckCircle2 className="h-3.5 w-3.5" />
            Excluded from group comparisons
          </div>
          <p className="text-sm leading-relaxed text-ink-2">
            {rare.length} category level
            {rare.length === 1 ? '' : 's'} hold too few rows to support a
            finding — the smallest is{' '}
            <span className="font-data">
              {rare[0].level}
            </span>{' '}
            in{' '}
            <span className="font-data">{rare[0].column}</span> at {rare[0].n}{' '}
            row{rare[0].n === 1 ? '' : 's'}. Ranking these alongside groups many
            times their size would present noise as a result.
          </p>
        </div>
      )}
    </div>
  )
}
