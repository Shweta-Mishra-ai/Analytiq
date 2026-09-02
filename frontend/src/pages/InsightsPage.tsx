import { useEffect, useState } from 'react'
import { apiGet } from '../api/client'
import { useApp } from '../store/app'
import { ErrorBox, NeedData, PageHeader, Panel, Spinner } from '../components/Ui'
import * as fmt from '../lib/format'

interface Insight {
  title: string
  problem: string
  cause: string
  evidence: string
  action: string
  impact: string
  severity: string
}

interface Attrition {
  rate: number
  n_left: number
  n_total: number
  severity: string
  cost_estimate: string
  interpretation: string
  n_flight_risk: number
}

interface Story {
  domain: string
  domain_confidence: number
  headline: string
  executive_summary: string
  top_insights: Insight[]
  attrition?: Attrition
  key_findings: string[]
  business_risks: string[]
  opportunities: string[]
  recommended_actions: string[]
  analysis_confidence: string
  anomalies: string[]
}

const sevStyle: Record<string, string> = {
  critical: 'border-rose/50 bg-rose/10',
  warning: 'border-amber/50 bg-amber/10',
  positive: 'border-teal/50 bg-teal/10',
  info: 'border-accent/50 bg-accent/10',
}

function InsightCard({ ins, n }: { ins: Insight; n: number }) {
  return (
    <div className={`rounded-xl border p-4 ${sevStyle[ins.severity] ?? sevStyle.info}`}>
      <div className="mb-2 flex items-center gap-2">
        <span className="rounded-full bg-panel px-2 py-0.5 text-[10px] font-bold text-mute">
          #{n}
        </span>
        <h3 className="text-sm font-bold text-ink">{ins.title}</h3>
        <span className="ml-auto text-[10px] tracking-wide text-mute uppercase">
          {ins.severity}
        </span>
      </div>
      <dl className="grid gap-x-4 gap-y-1.5 text-xs sm:grid-cols-2">
        {(
          [
            ['Problem', ins.problem],
            ['Cause', ins.cause],
            ['Evidence', ins.evidence],
            ['Action', ins.action],
          ] as const
        ).map(([k, v]) => (
          <div key={k}>
            <dt className="font-semibold text-mute">{k}</dt>
            <dd className="text-ink/90">{v}</dd>
          </div>
        ))}
      </dl>
      {ins.impact && (
        <div className="mt-2 text-xs">
          <span className="font-semibold text-mute">Impact: </span>
          <span className="text-ink/90">{ins.impact}</span>
        </div>
      )}
    </div>
  )
}

export default function InsightsPage() {
  const dataset = useApp((s) => s.dataset)
  const [story, setStory] = useState<Story | null>(null)
  const [error, setError] = useState('')
  const ds = dataset?.dataset_id

  useEffect(() => {
    if (!ds) return
    setStory(null)
    setError('')
    apiGet<Story>(`/api/analytics/${ds}/story`)
      .then(setStory)
      .catch((e) => setError(e.message))
  }, [ds])

  if (!dataset) return <NeedData />

  return (
    <div className="p-8">
      <PageHeader
        title="Business Insights"
        subtitle="Problem → Cause → Evidence → Action → Impact"
      />
      {error && <ErrorBox message={error} />}
      {!story && !error && <Spinner label="Generating analyst narrative…" />}

      {story && (
        <div className="space-y-5">
          <Panel>
            <div className="flex items-center gap-3">
              <span className="rounded-full bg-accent/15 px-3 py-1 text-xs font-bold text-accent uppercase">
                {story.domain}
              </span>
              <span className="text-xs text-mute">
                confidence: {story.analysis_confidence}
              </span>
            </div>
            <h2 className="mt-3 text-lg font-bold text-ink">{story.headline}</h2>
            <p className="mt-2 text-sm leading-relaxed text-mute">
              {story.executive_summary}
            </p>
          </Panel>

          {story.attrition && (
            <Panel title="Attrition analysis">
              <div className="grid grid-cols-3 gap-3">
                {(
                  [
                    // The rate arrives already expressed as a percentage.
                    // Multiplying again put "1990.0%" on the headline
                    // tile of a dataset with 19.9% attrition.
                    ['Rate', fmt.pct(story.attrition.rate), 'text-rose',
                     `${fmt.count(story.attrition.n_left)} of ${fmt.count(
                       story.attrition.n_total)}`],
                    ['Left', fmt.count(story.attrition.n_left), 'text-ink',
                     'over the period covered'],
                    ['Same risk profile',
                     fmt.count(story.attrition.n_flight_risk), 'text-amber',
                     'of those still here'],
                  ] as const
                ).map(([label, val, tone, sub]) => (
                  <div key={label}>
                    <div className="text-[11px] uppercase tracking-wide text-mute">
                      {label}
                    </div>
                    <div className={`font-data text-2xl font-bold ${tone}`}>
                      {val}
                    </div>
                    <div className="mt-0.5 text-[11px] text-faint">{sub}</div>
                  </div>
                ))}
              </div>
              {/* Prose belongs in prose, not in a 24px number tile: the
                  cost estimate is a sentence and was being rendered in
                  the display face beside three figures. */}
              <p className="mt-4 border-t border-edge pt-3 text-xs leading-relaxed text-ink2">
                {story.attrition.cost_estimate}
              </p>
              <p className="mt-2 text-xs text-mute">
                {story.attrition.interpretation}
              </p>
            </Panel>
          )}

          <div className="space-y-3">
            {story.top_insights.map((ins, i) => (
              <InsightCard key={i} ins={ins} n={i + 1} />
            ))}
          </div>

          <div className="grid gap-5 lg:grid-cols-3">
            {(
              [
                ['Business risks', story.business_risks, 'text-rose'],
                ['Opportunities', story.opportunities, 'text-teal'],
                ['Recommended actions', story.recommended_actions, 'text-accent'],
              ] as const
            ).map(([title, items, color]) => (
              <Panel key={title} title={title}>
                <ul className="space-y-1.5 text-xs text-mute">
                  {items.length ? (
                    items.map((x, i) => (
                      <li key={i} className="flex gap-2">
                        <span className={color}>▸</span> {x}
                      </li>
                    ))
                  ) : (
                    <li>—</li>
                  )}
                </ul>
              </Panel>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
