import { useEffect, useState } from 'react'
import { apiGet } from '../api/client'
import { useApp } from '../store/app'
import Outlook from '../components/Outlook'
import { ErrorBox, NeedData, PageHeader, Panel, Spinner } from '../components/Ui'

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
  /** Replacement cost range, when the data has a numeric salary column. */
  cost_low?: number | null
  cost_high?: number | null
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

function compact(v: number): string {
  const a = Math.abs(v)
  if (a >= 1e9) return `${(v / 1e9).toFixed(1)}B`
  if (a >= 1e6) return `${(v / 1e6).toFixed(1)}M`
  if (a >= 1e3) return `${(v / 1e3).toFixed(0)}K`
  return v.toFixed(0)
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
    <div className="page">
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

          {ds && <Outlook ds={ds} />}

          {story.attrition && (
            <Panel title="Attrition analysis">
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <div>
                  <div className="text-xs text-mute uppercase">Rate</div>
                  <div className="text-2xl font-bold text-rose">
                    {/* rate is already a percentage (19.6), as in the PDF;
                        multiplying by 100 showed 1960.0% */}
                    {story.attrition.rate.toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div className="text-xs text-mute uppercase">Left</div>
                  <div className="text-2xl font-bold text-ink">
                    {story.attrition.n_left.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-mute uppercase">Flight risk</div>
                  <div className="text-2xl font-bold text-amber">
                    {story.attrition.n_flight_risk.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-mute uppercase">Est. replacement cost</div>
                  <div className="text-2xl font-bold text-ink">
                    {story.attrition.cost_low != null && story.attrition.cost_high != null
                      ? `${compact(story.attrition.cost_low)}–${compact(story.attrition.cost_high)}`
                      : '—'}
                  </div>
                </div>
              </div>
              <p className="mt-3 text-xs text-mute">{story.attrition.interpretation}</p>
              {/* The basis of the cost figure; it used to fill the figure's
                  own slot in the large metric font. */}
              <p className="mt-1 text-xs text-mute">{story.attrition.cost_estimate}.</p>
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
