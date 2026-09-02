/**
 * Which model does which job.
 *
 * The rule this table renders: a task can only be pointed at a model
 * that can actually do it. The eligible list comes from the server —
 * the capability check is not re-implemented here, because two
 * implementations of one rule is one implementation and one bug waiting
 * to disagree with it.
 */
import { useCallback, useEffect, useState } from 'react'
import { AlertTriangle, Loader2, Play, RotateCcw } from 'lucide-react'
import { apiDelete, apiGet, apiPost } from '../api/client'
import { Badge, Btn, ErrorBox, Panel } from '../components/Ui'
import * as fmt from '../lib/format'

interface TaskRow {
  task: string
  label: string
  description: string
  requires: string[]
  min_context: number
  degrades_to: string
  assigned: string
  source: string
  resolved: string[]
  eligible: string[]
  served: boolean
}

interface ModelRow {
  id: string
  provider: string
  label: string
  capabilities: string[]
  tier: string
  context: number
  free: boolean
  declared: boolean
  notes: string
}

interface Problem {
  task: string
  model_id: string
  kind: string
  detail: string
  source: string
}

interface RoutingStatus {
  tasks: TaskRow[]
  models: ModelRow[]
  problems: Problem[]
  deprecated: { from: string; to: string }[]
  capabilities: Record<string, string>
}

interface CheckResult {
  task: string
  ok: boolean
  model: string
  latency_ms?: number
  reply?: string
  error?: string
  hint?: string
}

/**
 * Every response that sets state goes through this, not just the first
 * load. A write endpoint answering with a slightly different shape from
 * the read endpoint is the kind of thing that only shows up as a blank
 * page after someone changes a setting — which is exactly when they are
 * least inclined to trust the feature.
 */
function normalise(r: Partial<RoutingStatus> | undefined): RoutingStatus {
  return {
    tasks: Array.isArray(r?.tasks) ? r.tasks : [],
    models: Array.isArray(r?.models) ? r.models : [],
    problems: Array.isArray(r?.problems) ? r.problems : [],
    deprecated: Array.isArray(r?.deprecated) ? r.deprecated : [],
    capabilities: r?.capabilities ?? {},
  }
}

const SOURCE_LABEL: Record<string, string> = {
  default: 'Default',
  env: 'From environment',
  runtime: 'Changed here',
}

export default function TaskRouting() {
  const [status, setStatus] = useState<RoutingStatus | null>(null)
  const [checks, setChecks] = useState<Record<string, CheckResult>>({})
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')

  const load = useCallback(() => {
    apiGet<RoutingStatus>('/api/admin/routing')
      .then((r) => setStatus(normalise(r)))
      .catch((e: Error) => setError(e.message))
  }, [])

  useEffect(load, [load])

  const assign = useCallback(
    (task: string, modelId: string) => {
      setBusy(task)
      setError('')
      apiPost<RoutingStatus>('/api/admin/routing', { task, model_id: modelId })
        .then((r) => setStatus(normalise(r)))
        .catch((e: Error) => setError(e.message))
        .finally(() => setBusy(''))
    },
    [],
  )

  const test = useCallback((task: string) => {
    setBusy(`test:${task}`)
    setError('')
    apiPost<CheckResult>(`/api/admin/task-check?task=${encodeURIComponent(task)}`)
      .then((r) => setChecks((prev) => ({ ...prev, [task]: r })))
      .catch((e: Error) => setError(e.message))
      .finally(() => setBusy(''))
  }, [])

  const reset = useCallback(() => {
    setBusy('reset')
    apiDelete<RoutingStatus>('/api/admin/routing')
      .then((r) => setStatus(normalise(r)))
      .catch((e: Error) => setError(e.message))
      .finally(() => setBusy(''))
  }, [])

  if (!status) {
    return (
      <Panel title="Task routing">
        {error ? <ErrorBox message={error} /> : <p className="text-sm text-mute">Loading…</p>}
      </Panel>
    )
  }

  const byId = new Map(status.models.map((m) => [m.id, m]))
  const changed = status.tasks.some((t) => t.source === 'runtime')

  return (
    <Panel
      title="Task routing"
      subtitle="Each job goes to a model that can do it. A model that lacks the capability a task needs is not offered here, and cannot be reached as a fallback either."
      right={
        changed ? (
          <Btn variant="ghost" size="sm" onClick={reset} disabled={!!busy}>
            <RotateCcw className="h-3.5 w-3.5" /> Reset to environment
          </Btn>
        ) : undefined
      }
    >
      {error && (
        <div className="mb-3">
          <ErrorBox message={error} />
        </div>
      )}

      {status.problems.length > 0 && (
        <div className="mb-4 space-y-2">
          {status.problems.map((p) => (
            <div
              key={`${p.task}-${p.model_id}`}
              className="flex gap-2 rounded-lg border border-amber/25 bg-amber/[0.04] px-3 py-2 text-sm text-ink2"
            >
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber" />
              <span>{p.detail}</span>
            </div>
          ))}
        </div>
      )}

      {status.deprecated.map((d) => (
        <p key={d.from} className="mb-2 text-xs text-amber">
          LLM_ROUTING still names <span className="font-mono">{d.from}</span>; that
          row now controls <span className="font-mono">{d.to}</span>. Update the
          variable when convenient — it still works.
        </p>
      ))}

      {status.tasks.length === 0 && !error && (
        <p className="text-sm text-mute">
          This server did not report any routable tasks.
        </p>
      )}

      <div className="space-y-2">
        {status.tasks.map((row) => {
          const check = checks[row.task]
          return (
            <div key={row.task} className="rounded-lg border border-edge p-3">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-medium text-ink">{row.label}</span>
                    {row.requires.map((cap) => (
                      <Badge key={cap} tone="neutral">
                        {status.capabilities[cap] ?? cap}
                      </Badge>
                    ))}
                    {row.source !== 'default' && (
                      <Badge tone="accent">{SOURCE_LABEL[row.source]}</Badge>
                    )}
                  </div>
                  <p className="mt-1 text-xs leading-relaxed text-mute">
                    {row.description}
                  </p>
                </div>
                <Btn
                  variant="ghost"
                  size="sm"
                  onClick={() => test(row.task)}
                  disabled={!!busy || !row.served}
                >
                  {busy === `test:${row.task}` ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Play className="h-3.5 w-3.5" />
                  )}{' '}
                  Test
                </Btn>
              </div>

              <div className="mt-2 flex flex-wrap items-center gap-2">
                <label className="sr-only" htmlFor={`model-${row.task}`}>
                  Model for {row.label}
                </label>
                <select
                  id={`model-${row.task}`}
                  value={row.assigned}
                  disabled={!!busy}
                  onChange={(e) => assign(row.task, e.target.value)}
                  className="rounded-md border border-edge bg-panel2 px-2 py-1.5 font-mono text-xs text-ink"
                >
                  <option value="">
                    {row.eligible.length
                      ? '— not assigned —'
                      : '— no capable model configured —'}
                  </option>
                  {row.eligible.map((id) => (
                    <option key={id} value={id}>
                      {byId.get(id)?.label ?? id}
                      {byId.get(id)?.free ? ' · free' : ''}
                    </option>
                  ))}
                </select>
                {row.resolved.length > 1 && (
                  <span className="text-xs text-faint">
                    falls back to{' '}
                    {row.resolved
                      .slice(1, 3)
                      .map((id) => byId.get(id)?.label ?? id)
                      .join(' → ')}
                  </span>
                )}
              </div>

              {!row.served && (
                <p className="mt-2 rounded-md border border-amber/25 bg-amber/[0.04] px-2 py-1.5 text-xs text-ink2">
                  {row.degrades_to}
                </p>
              )}

              {check && (
                <p
                  className={`mt-2 text-xs ${check.ok ? 'text-teal' : 'text-rose'}`}
                >
                  {check.ok
                    ? `${check.model} answered in ${fmt.count(check.latency_ms ?? 0)} ms`
                    : check.error}
                  {!check.ok && check.hint ? ` — ${check.hint}` : ''}
                </p>
              )}
            </div>
          )
        })}
      </div>
    </Panel>
  )
}
