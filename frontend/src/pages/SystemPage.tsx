import { useCallback, useEffect, useState } from 'react'
import {
  Activity,
  CheckCircle2,
  Cpu,
  Loader2,
  Lock,
  ServerCog,
  XCircle,
} from 'lucide-react'
import { apiGet, apiPost } from '../api/client'
import { Badge, Btn, ErrorBox, PageHeader, Panel, Stat } from '../components/Ui'
import * as fmt from '../lib/format'
import TaskRouting from '../components/TaskRouting'

interface ProviderRow {
  name: string
  label: string
  configured: boolean
  model: string
  free: boolean
  local: boolean
  missing: string
}

interface Status {
  providers: Record<string, ProviderRow>
  configured: string[]
  routing: Record<string, string>
  order: string[]
  privacy_mode: boolean
  any_available: boolean
}

interface CheckRow extends ProviderRow {
  ok: boolean
  latency_ms: number
  reply: string
  error: string
  hint: string
}

interface CheckResult {
  checked_at: string
  providers: CheckRow[]
  working: string[]
  any_working: boolean
  summary: string
  privacy_mode: boolean
}

export default function SystemPage() {
  const [status, setStatus] = useState<Status | null>(null)
  const [check, setCheck] = useState<CheckResult | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    apiGet<Status>('/api/admin/llm-status')
      .then(setStatus)
      .catch((e: Error) => setError(e.message))
  }, [])

  const runCheck = useCallback(() => {
    setRunning(true)
    setError('')
    apiPost<CheckResult>('/api/admin/llm-check')
      .then(setCheck)
      .catch((e: Error) => setError(e.message))
      .finally(() => setRunning(false))
  }, [])

  const rows: (ProviderRow | CheckRow)[] = check
    ? check.providers
    : status
      ? Object.values(status.providers)
      : []

  return (
    <div className="space-y-6">
      <PageHeader
        title="System"
        subtitle="Which language models this deployment can reach, and whether the keys it holds actually work."
      />

      {error && <ErrorBox message={error} />}

      {status && (
        <div className="grid gap-4 sm:grid-cols-3">
          <Stat
            label="Providers configured"
            value={fmt.count(status.configured.length)}
            hint={
              status.configured.length
                ? status.configured.join(', ')
                : 'None — reports still build in the engines’ own wording'
            }
          />
          <Stat
            label="Privacy mode"
            value={status.privacy_mode ? 'On' : 'Off'}
            hint={
              status.privacy_mode
                ? 'Cloud providers are refused; only a local model may be used'
                : 'Prompts may be sent to the configured cloud providers'
            }
          />
          <Stat
            label="Narrative source"
            value={status.any_available ? 'Model' : 'Engines'}
            hint={
              status.any_available
                ? 'A model phrases the prose over the computed figures'
                : 'Every figure and sentence is written by the analysis engines'
            }
          />
        </div>
      )}

      <Panel
        title="Language model providers"
        subtitle="A key can be present and still be rejected — expired, wrong account, out of quota, or blocked by this network. Only a real call finds that out."
        right={
          <Btn onClick={runCheck} disabled={running}>
            {running ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> Calling each provider…
              </>
            ) : (
              <>
                <Activity className="h-4 w-4" /> Run live check
              </>
            )}
          </Btn>
        }
      >
        {check && (
          <div
            className={`mb-4 rounded-lg border p-3 text-sm ${
              check.any_working
                ? 'border-teal/30 bg-teal/5 text-ink'
                : 'border-amber/30 bg-amber/5 text-ink'
            }`}
          >
            {check.summary}
            <div className="mt-1 text-xs text-mute">
              Checked {new Date(check.checked_at).toLocaleString()}
            </div>
          </div>
        )}

        <div className="space-y-2">
          {rows.map((p) => (
            <ProviderCard key={p.name} row={p} checked={!!check} />
          ))}
        </div>
      </Panel>

      <TaskRouting />
    </div>
  )
}

function ProviderCard({
  row,
  checked,
}: {
  row: ProviderRow | CheckRow
  checked: boolean
}) {
  const c = row as CheckRow
  const hasResult = checked && row.configured
  const good = hasResult && c.ok
  // After a check, the row comes from the check result, which carries
  // `error`/`hint` rather than `missing` — so an unconfigured provider
  // lost the one line that said which variable to set. Both shapes are
  // read here so the advice survives the button being pressed.
  const problem = checked ? c.error : row.missing
  const advice = checked ? c.hint : ''

  return (
    <div className="rounded-lg border border-edge p-3">
      <div className="flex flex-wrap items-center gap-2">
        {row.local ? (
          <Cpu className="h-4 w-4 text-mute" />
        ) : (
          <ServerCog className="h-4 w-4 text-mute" />
        )}
        <span className="font-medium">{row.label}</span>
        {row.free && <Badge tone="teal">Free tier</Badge>}
        {row.local && <Badge tone="neutral">On your hardware</Badge>}
        {!row.configured && <Badge tone="neutral">Not configured</Badge>}
        {hasResult &&
          (good ? (
            <span className="inline-flex items-center gap-1 text-xs text-teal">
              <CheckCircle2 className="h-3.5 w-3.5" /> answered in{' '}
              {fmt.count(c.latency_ms)} ms
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 text-xs text-rose">
              <XCircle className="h-3.5 w-3.5" /> failed
            </span>
          ))}
      </div>

      <div className="mt-1 text-xs text-mute">
        {row.model || 'no model set'}
      </div>

      {!row.configured && problem && (
        <div className="mt-2 text-xs text-mute">
          <Lock className="mr-1 inline h-3 w-3" />
          {problem}
        </div>
      )}

      {row.configured && hasResult && !good && problem && (
        <div className="mt-2">
          <div className="rounded bg-rose/5 px-2 py-1 font-mono text-xs text-rose">
            {problem}
          </div>
        </div>
      )}

      {advice && !good && (
        <div className="mt-1.5 text-xs text-faint">{advice}</div>
      )}
    </div>
  )
}
