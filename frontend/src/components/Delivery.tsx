/**
 * Delivery — where a report goes after it is built.
 *
 * The backend gained three things the UI could not reach: a report that
 * survives the request that asked for it, a link that opens it for
 * someone with no account, and a schedule that produces one without
 * anyone asking. A capability nobody can press is not a feature, so
 * this is the surface for all three.
 *
 * The share link is the part that needs care in the wording as well as
 * the code. It widens access, so the panel says plainly what it grants,
 * when it stops working, and offers withdrawing it in the same place —
 * not buried in a settings page found after the link has already been
 * sent to the wrong address.
 */
import { useCallback, useEffect, useState } from 'react'
import {
  Ban,
  CalendarClock,
  Check,
  Clock,
  Download,
  Link2,
  Loader2,
  Mail,
  Trash2,
} from 'lucide-react'
import { apiBlob, apiDelete, apiGet, apiPost, downloadBlob } from '../api/client'
import { Badge, Btn, EmptyState, ErrorBox, Panel } from './Ui'

interface Artifact {
  artifact_id: string
  kind: string
  format: string
  filename: string
  title: string
  size_bytes: number
  created_at: number
  expires_at: number
  shared: boolean
  share_views: number
  shares_revoked: boolean
}

interface Schedule {
  schedule_id: string
  dataset_id: string
  kind: string
  cadence: string
  hour: number
  recipients: string[]
  enabled: boolean
  last_run_at: number
  last_status: string
  last_artifact_id: string
  next_run_at: number
}

const KIND_LABEL: Record<string, string> = {
  report: 'Analysis report',
  'health-report': 'Health report',
  deck: 'Deck',
}

/** "in 3 days" / "2 hours ago" — a bare timestamp makes the reader do
 *  arithmetic to answer the only question they have. */
function when(epochSeconds: number): string {
  if (!epochSeconds) return '—'
  const seconds = epochSeconds - Date.now() / 1000
  const ahead = seconds > 0
  const abs = Math.abs(seconds)
  const [n, unit] =
    abs < 90 ? [Math.round(abs), 'second']
    : abs < 5400 ? [Math.round(abs / 60), 'minute']
    : abs < 172800 ? [Math.round(abs / 3600), 'hour']
    : [Math.round(abs / 86400), 'day']
  const plural = `${n} ${unit}${n === 1 ? '' : 's'}`
  return ahead ? `in ${plural}` : `${plural} ago`
}

function size(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function Delivery({ datasetId }: { datasetId: string }) {
  const [artifacts, setArtifacts] = useState<Artifact[] | null>(null)
  const [schedules, setSchedules] = useState<Schedule[]>([])
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  // The minted link, kept beside the report it belongs to so it is
  // obvious which one is on the clipboard.
  const [link, setLink] = useState<{ id: string; url: string; expires: number } | null>(null)
  const [copied, setCopied] = useState(false)

  const [cadence, setCadence] = useState('weekly')
  const [kind, setKind] = useState('health-report')
  const [hour, setHour] = useState(7)
  const [recipients, setRecipients] = useState('')

  const load = useCallback(async () => {
    if (!datasetId) return
    try {
      const [a, s] = await Promise.all([
        apiGet<{ artifacts: Artifact[] }>(`/api/artifacts?dataset_id=${datasetId}`),
        apiGet<{ schedules: Schedule[] }>('/api/schedules'),
      ])
      // Defensive against a response that is not the shape expected —
      // a proxy error page, a partial deploy. A delivery panel that
      // throws takes the whole Reports page down with it, and the
      // report the user came for is the thing they lose.
      setArtifacts(Array.isArray(a?.artifacts) ? a.artifacts : [])
      setSchedules(
        Array.isArray(s?.schedules)
          ? s.schedules.filter((x) => x.dataset_id === datasetId)
          : [],
      )
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setArtifacts([])
    }
  }, [datasetId])

  useEffect(() => {
    setLink(null)
    setArtifacts(null)
    void load()
  }, [load])

  const act = async (tag: string, fn: () => Promise<unknown>) => {
    setBusy(tag)
    setError('')
    try {
      await fn()
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy('')
    }
  }

  const download = (art: Artifact) =>
    act(`get:${art.artifact_id}`, async () => {
      const blob = await apiBlob(`/api/artifacts/${art.artifact_id}/download`)
      downloadBlob(blob, art.filename)
    })

  const share = (art: Artifact) =>
    act(`share:${art.artifact_id}`, async () => {
      const res = await apiPost<{ url: string; expires_at: number }>(
        `/api/artifacts/${art.artifact_id}/share`,
        { ttl_days: 7 },
      )
      setLink({ id: art.artifact_id, url: res.url, expires: res.expires_at })
      setCopied(false)
    })

  const revoke = (art: Artifact) =>
    act(`revoke:${art.artifact_id}`, async () => {
      await apiDelete(`/api/artifacts/${art.artifact_id}/share`)
      setLink((cur) => (cur?.id === art.artifact_id ? null : cur))
    })

  const copy = async (url: string) => {
    try {
      await navigator.clipboard.writeText(url)
      setCopied(true)
    } catch {
      // A browser that refuses the clipboard is not an error worth
      // showing — the link is on screen and selectable.
      setCopied(false)
    }
  }

  const addSchedule = () =>
    act('schedule', async () => {
      await apiPost('/api/schedules', {
        dataset_id: datasetId,
        kind,
        cadence,
        hour,
        recipients: recipients
          .split(',')
          .map((r) => r.trim())
          .filter(Boolean),
      })
      setRecipients('')
    })

  const input =
    'rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink placeholder:text-mute focus:border-accent focus:outline-none'

  return (
    <>
      {error && (
        <div className="mb-4">
          <ErrorBox message={error} />
        </div>
      )}

      <Panel
        title="Reports you have built"
        subtitle="Kept for 30 days, then removed — a report is a photograph of the data at a moment, and an old one misleads."
      >
        {artifacts === null ? (
          <p className="text-sm text-mute">Loading…</p>
        ) : artifacts.length === 0 ? (
          <EmptyState
            title="Nothing built yet"
            hint="Generate a report above and it will be kept here, ready to download again or send to someone without an account."
            icon={<Clock className="h-6 w-6" />}
          />
        ) : (
          <ul className="space-y-2">
            {artifacts.map((art) => (
              <li
                key={art.artifact_id}
                className="rounded-xl border border-edge bg-panel2 px-4 py-3"
              >
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
                  <span className="text-sm font-semibold text-ink">
                    {art.title || KIND_LABEL[art.kind] || art.kind}
                  </span>
                  <Badge>{art.format.toUpperCase()}</Badge>
                  {art.shares_revoked ? (
                    <Badge tone="rose">link withdrawn</Badge>
                  ) : art.shared ? (
                    <Badge tone="teal">
                      shared · {art.share_views} {art.share_views === 1 ? 'view' : 'views'}
                    </Badge>
                  ) : null}
                  <span className="ml-auto text-xs text-mute">
                    {size(art.size_bytes)} · built {when(art.created_at)} ·
                    removed {when(art.expires_at)}
                  </span>
                </div>

                <div className="mt-2.5 flex flex-wrap gap-2">
                  <Btn
                    size="sm"
                    variant="subtle"
                    onClick={() => download(art)}
                    disabled={!!busy}
                  >
                    <span className="flex items-center gap-1.5">
                      {busy === `get:${art.artifact_id}` ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <Download className="h-3.5 w-3.5" />
                      )}
                      Download
                    </span>
                  </Btn>
                  <Btn
                    size="sm"
                    variant="ghost"
                    onClick={() => share(art)}
                    disabled={!!busy}
                  >
                    <span className="flex items-center gap-1.5">
                      <Link2 className="h-3.5 w-3.5" />
                      {art.shared && !art.shares_revoked ? 'New link' : 'Get a link'}
                    </span>
                  </Btn>
                  {art.shared && !art.shares_revoked && (
                    <Btn
                      size="sm"
                      variant="ghost"
                      onClick={() => revoke(art)}
                      disabled={!!busy}
                    >
                      <span className="flex items-center gap-1.5">
                        <Ban className="h-3.5 w-3.5" />
                        Withdraw
                      </span>
                    </Btn>
                  )}
                  <Btn
                    size="sm"
                    variant="ghost"
                    onClick={() =>
                      act(`del:${art.artifact_id}`, () =>
                        apiDelete(`/api/artifacts/${art.artifact_id}`),
                      )
                    }
                    disabled={!!busy}
                  >
                    <span className="flex items-center gap-1.5">
                      <Trash2 className="h-3.5 w-3.5" />
                      Delete
                    </span>
                  </Btn>
                </div>

                {link?.id === art.artifact_id && (
                  <div className="mt-3 rounded-lg border border-teal/30 bg-teal/[0.06] px-3 py-2.5">
                    <div className="flex items-center gap-2">
                      <code className="min-w-0 flex-1 truncate text-xs text-ink2">
                        {link.url}
                      </code>
                      <Btn size="sm" variant="subtle" onClick={() => copy(link.url)}>
                        <span className="flex items-center gap-1.5">
                          {copied ? (
                            <Check className="h-3.5 w-3.5" />
                          ) : (
                            <Link2 className="h-3.5 w-3.5" />
                          )}
                          {copied ? 'Copied' : 'Copy'}
                        </span>
                      </Btn>
                    </div>
                    <p className="mt-1.5 text-[11px] leading-relaxed text-mute">
                      Opens this one report for anyone who has the link, with
                      no sign-in — and grants nothing else: not your data, not
                      your other reports. It stops working{' '}
                      {when(link.expires)}, and you can withdraw it before
                      then.
                    </p>
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </Panel>

      <Panel
        title="Send it on a schedule"
        subtitle="A report that arrives on its own, without anyone opening the app."
        className="mt-5"
      >
        {schedules.length > 0 && (
          <ul className="mb-4 space-y-2">
            {schedules.map((s) => (
              <li
                key={s.schedule_id}
                className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-xl border border-edge bg-panel2 px-4 py-3"
              >
                <CalendarClock className="h-4 w-4 shrink-0 text-accent" />
                <span className="text-sm text-ink">
                  {KIND_LABEL[s.kind] ?? s.kind}, {s.cadence} at{' '}
                  {String(s.hour).padStart(2, '0')}:00
                </span>
                {s.recipients.length > 0 ? (
                  <Badge tone="teal">
                    <span className="flex items-center gap-1">
                      <Mail className="h-3 w-3" />
                      {s.recipients.join(', ')}
                    </span>
                  </Badge>
                ) : (
                  <Badge>waits here</Badge>
                )}
                <span className="ml-auto text-xs text-mute">
                  next {when(s.next_run_at)}
                  {s.last_status && ` · last: ${s.last_status}`}
                </span>
                <Btn
                  size="sm"
                  variant="ghost"
                  disabled={!!busy}
                  onClick={() =>
                    act(`sched:${s.schedule_id}`, () =>
                      apiDelete(`/api/schedules/${s.schedule_id}`),
                    )
                  }
                >
                  Remove
                </Btn>
              </li>
            ))}
          </ul>
        )}

        <div className="flex flex-wrap items-end gap-3">
          <div>
            <label className="mb-1 block text-xs text-mute" htmlFor="sched-kind">
              Report
            </label>
            <select
              id="sched-kind"
              value={kind}
              onChange={(e) => setKind(e.target.value)}
              className={input}
            >
              <option value="health-report">Health report</option>
              <option value="report">Analysis report</option>
            </select>
          </div>
          <div>
            <label className="mb-1 block text-xs text-mute" htmlFor="sched-cadence">
              How often
            </label>
            <select
              id="sched-cadence"
              value={cadence}
              onChange={(e) => setCadence(e.target.value)}
              className={input}
            >
              <option value="daily">Every day</option>
              <option value="weekly">Every week</option>
              <option value="monthly">Every month</option>
            </select>
          </div>
          <div>
            <label className="mb-1 block text-xs text-mute" htmlFor="sched-hour">
              At
            </label>
            <select
              id="sched-hour"
              value={hour}
              onChange={(e) => setHour(Number(e.target.value))}
              className={input}
            >
              {Array.from({ length: 24 }, (_, h) => (
                <option key={h} value={h}>
                  {String(h).padStart(2, '0')}:00
                </option>
              ))}
            </select>
          </div>
          <div className="min-w-[14rem] flex-1">
            <label className="mb-1 block text-xs text-mute" htmlFor="sched-to">
              Email it to (optional, comma separated)
            </label>
            <input
              id="sched-to"
              value={recipients}
              onChange={(e) => setRecipients(e.target.value)}
              placeholder="cfo@client.com"
              className={`w-full ${input}`}
            />
          </div>
          <Btn onClick={addSchedule} disabled={!!busy || !datasetId}>
            {busy === 'schedule' ? 'Adding…' : 'Add schedule'}
          </Btn>
        </div>
        <p className="mt-2 text-xs text-mute">
          Left without an address the report is still built on time and waits
          here for you. Emailing needs SMTP configured on the server; until it
          is, the schedule says so rather than quietly sending nothing.
        </p>
      </Panel>
    </>
  )
}
