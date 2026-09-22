import { useCallback, useEffect, useState } from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  CreditCard,
  Infinity as InfinityIcon,
  KeyRound,
  Loader2,
  UserCircle2,
} from 'lucide-react'
import { apiGet, apiPost } from '../api/client'
import { Badge, Btn, ErrorBox, PageHeader, Panel } from '../components/Ui'
import * as fmt from '../lib/format'

/**
 * What this account is on, what it has used, and how to move.
 *
 * The backend has had plans, quotas and Stripe checkout since signup
 * landed, and nothing in the UI called any of it — an account could be
 * created and then never see its own ceilings or a way past them. A
 * quota that only announces itself at the moment it refuses an upload
 * is a bad way to learn what you bought.
 */

interface Allowance {
  username: string
  email: string
  is_admin: boolean
  plan: string
  plan_label: string
  branding: boolean
  limits: Record<string, number | null>
  used: Record<string, number>
  remaining: Record<string, number | null>
  metered: boolean
}

interface PlanRow {
  key: string
  label: string
  price_inr_month: number
  datasets: number | null
  rows_per_dataset: number | null
  max_upload_mb: number | null
  reports_per_month: number | null
  scheduled_reports: number | null
  branding: boolean
  notes: string
}

interface Catalogue {
  plans: PlanRow[]
  checkout_available: boolean
  metered: boolean
  currency: string
}

/** The ceilings worth showing, in the order they get hit in practice. */
const METERS: { key: string; label: string; usedKey: string }[] = [
  { key: 'datasets', label: 'Datasets', usedKey: 'datasets' },
  { key: 'reports_per_month', label: 'Reports this month', usedKey: 'reports' },
  { key: 'models_per_month', label: 'Models this month', usedKey: 'models' },
]

/** A ceiling of null is no limit; zero is not included. Different things. */
function capText(cap: number | null | undefined): string {
  if (cap === null || cap === undefined) return 'Unlimited'
  if (cap === 0) return 'Not included'
  return fmt.count(cap)
}

function Meter({
  label,
  used,
  cap,
}: {
  label: string
  used: number
  cap: number | null
}) {
  if (cap === null) {
    return (
      <div className="flex items-baseline justify-between gap-3 py-2.5">
        <span className="text-sm text-ink2">{label}</span>
        <span className="flex items-center gap-1.5 text-sm text-mute">
          <span className="tabular-nums text-ink">{fmt.count(used)}</span>
          <span className="text-faint">of</span>
          <InfinityIcon size={14} className="text-teal" aria-label="unlimited" />
        </span>
      </div>
    )
  }

  const share = cap > 0 ? Math.min(1, used / cap) : 1
  // Amber before it bites, not after. Somebody at 80% of their reports
  // wants to know now, not when the next one is refused.
  const tone =
    share >= 1 ? 'bg-rose' : share >= 0.8 ? 'bg-amber' : 'bg-accent'

  return (
    <div className="py-2.5">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-sm text-ink2">{label}</span>
        <span className="text-sm tabular-nums text-mute">
          <span className="text-ink">{fmt.count(used)}</span> of{' '}
          {cap === 0 ? 'none' : fmt.count(cap)}
        </span>
      </div>
      <div
        className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-panel2"
        role="meter"
        aria-valuenow={used}
        aria-valuemin={0}
        aria-valuemax={cap}
        aria-label={label}
      >
        <div
          className={`h-full rounded-full transition-all ${tone}`}
          style={{ width: `${Math.max(share * 100, used > 0 ? 3 : 0)}%` }}
        />
      </div>
    </div>
  )
}

export default function AccountPage() {
  const [account, setAccount] = useState<Allowance | null>(null)
  const [catalogue, setCatalogue] = useState<Catalogue | null>(null)
  const [error, setError] = useState('')
  const [moving, setMoving] = useState('')

  // Password change
  const [current, setCurrent] = useState('')
  const [next, setNext] = useState('')
  const [pwError, setPwError] = useState('')
  const [pwDone, setPwDone] = useState(false)
  const [saving, setSaving] = useState(false)

  const load = useCallback(() => {
    apiGet<Allowance>('/api/account')
      .then(setAccount)
      .catch((e: Error) => setError(e.message))
    apiGet<Catalogue>('/api/billing/plans')
      .then(setCatalogue)
      .catch(() => {
        /* The catalogue is the upgrade path, not the account itself —
           a deployment that cannot sell still shows what you are on. */
      })
  }, [])

  useEffect(load, [load])

  async function upgrade(planKey: string) {
    setMoving(planKey)
    setError('')
    try {
      const res = await apiPost<{ url: string }>('/api/billing/checkout', {
        plan: planKey,
      })
      window.location.href = res.url
    } catch (e) {
      setError((e as Error).message)
      setMoving('')
    }
  }

  async function changePassword() {
    setSaving(true)
    setPwError('')
    setPwDone(false)
    try {
      await apiPost('/api/account/password', {
        current_password: current,
        new_password: next,
      })
      setPwDone(true)
      setCurrent('')
      setNext('')
    } catch (e) {
      setPwError((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  if (error && !account) return <ErrorBox message={error} />
  if (!account) {
    return (
      <div className="flex items-center gap-2 py-16 text-sm text-mute">
        <Loader2 size={16} className="animate-spin" />
        Loading your account…
      </div>
    )
  }

  const current_plan = account.plan
  const canBuy = catalogue?.checkout_available ?? false

  return (
    <div className="p-8">
      <PageHeader
        title="Account"
        subtitle="What you are on, what you have used, and what it would take to lift a ceiling."
        right={
          <Badge tone={account.metered ? 'accent' : 'teal'}>
            {account.plan_label}
          </Badge>
        }
      />

      {error && (
        <div className="mb-5">
          <ErrorBox message={error} />
        </div>
      )}

      <div className="grid gap-5 lg:grid-cols-[1fr_1.3fr]">
        <div className="space-y-5">
          <Panel title="You" subtitle="Who this session belongs to.">
            <dl className="divide-y divide-edge text-sm">
              <div className="flex items-baseline justify-between gap-3 py-2.5">
                <dt className="text-mute">Username</dt>
                <dd className="font-medium text-ink">{account.username}</dd>
              </div>
              {account.email && (
                <div className="flex items-baseline justify-between gap-3 py-2.5">
                  <dt className="text-mute">Email</dt>
                  <dd className="text-ink2">{account.email}</dd>
                </div>
              )}
              <div className="flex items-baseline justify-between gap-3 py-2.5">
                <dt className="text-mute">Plan</dt>
                <dd className="text-ink">{account.plan_label}</dd>
              </div>
              {account.is_admin && (
                <div className="flex items-baseline justify-between gap-3 py-2.5">
                  <dt className="text-mute">Role</dt>
                  <dd>
                    <Badge tone="amber">Administrator</Badge>
                  </dd>
                </div>
              )}
            </dl>
          </Panel>

          <Panel
            title="Usage"
            subtitle={
              account.metered
                ? 'Counts reset at the start of each month.'
                : 'This deployment is self-hosted, so nothing is capped.'
            }
          >
            {account.metered ? (
              <div className="divide-y divide-edge">
                {METERS.map((m) => (
                  <Meter
                    key={m.key}
                    label={m.label}
                    used={Number(account.used[m.usedKey] ?? 0)}
                    cap={
                      account.limits[m.key] === undefined
                        ? null
                        : account.limits[m.key]
                    }
                  />
                ))}
                <div className="flex items-baseline justify-between gap-3 py-2.5">
                  <span className="text-sm text-ink2">Rows per dataset</span>
                  <span className="text-sm tabular-nums text-mute">
                    {capText(account.limits.rows_per_dataset)}
                  </span>
                </div>
                <div className="flex items-baseline justify-between gap-3 py-2.5">
                  <span className="text-sm text-ink2">Largest upload</span>
                  <span className="text-sm tabular-nums text-mute">
                    {account.limits.max_upload_mb === null
                      ? 'Unlimited'
                      : `${fmt.count(account.limits.max_upload_mb)} MB`}
                  </span>
                </div>
              </div>
            ) : (
              <p className="flex items-start gap-2 text-sm leading-relaxed text-mute">
                <CheckCircle2
                  size={15}
                  className="mt-0.5 shrink-0 text-teal"
                />
                Capping the container you pay for yourself would serve
                nobody. Every ceiling is off.
              </p>
            )}
          </Panel>

          <Panel
            title="Password"
            subtitle="Changing it does not sign out your other sessions."
          >
            <div className="space-y-3">
              <label className="block">
                <span className="mb-1 block text-xs font-medium text-mute">
                  Current password
                </span>
                <input
                  type="password"
                  value={current}
                  onChange={(e) => setCurrent(e.target.value)}
                  autoComplete="current-password"
                  className="w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink outline-none focus:border-accent"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-xs font-medium text-mute">
                  New password
                </span>
                <input
                  type="password"
                  value={next}
                  onChange={(e) => setNext(e.target.value)}
                  autoComplete="new-password"
                  className="w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm text-ink outline-none focus:border-accent"
                />
              </label>

              {pwError && <ErrorBox message={pwError} />}
              {pwDone && (
                <p className="flex items-center gap-1.5 text-xs text-teal">
                  <CheckCircle2 size={13} /> Password changed.
                </p>
              )}

              <Btn
                onClick={changePassword}
                disabled={saving || !current || !next}
                size="sm"
              >
                {saving ? (
                  <span className="flex items-center gap-1.5">
                    <Loader2 size={13} className="animate-spin" /> Saving…
                  </span>
                ) : (
                  <span className="flex items-center gap-1.5">
                    <KeyRound size={13} /> Change password
                  </span>
                )}
              </Btn>
            </div>
          </Panel>
        </div>

        <Panel
          title="Plans"
          subtitle={
            account.metered
              ? 'Every tier runs the same analysis. What changes is how much of it you can do.'
              : 'Shown for reference. Nothing is metered on a self-hosted deployment.'
          }
        >
          {!catalogue ? (
            <div className="flex items-center gap-2 py-6 text-sm text-mute">
              <Loader2 size={15} className="animate-spin" />
              Loading plans…
            </div>
          ) : (
            <>
              {!canBuy && account.metered && (
                <p className="mb-4 flex items-start gap-2 rounded-lg border border-amber/25 bg-amber/[0.06] px-3 py-2.5 text-xs leading-relaxed text-ink2">
                  <AlertTriangle
                    size={14}
                    className="mt-0.5 shrink-0 text-amber"
                  />
                  Payments are not set up on this deployment. Ask whoever runs
                  it to move your account.
                </p>
              )}

              <div className="space-y-3">
                {catalogue.plans.map((p) => {
                  const isCurrent = p.key === current_plan
                  return (
                    <div
                      key={p.key}
                      className={`rounded-xl border p-4 transition ${
                        isCurrent
                          ? 'border-accent/40 bg-accent/[0.05]'
                          : 'border-edge bg-panel2/30'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <h3 className="text-sm font-semibold text-ink">
                              {p.label}
                            </h3>
                            {isCurrent && (
                              <Badge tone="accent">Your plan</Badge>
                            )}
                          </div>
                          <p className="mt-1 text-xs leading-relaxed text-mute">
                            {p.notes}
                          </p>
                        </div>
                        <div className="shrink-0 text-right">
                          <div className="text-sm font-semibold tabular-nums text-ink">
                            {p.price_inr_month === 0
                              ? 'Free'
                              : `₹${fmt.count(p.price_inr_month)}`}
                          </div>
                          {p.price_inr_month > 0 && (
                            <div className="text-[11px] text-faint">
                              per month
                            </div>
                          )}
                        </div>
                      </div>

                      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5 text-[11px] sm:grid-cols-3">
                        {[
                          ['Datasets', capText(p.datasets)],
                          ['Rows each', capText(p.rows_per_dataset)],
                          [
                            'Upload',
                            p.max_upload_mb === null
                              ? 'Unlimited'
                              : `${fmt.count(p.max_upload_mb)} MB`,
                          ],
                          ['Reports / mo', capText(p.reports_per_month)],
                          ['Scheduled', capText(p.scheduled_reports)],
                          // branding=true means the deliverable carries
                          // OUR name — the one thing a freelancer pays
                          // to remove — so the paid tiers are the false
                          // ones. Reading it the intuitive way round
                          // advertises the opposite of what you get.
                          [
                            'Report footer',
                            p.branding ? 'Analytiq mark' : 'Your name only',
                          ],
                        ].map(([k, v]) => (
                          <div key={k}>
                            <dt className="text-faint">{k}</dt>
                            <dd className="tabular-nums text-ink2">{v}</dd>
                          </div>
                        ))}
                      </dl>

                      {!isCurrent &&
                        account.metered &&
                        p.price_inr_month > 0 && (
                          <div className="mt-3">
                            <Btn
                              size="sm"
                              variant={canBuy ? 'primary' : 'ghost'}
                              disabled={!canBuy || moving === p.key}
                              onClick={() => upgrade(p.key)}
                            >
                              {moving === p.key ? (
                                <span className="flex items-center gap-1.5">
                                  <Loader2
                                    size={13}
                                    className="animate-spin"
                                  />
                                  Opening checkout…
                                </span>
                              ) : (
                                <span className="flex items-center gap-1.5">
                                  <CreditCard size={13} /> Move to {p.label}
                                </span>
                              )}
                            </Btn>
                          </div>
                        )}
                    </div>
                  )
                })}
              </div>

              <p className="mt-4 flex items-start gap-2 text-[11px] leading-relaxed text-faint">
                <UserCircle2 size={13} className="mt-0.5 shrink-0" />
                No analysis is behind a paywall. Every plan runs the same
                statistics, the same models and the same reports — a tier
                caps how much you can do, never how correct it is.
              </p>
            </>
          )}
        </Panel>
      </div>
    </div>
  )
}
