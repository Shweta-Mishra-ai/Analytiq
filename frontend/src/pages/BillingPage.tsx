import { useCallback, useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { CheckCircle2, Coins, CreditCard, Infinity as InfinityIcon, Smartphone } from 'lucide-react'
import { apiBlob, apiGet, apiPost, downloadBlob } from '../api/client'
import { Badge, Btn, ErrorBox, PageHeader, Panel, Skeleton, Stat } from '../components/Ui'

interface Pack {
  id: string
  name: string
  credits: number
  inr: number
  usd: number
}
interface Balance {
  unlimited: boolean
  free_left: number
  free_daily: number
  paid: number
  total: number
  needs_verification?: boolean
}
interface Entry {
  t: number
  change: number
  kind: string
  note: string
}
interface Overview {
  balance: Balance
  history: Entry[]
  recovered: number
  packs: Pack[]
  costs: { action: string; label: string; credits: number }[]
  free_daily: number
  providers: { razorpay: boolean; stripe: boolean; paypal: boolean; upi?: boolean }
  stripe_currency: string
  invoices: {
    number: string
    issued_at: number
    amount: number
    currency: string
    credits: number
    tax_invoice: boolean
  }[]
  upi: { id: string; payee: string; links: Record<string, string> } | null
  upi_requests: { id: string; pack: string; credits: number; utr: string; status: string; created: number; note: string }[]
}

const money = (amount: number, currency: string) =>
  currency === 'INR' ? inr(amount) : currency === 'USD' ? usd(amount) : `${currency} ${amount / 100}`

/** Paying the business's UPI ID straight from a UPI app, then giving the
 *  reference so an admin can match it in the bank account. */
function UpiDirect({ data, onDone }: { data: Overview; onDone: () => void }) {
  const [packId, setPackId] = useState(data.packs[0]?.id ?? '')
  const [utr, setUtr] = useState('')
  const [msg, setMsg] = useState('')
  const [err, setErr] = useState('')
  const upi = data.upi!
  const pack = data.packs.find((p) => p.id === packId) ?? data.packs[0]
  const submit = async () => {
    setErr('')
    setMsg('')
    try {
      await apiPost('/api/billing/upi', { pack: packId, utr })
      setMsg('Thank you. We will check the payment and add your credits, usually within a working day.')
      setUtr('')
      onDone()
    } catch (e) {
      setErr((e as Error).message)
    }
  }
  return (
    <Panel
      title="Pay directly by UPI"
      subtitle="Pay from any UPI app, then enter the 12-digit UPI reference (UTR). Credits are added once the payment is checked."
    >
      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2 text-sm">
          <label className="block text-xs text-mute">Pack</label>
          <select
            value={packId}
            onChange={(e) => setPackId(e.target.value)}
            className="w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-ink"
          >
            {data.packs.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}: {p.credits} credits for {inr(p.inr)}
              </option>
            ))}
          </select>
          <div className="rounded-lg border border-edge bg-panel2 p-3">
            <div className="text-xs text-mute">Pay exactly {pack ? inr(pack.inr) : ''} to</div>
            <div className="font-data text-base text-ink" data-testid="upi-id">{upi.id}</div>
            <div className="text-xs text-faint">{upi.payee}</div>
            <a
              href={upi.links[packId]}
              className="btn-primary mt-3 inline-block rounded-lg px-3 py-1.5 text-xs font-semibold text-white md:hidden"
            >
              Open UPI app
            </a>
          </div>
        </div>
        <div className="space-y-2 text-sm">
          <label className="block text-xs text-mute">UPI reference (UTR), after paying</label>
          <input
            value={utr}
            onChange={(e) => setUtr(e.target.value)}
            inputMode="numeric"
            placeholder="12 digits, e.g. 412345678901"
            className="w-full rounded-lg border border-edge bg-panel2 px-3 py-2 text-ink focus:border-accent focus:outline-none"
          />
          <Btn onClick={submit} disabled={utr.replace(/\s/g, '').length !== 12} size="sm">
            I have paid — check my payment
          </Btn>
          {msg && <p className="text-teal">{msg}</p>}
          {err && <p className="text-rose">{err}</p>}
          {data.upi_requests.length > 0 && (
            <ul className="mt-2 space-y-1 text-xs">
              {data.upi_requests.map((r) => (
                <li key={r.id} className="flex justify-between gap-2">
                  <span className="text-ink2">
                    {r.credits} credits · UTR …{r.utr.slice(-4)}
                    {r.status === 'rejected' && r.note && (
                      <span className="block text-rose">{r.note}</span>
                    )}
                  </span>
                  <Badge tone={r.status === 'approved' ? 'teal' : r.status === 'rejected' ? 'rose' : 'amber'}>
                    {r.status === 'pending' ? 'being checked' : r.status}
                  </Badge>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </Panel>
  )
}

type RazorpayResponse = {
  razorpay_order_id: string
  razorpay_payment_id: string
  razorpay_signature: string
}
declare global {
  interface Window {
    Razorpay?: new (opts: Record<string, unknown>) => { open: () => void }
  }
}

const inr = (paise: number) =>
  '₹' + (paise / 100).toLocaleString('en-IN', { maximumFractionDigits: 2 })
const usd = (cents: number) => '$' + (cents / 100).toFixed(2).replace(/\.00$/, '')

function loadRazorpay(): Promise<void> {
  if (window.Razorpay) return Promise.resolve()
  return new Promise((resolve, reject) => {
    const s = document.createElement('script')
    s.src = 'https://checkout.razorpay.com/v1/checkout.js'
    s.onload = () => resolve()
    s.onerror = () => reject(new Error('The Razorpay payment window could not be loaded.'))
    document.body.appendChild(s)
  })
}

export default function BillingPage() {
  const [data, setData] = useState<Overview | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState('')
  const [params, setParams] = useSearchParams()
  const handledReturn = useRef(false)

  const load = useCallback(async () => {
    try {
      const d = await apiGet<Overview>('/api/billing')
      setData(d)
      if (d.recovered > 0)
        setNotice(`${d.recovered} credits from an earlier payment have been added.`)
    } catch (e) {
      setError((e as Error).message)
    }
  }, [])

  // Back from Stripe or PayPal: confirm the payment with the server once.
  useEffect(() => {
    if (handledReturn.current) return
    handledReturn.current = true
    const provider = params.get('provider')
    const confirm = async () => {
      try {
        if (provider === 'stripe' && params.get('session_id')) {
          const r = await apiPost<{ credited: number }>('/api/billing/stripe/confirm', {
            session_id: params.get('session_id'),
          })
          setNotice(r.credited ? `Payment received: ${r.credited} credits added.` : 'This payment was already added.')
        } else if (provider === 'paypal' && params.get('token')) {
          const r = await apiPost<{ credited: number }>('/api/billing/paypal/confirm', {
            order_id: params.get('token'),
          })
          setNotice(r.credited ? `Payment received: ${r.credited} credits added.` : 'This payment was already added.')
        } else if (params.get('cancelled')) {
          setNotice('Payment cancelled. Nothing was charged.')
        }
      } catch (e) {
        setError((e as Error).message)
      }
      if (provider || params.get('cancelled')) setParams({}, { replace: true })
      load()
    }
    confirm()
  }, [params, setParams, load])

  const buy = async (provider: 'razorpay' | 'stripe' | 'paypal', pack: Pack) => {
    setError('')
    setNotice('')
    setBusy(`${provider}:${pack.id}`)
    try {
      const r = await apiPost<Record<string, unknown>>('/api/billing/checkout', {
        provider,
        pack: pack.id,
      })
      if (provider === 'razorpay') {
        await loadRazorpay()
        const rzp = new window.Razorpay!({
          key: r.key_id,
          order_id: r.order_id,
          amount: r.amount,
          currency: r.currency,
          name: r.name,
          description: r.description,
          prefill: r.prefill,
          theme: { color: '#2563eb' },
          handler: async (resp: RazorpayResponse) => {
            try {
              const v = await apiPost<{ credited: number }>('/api/billing/razorpay/verify', {
                order_id: resp.razorpay_order_id,
                payment_id: resp.razorpay_payment_id,
                signature: resp.razorpay_signature,
              })
              setNotice(`Payment received: ${v.credited || pack.credits} credits added.`)
            } catch (e) {
              setError(
                `${(e as Error).message} If money left your account, it will be added ` +
                  'automatically the next time you open this page.',
              )
            }
            load()
          },
          modal: { ondismiss: () => setBusy('') },
        })
        rzp.open()
      } else {
        window.location.href = String(r.url)
      }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      if (provider === 'razorpay') setBusy('')
    }
  }

  if (!data)
    return (
      <div className="p-6 lg:p-8">
        <PageHeader title="Billing" subtitle="Your credits and buying more" />
        {error ? <ErrorBox message={error} /> : <Skeleton rows={6} />}
      </div>
    )

  const b = data.balance
  const anyProvider =
    data.providers.razorpay || data.providers.stripe || data.providers.paypal || data.providers.upi

  return (
    <div className="mx-auto max-w-5xl p-6 lg:p-8">
      <PageHeader
        title="Billing"
        subtitle="Looking at your data is always free. Reports, model training and AI answers use credits."
      />
      {notice && (
        <div className="mb-4 flex items-center gap-2 rounded-xl border border-teal/30 bg-teal/8 px-4 py-3 text-sm text-teal">
          <CheckCircle2 className="h-4 w-4 shrink-0" /> {notice}
        </div>
      )}
      {error && (
        <div className="mb-4">
          <ErrorBox message={error} />
        </div>
      )}

      {b.unlimited ? (
        <div className="surface mb-6 flex items-center gap-3 p-5">
          <InfinityIcon className="h-6 w-6 text-accent" />
          <div>
            <div className="font-semibold text-ink">Unlimited account</div>
            <div className="text-sm text-mute">Nothing you do here uses credits.</div>
          </div>
        </div>
      ) : (
        <div className="mb-6 grid gap-3 sm:grid-cols-3">
          <Stat label="Credits available" value={b.total} tone="accent" />
          <Stat
            label="Free today"
            value={`${b.free_left} of ${b.free_daily}`}
            hint={
              b.needs_verification
                ? 'Confirm your email to switch these on'
                : 'Renews every midnight (India time)'
            }
          />
          <Stat label="Bought credits" value={b.paid} hint="Never expire" />
        </div>
      )}

      <Panel title="Buy credits" subtitle="One-time payment. No subscription, nothing renews.">
        {!anyProvider && (
          <p className="text-sm text-mute">
            Online payment is not switched on yet. Contact support to buy credits.
          </p>
        )}
        <div className="grid gap-4 md:grid-cols-3">
          {data.packs.map((p, i) => (
            <div
              key={p.id}
              className={`flex flex-col rounded-xl border p-5 ${
                i === 1 ? 'border-accent/50 bg-accent/5' : 'border-edge bg-panel2'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-semibold text-ink">{p.name}</span>
                {i === 1 && <Badge tone="accent">Popular</Badge>}
              </div>
              <div className="mt-3 flex items-baseline gap-1.5">
                <Coins className="h-4 w-4 text-amber" />
                <span className="t-metric text-3xl text-ink">{p.credits}</span>
                <span className="text-sm text-mute">credits</span>
              </div>
              <div className="mt-1 text-sm text-ink2">
                {inr(p.inr)} <span className="text-faint">· {usd(p.usd)}</span>
              </div>
              <div className="text-xs text-faint">
                {inr(Math.round(p.inr / p.credits))} per credit
              </div>
              <div className="mt-4 flex flex-col gap-2">
                {data.providers.razorpay && (
                  <Btn onClick={() => buy('razorpay', p)} disabled={!!busy} size="sm">
                    <span className="inline-flex items-center gap-1.5">
                      <Smartphone className="h-3.5 w-3.5" /> UPI / Card / Netbanking · {inr(p.inr)}
                    </span>
                  </Btn>
                )}
                {data.providers.stripe && (
                  <Btn variant="subtle" onClick={() => buy('stripe', p)} disabled={!!busy} size="sm">
                    <span className="inline-flex items-center gap-1.5">
                      <CreditCard className="h-3.5 w-3.5" /> Card (Stripe) ·{' '}
                      {data.stripe_currency === 'INR' ? inr(p.inr) : usd(p.usd)}
                    </span>
                  </Btn>
                )}
                {data.providers.paypal && (
                  <Btn variant="subtle" onClick={() => buy('paypal', p)} disabled={!!busy} size="sm">
                    PayPal · {usd(p.usd)}
                  </Btn>
                )}
              </div>
            </div>
          ))}
        </div>
      </Panel>

      {data.upi && (
        <div className="mt-6">
          <UpiDirect data={data} onDone={load} />
        </div>
      )}

      {data.invoices.length > 0 && (
        <div className="mt-6">
          <Panel title="Invoices and receipts">
            <ul className="space-y-1.5 text-sm">
              {data.invoices.map((i) => (
                <li key={i.number} className="flex items-center justify-between gap-3">
                  <span className="text-ink2">
                    <span className="font-data">{i.number}</span>
                    <span className="ml-2 text-xs text-faint">
                      {new Date(i.issued_at * 1000).toLocaleDateString()} · {i.credits} credits ·{' '}
                      {money(i.amount, i.currency)}
                    </span>
                  </span>
                  <Btn
                    variant="ghost"
                    size="sm"
                    onClick={async () => {
                      try {
                        downloadBlob(await apiBlob(`/api/billing/invoices/${i.number}`), `${i.number}.pdf`)
                      } catch (e) {
                        setError((e as Error).message)
                      }
                    }}
                  >
                    {i.tax_invoice ? 'Tax invoice' : 'Receipt'} PDF
                  </Btn>
                </li>
              ))}
            </ul>
          </Panel>
        </div>
      )}

      <div className="mt-6 grid gap-6 md:grid-cols-2">
        <Panel title="What uses credits">
          <table className="w-full text-sm">
            <tbody>
              {data.costs.map((c) => (
                <tr key={c.action} className="border-b border-edge last:border-0">
                  <td className="py-2 text-ink2">{c.label}</td>
                  <td className="py-2 text-right font-data text-ink">{c.credits}</td>
                </tr>
              ))}
              <tr>
                <td className="py-2 text-mute">Upload, dashboards, charts, insights</td>
                <td className="py-2 text-right text-teal">Free</td>
              </tr>
            </tbody>
          </table>
          <p className="mt-3 text-xs text-faint">
            If an action fails, its credits are returned automatically.
          </p>
        </Panel>
        <Panel title="Recent activity">
          {data.history.length === 0 ? (
            <p className="text-sm text-mute">Nothing yet.</p>
          ) : (
            <ul className="max-h-72 space-y-1.5 overflow-y-auto text-sm">
              {data.history.map((e, i) => (
                <li key={i} className="flex justify-between gap-3">
                  <span className="min-w-0 truncate text-ink2">
                    {e.note}
                    <span className="ml-2 text-xs text-faint">
                      {new Date(e.t * 1000).toLocaleString()}
                    </span>
                  </span>
                  <span className={`font-data ${e.change < 0 ? 'text-rose' : 'text-teal'}`}>
                    {e.change > 0 ? '+' : ''}
                    {e.change || ''}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
    </div>
  )
}
