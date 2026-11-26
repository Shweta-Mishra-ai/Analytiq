/**
 * Public pages: pricing, terms, privacy, refunds and contact. Reachable
 * without signing in — payment providers check them before switching an
 * account on, and visitors read them before signing up.
 *
 * The business name, address, phone and email come from the server's
 * settings (BUSINESS_NAME, BUSINESS_ADDRESS, BUSINESS_PHONE, SUPPORT_EMAIL).
 * The policy text is a starting draft: have it reviewed for your business.
 */
import { useEffect, useState, type ReactNode } from 'react'
import { Link, NavLink } from 'react-router-dom'
import { Database } from 'lucide-react'
import { apiGet } from '../api/client'

interface Business {
  name: string
  address: string
  phone: string
  email: string
}
interface Plans {
  packs: { id: string; name: string; credits: number; inr: number; usd: number }[]
  costs: { action: string; label: string; credits: number }[]
  free_daily: number
}

function useBusiness(): Business {
  const [b, setB] = useState<Business>({ name: '', address: '', phone: '', email: '' })
  useEffect(() => {
    apiGet<{ business?: Business }>('/api/health')
      .then((h) => h.business && setB(h.business))
      .catch(() => {})
  }, [])
  return b
}

function Missing({ what }: { what: string }) {
  return <span className="rounded bg-amber/15 px-1 text-amber">[{what} not set]</span>
}

const links = [
  ['/pricing', 'Pricing'],
  ['/terms', 'Terms'],
  ['/privacy', 'Privacy'],
  ['/refunds', 'Refunds'],
  ['/contact', 'Contact'],
] as const

export function PublicShell({ title, children }: { title: string; children: ReactNode }) {
  const b = useBusiness()
  return (
    <div className="min-h-full bg-bg">
      <header className="border-b border-edge bg-panel">
        <div className="mx-auto flex max-w-4xl flex-wrap items-center justify-between gap-3 px-4 py-4">
          <Link to="/" className="flex items-center gap-2 font-bold text-ink">
            <Database className="h-4 w-4 text-accent" /> Analytiq
          </Link>
          <nav className="flex flex-wrap gap-4 text-sm">
            {links.map(([to, label]) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) => (isActive ? 'font-semibold text-accent' : 'text-mute hover:text-ink')}
              >
                {label}
              </NavLink>
            ))}
            <Link to="/" className="font-semibold text-accent">
              Sign in
            </Link>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-4xl px-4 py-10">
        <h1 className="t-display mb-6 text-ink">{title}</h1>
        <div className="policy space-y-4 text-sm leading-relaxed text-ink2">{children}</div>
      </main>
      <footer className="border-t border-edge py-6 text-center text-xs text-faint">
        © {new Date().getFullYear()} {b.name || 'Analytiq'}
      </footer>
    </div>
  )
}

function H({ children }: { children: ReactNode }) {
  return <h2 className="pt-4 text-base font-semibold text-ink">{children}</h2>
}

function Who() {
  const b = useBusiness()
  return <>{b.name || <Missing what="BUSINESS_NAME" />}</>
}

export function PricingPage() {
  const [plans, setPlans] = useState<Plans | null>(null)
  useEffect(() => {
    apiGet<Plans>('/api/billing/plans').then(setPlans).catch(() => {})
  }, [])
  return (
    <PublicShell title="Pricing">
      <p>
        Start free: every account gets <b>{plans?.free_daily ?? 5} free credits every day</b>.
        Uploading data, dashboards, charts and insights are always free. Reports, model training
        and AI answers use credits. Buy more when you need them: one-time packs, no subscription,
        and bought credits never expire.
      </p>
      <div className="grid gap-4 pt-2 sm:grid-cols-3">
        {plans?.packs.map((p) => (
          <div key={p.id} className="surface p-5">
            <div className="font-semibold text-ink">{p.name}</div>
            <div className="t-metric mt-2 text-3xl text-ink">{p.credits}</div>
            <div className="text-xs text-mute">credits</div>
            <div className="mt-3 text-ink">
              ₹{(p.inr / 100).toLocaleString('en-IN')}{' '}
              <span className="text-faint">· ${(p.usd / 100).toFixed(2)}</span>
            </div>
          </div>
        ))}
      </div>
      <H>What uses credits</H>
      <ul className="list-disc pl-5">
        {plans?.costs.map((c) => (
          <li key={c.action}>
            {c.label}: {c.credits} credit{c.credits === 1 ? '' : 's'}
          </li>
        ))}
      </ul>
      <p>
        Pay by UPI, card, netbanking or wallet (Razorpay, in rupees), by card through Stripe, or
        with PayPal. Prices include applicable taxes unless the invoice says otherwise.
      </p>
    </PublicShell>
  )
}

export function TermsPage() {
  return (
    <PublicShell title="Terms of Service">
      <p>
        These terms are an agreement between you and <Who /> (&quot;we&quot;), which operates
        Analytiq. By creating an account you accept them.
      </p>
      <H>The service</H>
      <p>
        Analytiq analyses data files you upload and produces charts, statistics, predictions and
        reports. Results are computed from your data as supplied; they describe associations and
        do not establish causes. You are responsible for decisions you make with them.
      </p>
      <H>Your account</H>
      <p>
        Keep your password private; you are responsible for activity on your account. One person
        or organisation per account; creating accounts to collect extra free credits is not
        allowed and such accounts may be closed.
      </p>
      <H>Your data</H>
      <p>
        You keep all rights to the data you upload. You confirm you are allowed to upload it,
        including any personal data in it. We use it only to provide the service to you, as
        described in the Privacy Policy.
      </p>
      <H>Credits and payment</H>
      <p>
        Credits are bought in one-time packs at the prices on the Pricing page. Free daily
        credits have no cash value and do not carry over. Refunds follow the Refund Policy.
      </p>
      <H>Acceptable use</H>
      <p>
        Do not upload malware, attempt to access other users&apos; data, overload the service,
        or use it for anything unlawful. We may suspend accounts that do.
      </p>
      <H>Availability and liability</H>
      <p>
        We work to keep the service available and accurate but provide it &quot;as is&quot;. To
        the extent the law allows, our total liability for any claim is limited to the amount
        you paid us in the three months before it.
      </p>
      <H>Changes and law</H>
      <p>
        We may update these terms and will show the new date here. These terms are governed by
        the laws of India.
      </p>
    </PublicShell>
  )
}

export function PrivacyPage() {
  const b = useBusiness()
  return (
    <PublicShell title="Privacy Policy">
      <p>
        <Who /> operates Analytiq and is responsible for the personal data described here, in
        line with India&apos;s Digital Personal Data Protection Act, 2023.
      </p>
      <H>What we collect</H>
      <ul className="list-disc pl-5">
        <li>Account details: username, email address and a salted hash of your password.</li>
        <li>The files you upload and the results computed from them.</li>
        <li>Payment records: amount, pack and the payment provider&apos;s reference. Card,
          UPI and bank details are handled by the payment provider and never reach us.</li>
        <li>Technical logs (time, page, error references) used to keep the service working.</li>
      </ul>
      <H>How we use it</H>
      <p>
        Only to run the service for you, process payments, prevent abuse and answer support
        requests. We do not sell your data or use it for advertising, and we do not use your
        files to train models.
      </p>
      <H>Who else handles it</H>
      <p>
        Payment providers (Razorpay, Stripe, PayPal) for payments, and our hosting provider. If
        AI features are switched on, the parts of your data needed to answer a question may be
        sent to the AI provider used for that answer.
      </p>
      <H>How long we keep it</H>
      <p>
        Uploaded files are deleted automatically after the retention period shown on the upload
        page, or when you delete them. Deleting your account deletes all of your files. Payment
        records are kept as long as tax law requires.
      </p>
      <H>Your rights</H>
      <p>
        You can access, correct or delete your data, and withdraw consent, by writing to{' '}
        {b.email || <Missing what="SUPPORT_EMAIL" />}. We reply within 30 days.
      </p>
    </PublicShell>
  )
}

export function RefundsPage() {
  const b = useBusiness()
  return (
    <PublicShell title="Refund & Cancellation Policy">
      <p>
        Credit packs are one-time purchases; there is no subscription to cancel.
      </p>
      <ul className="list-disc pl-5">
        <li>
          <b>Failed actions:</b> if a report, training run or answer fails, its credits are
          returned to your balance automatically.
        </li>
        <li>
          <b>Unused credits:</b> you can ask for a refund of a pack within 7 days of buying it if
          none of its credits have been used.
        </li>
        <li>
          <b>Duplicate or wrong charges:</b> refunded in full once confirmed.
        </li>
        <li>
          Used credits and free daily credits are not refundable.
        </li>
      </ul>
      <p>
        To ask for a refund, write to {b.email || <Missing what="SUPPORT_EMAIL" />} with your
        username and the payment reference. Approved refunds go back to the original payment
        method within 5–7 business days, depending on your bank or provider.
      </p>
    </PublicShell>
  )
}

export function ContactPage() {
  const b = useBusiness()
  return (
    <PublicShell title="Contact">
      <dl className="grid gap-3 sm:grid-cols-[10rem_1fr]">
        <dt className="text-mute">Business</dt>
        <dd>{b.name || <Missing what="BUSINESS_NAME" />}</dd>
        <dt className="text-mute">Email</dt>
        <dd>{b.email || <Missing what="SUPPORT_EMAIL" />}</dd>
        <dt className="text-mute">Phone</dt>
        <dd>{b.phone || <Missing what="BUSINESS_PHONE" />}</dd>
        <dt className="text-mute">Address</dt>
        <dd className="whitespace-pre-line">{b.address || <Missing what="BUSINESS_ADDRESS" />}</dd>
      </dl>
      <p className="pt-2">We answer within two business days.</p>
    </PublicShell>
  )
}
