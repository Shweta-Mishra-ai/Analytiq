import type { ReactNode } from 'react'
import { AlertTriangle, Loader2, Upload } from 'lucide-react'
import { Link } from 'react-router-dom'

export function PageHeader({
  title,
  subtitle,
  right,
}: {
  title: string
  subtitle?: string
  right?: ReactNode
}) {
  return (
    <div className="mb-6 flex items-end justify-between gap-4 border-b border-edge pb-5">
      <div className="min-w-0">
        <h1 className="t-display text-ink">{title}</h1>
        {subtitle && (
          <p className="mt-1 text-sm leading-relaxed text-mute">{subtitle}</p>
        )}
      </div>
      {right && <div className="shrink-0">{right}</div>}
    </div>
  )
}

export function Panel({
  title,
  subtitle,
  right,
  children,
  className = '',
  interactive = false,
}: {
  title?: string
  subtitle?: string
  right?: ReactNode
  children: ReactNode
  className?: string
  interactive?: boolean
}) {
  return (
    <div className={`surface ${interactive ? 'surface-hover' : ''} p-5 ${className}`}>
      {(title || right) && (
        <div className="mb-4 flex items-start justify-between gap-3">
          <div className="min-w-0">
            {title && (
              <h2 className="text-sm font-semibold tracking-tight text-ink">
                {title}
              </h2>
            )}
            {subtitle && <p className="mt-0.5 text-xs text-mute">{subtitle}</p>}
          </div>
          {right && <div className="shrink-0">{right}</div>}
        </div>
      )}
      {children}
    </div>
  )
}

/** A single headline number. Keeps metrics visually consistent app-wide
 *  instead of every page inventing its own stat block. */
export function Stat({
  label,
  value,
  tone = 'ink',
  hint,
}: {
  label: string
  value: string | number
  tone?: 'ink' | 'accent' | 'teal' | 'amber' | 'rose' | 'violet'
  hint?: string
}) {
  const tones: Record<string, string> = {
    ink: 'text-ink',
    accent: 'text-accent',
    teal: 'text-teal',
    amber: 'text-amber',
    rose: 'text-rose',
    violet: 'text-violet',
  }
  return (
    <div className="surface p-4">
      <div className="t-label">{label}</div>
      <div className={`t-metric mt-1.5 text-2xl ${tones[tone]}`}>{value}</div>
      {hint && <div className="mt-1 text-xs text-faint">{hint}</div>}
    </div>
  )
}

export function Spinner({ label = 'Working…' }: { label?: string }) {
  return (
    <div className="flex items-center gap-2.5 py-8 text-sm text-mute">
      <Loader2 className="h-4 w-4 animate-spin text-accent" />
      {label}
    </div>
  )
}

/** Shaped placeholder for content that is loading. Reads as "on its way";
 *  a lone spinner on an empty page reads as "broken". */
export function Skeleton({
  rows = 3,
  className = '',
}: {
  rows?: number
  className?: string
}) {
  return (
    <div className={`space-y-2.5 ${className}`}>
      {Array.from({ length: rows }).map((_, i) => (
        <div
          key={i}
          className="skeleton h-3.5"
          style={{ width: `${100 - i * 12}%` }}
        />
      ))}
    </div>
  )
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2.5 rounded-xl border border-rose/30 bg-rose/8 px-4 py-3 text-sm text-rose">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <span className="leading-relaxed">{message}</span>
    </div>
  )
}

/** Shown when a page needs a dataset and none is loaded. An empty state
 *  should explain the situation and offer the next action, not just
 *  announce that something is missing. */
export function NeedData() {
  return (
    <div className="flex min-h-full items-center justify-center p-8">
      <div className="surface max-w-md p-10 text-center">
        <div className="mx-auto mb-4 flex h-11 w-11 items-center justify-center rounded-xl border border-edge bg-panel2">
          <Upload className="h-5 w-5 text-accent" />
        </div>
        <h2 className="text-base font-semibold text-ink">No dataset loaded</h2>
        <p className="mt-1.5 text-sm leading-relaxed text-mute">
          Upload a CSV or Excel file to start analysing. Everything on this
          page works from the dataset you load.
        </p>
        <Link
          to="/"
          className="btn-primary mt-5 inline-block rounded-lg px-4 py-2 text-sm font-semibold text-white"
        >
          Upload a dataset
        </Link>
      </div>
    </div>
  )
}

/**
 * Empty state for a section that ran fine and found nothing.
 *
 * A table of thirteen rows all reading "100" and "—" is a page that ran
 * correctly and said nothing, and it reads as a broken feature rather
 * than as a clean bill of health. `tone="good"` is for the second case:
 * the emptiness IS the result.
 */
export function EmptyState({
  title,
  hint,
  icon,
  tone = 'neutral',
  action,
}: {
  title: string
  hint?: string
  icon?: ReactNode
  tone?: 'neutral' | 'good'
  action?: ReactNode
}) {
  return (
    <div
      className={`flex flex-col items-center justify-center rounded-xl border border-dashed px-6 py-10 text-center ${
        tone === 'good'
          ? 'border-teal/25 bg-teal/[0.04]'
          : 'border-edge bg-panel2/30'
      }`}
    >
      {icon && (
        <div className={`mb-3 ${tone === 'good' ? 'text-teal' : 'text-faint'}`}>
          {icon}
        </div>
      )}
      <p className="text-sm font-medium text-ink2">{title}</p>
      {hint && (
        <p className="mt-1 max-w-md text-xs leading-relaxed text-mute">{hint}</p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

export function Btn({
  children,
  onClick,
  disabled,
  variant = 'primary',
  size = 'md',
  className = '',
}: {
  children: ReactNode
  onClick?: () => void
  disabled?: boolean
  variant?: 'primary' | 'ghost' | 'danger' | 'subtle'
  size?: 'sm' | 'md'
  className?: string
}) {
  const styles = {
    primary: 'btn-primary text-white',
    subtle: 'bg-panel2 text-ink border border-edge hover:border-edge2 hover:bg-edge/40',
    ghost: 'border border-edge text-mute hover:text-ink hover:bg-panel2',
    danger: 'bg-rose/85 text-white hover:bg-rose',
  }
  const sizes = {
    sm: 'px-3 py-1.5 text-xs',
    md: 'px-4 py-2 text-sm',
  }
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`rounded-lg font-semibold transition disabled:cursor-not-allowed disabled:opacity-40 ${styles[variant]} ${sizes[size]} ${className}`}
    >
      {children}
    </button>
  )
}

/** Small status pill — severity, verdicts, counts. */
export function Badge({
  children,
  tone = 'neutral',
}: {
  children: ReactNode
  tone?: 'neutral' | 'accent' | 'teal' | 'amber' | 'rose'
}) {
  const tones = {
    neutral: 'border-edge2 bg-panel2 text-mute',
    accent: 'border-accent/30 bg-accent/10 text-accent',
    teal: 'border-teal/30 bg-teal/10 text-teal',
    amber: 'border-amber/30 bg-amber/10 text-amber',
    rose: 'border-rose/30 bg-rose/10 text-rose',
  }
  return (
    <span
      className={`inline-flex items-center rounded-md border px-2 py-0.5 text-[11px] font-semibold ${tones[tone]}`}
    >
      {children}
    </span>
  )
}
