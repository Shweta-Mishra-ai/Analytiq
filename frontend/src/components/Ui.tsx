import type { ReactNode } from 'react'
import { AlertTriangle, Loader2 } from 'lucide-react'
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
    <div className="mb-5 flex items-end justify-between gap-4">
      <div>
        <h1 className="text-xl font-bold text-ink">{title}</h1>
        {subtitle && <p className="mt-0.5 text-sm text-mute">{subtitle}</p>}
      </div>
      {right}
    </div>
  )
}

export function Panel({
  title,
  children,
  className = '',
}: {
  title?: string
  children: ReactNode
  className?: string
}) {
  return (
    <div className={`rounded-xl border border-edge bg-panel p-4 ${className}`}>
      {title && (
        <h2 className="mb-3 text-sm font-semibold text-ink">{title}</h2>
      )}
      {children}
    </div>
  )
}

export function Spinner({ label = 'Working…' }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 py-8 text-sm text-mute">
      <Loader2 className="h-4 w-4 animate-spin text-accent" /> {label}
    </div>
  )
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2 rounded-lg border border-rose/40 bg-rose/10 px-4 py-3 text-sm text-rose">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <span>{message}</span>
    </div>
  )
}

export function NeedData() {
  return (
    <div className="p-8">
      <div className="rounded-xl border border-edge bg-panel p-10 text-center">
        <p className="text-mute">No dataset loaded yet.</p>
        <Link
          to="/"
          className="mt-3 inline-block rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-white hover:opacity-90"
        >
          Upload a dataset →
        </Link>
      </div>
    </div>
  )
}

export function Btn({
  children,
  onClick,
  disabled,
  variant = 'primary',
  className = '',
}: {
  children: ReactNode
  onClick?: () => void
  disabled?: boolean
  variant?: 'primary' | 'ghost' | 'danger'
  className?: string
}) {
  const styles = {
    primary: 'bg-accent text-white hover:opacity-90',
    ghost: 'border border-edge text-mute hover:text-ink hover:bg-panel2',
    danger: 'bg-rose/80 text-white hover:bg-rose',
  }
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`rounded-lg px-4 py-2 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-40 ${styles[variant]} ${className}`}
    >
      {children}
    </button>
  )
}
