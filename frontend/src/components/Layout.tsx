import { NavLink, Outlet, useLocation } from 'react-router-dom'
import {
  Upload,
  ShieldCheck,
  LayoutDashboard,
  FlaskConical,
  FlaskRound,
  Brain,
  Lightbulb,
  Briefcase,
  FileText,
  MessageSquare,
  Database,
  Layers,
  LogOut,
  Users,
  Activity,
  GitCompare,
  Gauge,
} from 'lucide-react'
import { useApp } from '../store/app'
import { getToken, setToken } from '../api/client'
import ErrorBoundary from './ErrorBoundary'

// Grouped so the sidebar stays readable as the analysis surface grows —
// a flat list of 15 links makes it hard to find anything.
const navGroups = [
  {
    label: 'Data',
    items: [
      { to: '/', label: 'Upload', icon: Upload },
      { to: '/quality', label: 'Data Quality', icon: ShieldCheck },
    ],
  },
  {
    label: 'Explore',
    items: [
      { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
      { to: '/eda', label: 'Deep EDA', icon: FlaskConical },
      { to: '/insights', label: 'Insights', icon: Lightbulb },
      { to: '/bi', label: 'Business Intel', icon: Briefcase },
    ],
  },
  {
    label: 'Advanced',
    items: [
      { to: '/deep-analysis', label: 'Deep Analysis', icon: Gauge },
      { to: '/segments', label: 'Customer Segments', icon: Users },
      { to: '/ab-test', label: 'A/B Test', icon: FlaskRound },
      { to: '/survival', label: 'Survival', icon: Activity },
      { to: '/compare', label: 'Compare', icon: GitCompare },
      { to: '/ml', label: 'ML Predictions', icon: Brain },
    ],
  },
  {
    label: 'Deliver',
    items: [
      { to: '/chat', label: 'AI Chat', icon: MessageSquare },
      { to: '/rag', label: 'RAG Studio', icon: Layers },
      { to: '/reports', label: 'Reports', icon: FileText },
    ],
  },
]

export default function Layout() {
  const dataset = useApp((s) => s.dataset)
  const { pathname } = useLocation()
  return (
    <div className="flex h-full">
      <aside className="flex w-60 shrink-0 flex-col border-r border-edge bg-panel">
        <div className="flex items-center gap-2.5 px-5 py-5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-accent/25 bg-accent/12">
            <Database className="h-4 w-4 text-accent" />
          </div>
          <div className="min-w-0">
            <div className="text-sm font-bold tracking-tight text-ink">
              Analytiq
            </div>
            <div className="text-[10px] tracking-wide text-faint">
              Data Analysis Workbench
            </div>
          </div>
        </div>
        <nav className="flex-1 overflow-y-auto px-3 pb-2">
          {navGroups.map((group) => (
            <div key={group.label} className="mb-4">
              <div className="t-label px-2.5 pb-1.5">{group.label}</div>
              <div className="space-y-0.5">
                {group.items.map(({ to, label, icon: Icon }) => (
                  <NavLink
                    key={to}
                    to={to}
                    end={to === '/'}
                    className={({ isActive }) =>
                      `group relative flex items-center gap-2.5 rounded-lg px-2.5 py-[7px] text-[13px] transition-all duration-150 ${
                        isActive
                          ? 'bg-accent/12 font-semibold text-accent'
                          : 'text-mute hover:bg-panel2 hover:text-ink2'
                      }`
                    }
                  >
                    {({ isActive }) => (
                      <>
                        {/* active rail — a 2px marker reads as "you are
                            here" far faster than colour alone */}
                        <span
                          className={`absolute top-1/2 left-0 h-4 w-[2px] -translate-y-1/2 rounded-r-full bg-accent transition-opacity ${
                            isActive ? 'opacity-100' : 'opacity-0'
                          }`}
                        />
                        <Icon className="h-4 w-4 shrink-0" />
                        {label}
                      </>
                    )}
                  </NavLink>
                ))}
              </div>
            </div>
          ))}
        </nav>

        <div className="border-t border-edge px-4 py-3">
          {dataset ? (
            <>
              <div className="flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-teal" />
                <span className="truncate text-[12px] font-medium text-ink2">
                  {dataset.filename}
                </span>
              </div>
              <div className="mt-0.5 pl-3 font-data text-[11px] text-faint">
                {dataset.rows.toLocaleString()} × {dataset.cols}
              </div>
            </>
          ) : (
            <div className="flex items-center gap-1.5 text-[11px] text-faint">
              <span className="h-1.5 w-1.5 rounded-full bg-faint" />
              No dataset loaded
            </div>
          )}
        </div>

        {getToken() && (
          <button
            onClick={() => {
              setToken('')
              window.location.reload()
            }}
            className="flex items-center gap-2 border-t border-edge px-4 py-3 text-[11px] text-faint transition-colors hover:bg-panel2 hover:text-ink"
          >
            <LogOut className="h-3.5 w-3.5" /> Sign out
          </button>
        )}
      </aside>
      <main className="min-w-0 flex-1 overflow-y-auto">
        {/* Keyed on the route so navigating away clears a failed page
            rather than leaving the error panel in place for the next
            one. */}
        <ErrorBoundary key={pathname} label="This page">
          <Outlet />
        </ErrorBoundary>
      </main>
    </div>
  )
}
