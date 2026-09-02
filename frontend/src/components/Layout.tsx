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
  Search,
  FileLock2,
  ServerCog,
} from 'lucide-react'
import { getToken, setToken } from '../api/client'
import CommandPalette from './CommandPalette'
import DatasetSwitcher from './DatasetSwitcher'

// Grouped so the sidebar stays readable as the analysis surface grows —
// a flat list of 15 links makes it hard to find anything.
const navGroups = [
  {
    label: 'Data',
    items: [
      { to: '/', label: 'Upload', icon: Upload },
      { to: '/quality', label: 'Data Quality', icon: ShieldCheck },
      { to: '/governance', label: 'Governance', icon: FileLock2 },
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
      { to: '/system', label: 'System', icon: ServerCog },
    ],
  },
]

/** Every destination, flat, for the palette to search. */
const allPages = navGroups.flatMap((g) =>
  g.items.map((i) => ({ to: i.to, label: i.label, group: g.label })),
)

export default function Layout() {
  const { pathname } = useLocation()
  const current = allPages.find((p) =>
    p.to === '/' ? pathname === '/' : pathname.startsWith(p.to),
  )

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
      <div className="flex min-w-0 flex-1 flex-col">
        {/* The one piece of state every page depends on, kept in view.
            It used to be four words of grey text at the foot of the
            sidebar, below the fold on a short window, with no way to
            switch without navigating back to Upload. */}
        <header className="flex h-14 shrink-0 items-center gap-4 border-b border-edge bg-panel/60 px-6 backdrop-blur">
          <nav aria-label="Breadcrumb" className="flex items-center gap-1.5">
            <span className="text-[13px] text-faint">
              {current?.group ?? 'Analytiq'}
            </span>
            <span className="text-faint">/</span>
            <span className="text-[13px] font-medium text-ink">
              {current?.label ?? 'Home'}
            </span>
          </nav>

          <div className="ml-auto flex items-center gap-3">
            <button
              onClick={() =>
                window.dispatchEvent(
                  new KeyboardEvent('keydown', { key: 'k', metaKey: true }),
                )
              }
              className="hidden items-center gap-2 rounded-lg border border-edge bg-panel2 px-2.5 py-1.5 text-[12px] text-mute transition hover:border-edge2 hover:text-ink2 sm:flex"
            >
              <Search className="h-3.5 w-3.5" />
              Search
              <kbd className="rounded border border-edge px-1 font-data text-[10px] text-faint">
                ⌘K
              </kbd>
            </button>
            <DatasetSwitcher />
          </div>
        </header>

        <main className="min-w-0 flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
      <CommandPalette pages={allPages} />
    </div>
  )
}
