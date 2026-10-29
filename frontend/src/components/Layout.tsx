import { useEffect, useState } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'
import {
  Menu,
  X,
  Upload,
  ShieldCheck,
  Columns3,
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
import { ApiError, apiGet, getToken, setToken } from '../api/client'

// Grouped by the order work is done in: load and check the data, explore
// it, analyse and model it, then deliver. "Advanced" mixed statistical
// tests, segmentation and machine learning under one vague label.
const navGroups = [
  {
    label: 'Data',
    items: [
      { to: '/', label: 'Upload', icon: Upload },
      { to: '/columns', label: 'Columns', icon: Columns3 },
      { to: '/quality', label: 'Data Quality', icon: ShieldCheck },
    ],
  },
  {
    label: 'Explore',
    items: [
      { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
      { to: '/insights', label: 'Insights', icon: Lightbulb },
      { to: '/bi', label: 'Business Intel', icon: Briefcase },
      { to: '/eda', label: 'Deep EDA', icon: FlaskConical },
    ],
  },
  {
    label: 'Analyse',
    items: [
      { to: '/deep-analysis', label: 'Deep Analysis', icon: Gauge },
      { to: '/segments', label: 'Customer Segments', icon: Users },
      { to: '/compare', label: 'Compare', icon: GitCompare },
    ],
  },
  {
    label: 'Test & predict',
    items: [
      { to: '/ml', label: 'ML Predictions', icon: Brain },
      { to: '/ab-test', label: 'A/B Test', icon: FlaskRound },
      { to: '/survival', label: 'Survival', icon: Activity },
    ],
  },
  {
    label: 'Deliver',
    items: [
      { to: '/reports', label: 'Reports', icon: FileText },
      { to: '/chat', label: 'AI Chat', icon: MessageSquare },
      { to: '/rag', label: 'RAG Studio', icon: Layers },
    ],
  },
]

function Brand() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-accent/25 bg-accent/12">
        <Database className="h-4 w-4 text-accent" />
      </div>
      <div className="min-w-0">
        <div className="text-sm font-bold tracking-tight text-ink">Analytiq</div>
        <div className="text-[10px] tracking-wide text-faint">Data Analysis Workbench</div>
      </div>
    </div>
  )
}

export default function Layout() {
  const dataset = useApp((s) => s.dataset)
  const setDataset = useApp((s) => s.setDataset)

  // The selected dataset is persisted in localStorage, so it outlives the
  // dataset itself (deleted, or expired by the retention sweep). Without
  // this check every page then failed with "Dataset not found" and the
  // sidebar kept naming a file that no longer exists.
  const datasetId = dataset?.dataset_id
  useEffect(() => {
    if (!datasetId) return
    apiGet(`/api/datasets/${datasetId}`).catch((e) => {
      if (e instanceof ApiError && e.status === 404) setDataset(null)
    })
  }, [datasetId, setDataset])

  // On a phone the sidebar is a drawer; it closes whenever the page changes.
  const [open, setOpen] = useState(false)
  const location = useLocation()
  useEffect(() => setOpen(false), [location.pathname])

  return (
    <div className="flex h-full flex-col md:flex-row">
      {/* Phone top bar. The fixed 240px sidebar left a 150px strip for
          the page on a 390px screen. */}
      <header className="flex items-center justify-between border-b border-edge bg-panel px-4 py-3 md:hidden">
        <Brand />
        <button
          onClick={() => setOpen(true)}
          className="rounded-lg border border-edge p-2 text-ink2"
          aria-label="Open navigation"
        >
          <Menu className="h-4 w-4" />
        </button>
      </header>
      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/60 md:hidden"
          onClick={() => setOpen(false)}
          aria-hidden
        />
      )}
      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-64 shrink-0 flex-col border-r border-edge bg-panel transition-transform duration-200 md:static md:w-60 md:translate-x-0 ${
          open ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex items-center justify-between px-5 py-5">
          <Brand />
          <button
            onClick={() => setOpen(false)}
            className="rounded p-1 text-faint hover:text-ink md:hidden"
            aria-label="Close navigation"
          >
            <X className="h-4 w-4" />
          </button>
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
                <span className="truncate text-[12px] font-medium text-ink2" title={dataset.filename}>
                  {dataset.filename}
                </span>
              </div>
              <div className="mt-0.5 flex items-center justify-between pl-3">
                <span className="font-data text-[11px] text-faint">
                  {dataset.rows.toLocaleString()} × {dataset.cols}
                </span>
                <Link to="/" className="text-[11px] font-medium text-accent hover:underline">
                  Change
                </Link>
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
              // The next person to sign in on this browser must not inherit
              // this one's dataset selection (it names their file).
              setDataset(null)
              window.location.reload()
            }}
            className="flex items-center gap-2 border-t border-edge px-4 py-3 text-[11px] text-faint transition-colors hover:bg-panel2 hover:text-ink"
          >
            <LogOut className="h-3.5 w-3.5" /> Sign out
          </button>
        )}
      </aside>
      <main className="min-w-0 flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  )
}
