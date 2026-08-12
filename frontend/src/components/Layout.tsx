import { NavLink, Outlet } from 'react-router-dom'
import {
  Upload,
  ShieldCheck,
  LayoutDashboard,
  FlaskConical,
  Brain,
  Lightbulb,
  Briefcase,
  FileText,
  MessageSquare,
  Database,
  Layers,
  LogOut,
} from 'lucide-react'
import { useApp } from '../store/app'
import { getToken, setToken } from '../api/client'

const nav = [
  { to: '/', label: 'Upload', icon: Upload },
  { to: '/quality', label: 'Data Quality', icon: ShieldCheck },
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/eda', label: 'Deep EDA', icon: FlaskConical },
  { to: '/insights', label: 'Insights', icon: Lightbulb },
  { to: '/bi', label: 'Business Intel', icon: Briefcase },
  { to: '/ml', label: 'ML Predictions', icon: Brain },
  { to: '/chat', label: 'AI Chat', icon: MessageSquare },
  { to: '/rag', label: 'RAG Studio', icon: Layers },
  { to: '/reports', label: 'Reports', icon: FileText },
]

export default function Layout() {
  const dataset = useApp((s) => s.dataset)
  return (
    <div className="flex h-full">
      <aside className="flex w-56 shrink-0 flex-col border-r border-edge bg-panel">
        <div className="flex items-center gap-2 px-5 py-5">
          <Database className="h-6 w-6 text-accent" />
          <div>
            <div className="font-data text-sm font-semibold tracking-tight">
              Analytiq
            </div>
            <div className="text-[10px] text-mute">Data Analysis Workbench</div>
          </div>
        </div>
        <nav className="flex-1 space-y-0.5 px-2">
          {nav.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-2 text-[13px] transition-colors ${
                  isActive
                    ? 'bg-accent/15 font-semibold text-accent'
                    : 'text-mute hover:bg-panel2 hover:text-ink'
                }`
              }
            >
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="border-t border-edge px-4 py-3 text-[11px] text-mute">
          {dataset ? (
            <>
              <div className="truncate font-medium text-ink">
                {dataset.filename}
              </div>
              <div>
                {dataset.rows.toLocaleString()} rows × {dataset.cols} cols
              </div>
            </>
          ) : (
            'No dataset loaded'
          )}
        </div>
        {getToken() && (
          <button
            onClick={() => {
              setToken('')
              window.location.reload()
            }}
            className="flex items-center gap-2 border-t border-edge px-4 py-3 text-[11px] text-mute transition-colors hover:bg-panel2 hover:text-ink"
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
