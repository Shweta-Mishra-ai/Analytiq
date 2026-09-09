import { lazy, Suspense } from 'react'
import { BrowserRouter, Link, Route, Routes } from 'react-router-dom'
import AuthGate from './components/AuthGate'
import Layout from './components/Layout'

// Route-level code splitting: each page becomes its own chunk, fetched
// only when visited, so landing on Upload doesn't pull in the EDA/BI/ML
// page code too. (Plotly itself is already split separately, inside
// PlotlyChart.tsx.)
const UploadPage = lazy(() => import('./pages/UploadPage'))
const QualityPage = lazy(() => import('./pages/QualityPage'))
const GovernancePage = lazy(() => import('./pages/GovernancePage'))
const DashboardPage = lazy(() => import('./pages/DashboardPage'))
const EdaPage = lazy(() => import('./pages/EdaPage'))
const InsightsPage = lazy(() => import('./pages/InsightsPage'))
const BiPage = lazy(() => import('./pages/BiPage'))
const DeepAnalysisPage = lazy(() => import('./pages/DeepAnalysisPage'))
const SegmentsPage = lazy(() => import('./pages/SegmentsPage'))
const AbTestPage = lazy(() => import('./pages/AbTestPage'))
const SurvivalPage = lazy(() => import('./pages/SurvivalPage'))
const ComparePage = lazy(() => import('./pages/ComparePage'))
const MlPage = lazy(() => import('./pages/MlPage'))
const ChatPage = lazy(() => import('./pages/ChatPage'))
const RagPage = lazy(() => import('./pages/RagPage'))
const ReportsPage = lazy(() => import('./pages/ReportsPage'))
const SystemPage = lazy(() => import('./pages/SystemPage'))

function NotFound() {
  return (
    <div className="p-8">
      <div className="mx-auto mt-16 max-w-md rounded-xl border border-edge bg-panel2 px-6 py-10 text-center">
        <p className="text-sm font-semibold text-ink">
          There is no page at this address
        </p>
        <p className="mt-2 text-xs leading-relaxed text-mute">
          The link may be out of date, or the address mistyped. Everything
          the app can do is in the sidebar.
        </p>
        <Link
          to="/"
          className="mt-5 inline-block rounded-lg btn-primary px-4 py-2 text-sm font-semibold text-white"
        >
          Go to Upload
        </Link>
      </div>
    </div>
  )
}

function RouteFallback() {
  return (
    <div className="flex h-full items-center justify-center py-24 text-xs text-mute">
      Loading…
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthGate>
        <Suspense fallback={<RouteFallback />}>
          <Routes>
            <Route element={<Layout />}>
              <Route path="/" element={<UploadPage />} />
              <Route path="/quality" element={<QualityPage />} />
              <Route path="/governance" element={<GovernancePage />} />
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/eda" element={<EdaPage />} />
              <Route path="/insights" element={<InsightsPage />} />
              <Route path="/bi" element={<BiPage />} />
              <Route path="/deep-analysis" element={<DeepAnalysisPage />} />
              <Route path="/segments" element={<SegmentsPage />} />
              <Route path="/ab-test" element={<AbTestPage />} />
              <Route path="/survival" element={<SurvivalPage />} />
              <Route path="/compare" element={<ComparePage />} />
              <Route path="/ml" element={<MlPage />} />
              <Route path="/chat" element={<ChatPage />} />
              <Route path="/rag" element={<RagPage />} />
              <Route path="/reports" element={<ReportsPage />} />
              <Route path="/system" element={<SystemPage />} />
              {/* Without this, any address that is not in the list above
                  renders an empty page inside the app shell — and a
                  blank screen reads as a crash, not as a wrong URL.
                  /upload is the most likely one to be typed, since that
                  is what the nav item is called. */}
              <Route path="*" element={<NotFound />} />
            </Route>
          </Routes>
        </Suspense>
      </AuthGate>
    </BrowserRouter>
  )
}
