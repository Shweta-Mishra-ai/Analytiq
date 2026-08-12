import { lazy, Suspense } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import AuthGate from './components/AuthGate'
import Layout from './components/Layout'

// Route-level code splitting: each page becomes its own chunk, fetched
// only when visited, so landing on Upload doesn't pull in the EDA/BI/ML
// page code too. (Plotly itself is already split separately, inside
// PlotlyChart.tsx.)
const UploadPage = lazy(() => import('./pages/UploadPage'))
const QualityPage = lazy(() => import('./pages/QualityPage'))
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
            </Route>
          </Routes>
        </Suspense>
      </AuthGate>
    </BrowserRouter>
  )
}
