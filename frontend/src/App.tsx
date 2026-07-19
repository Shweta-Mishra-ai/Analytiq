import { BrowserRouter, Route, Routes } from 'react-router-dom'
import AuthGate from './components/AuthGate'
import Layout from './components/Layout'
import UploadPage from './pages/UploadPage'
import QualityPage from './pages/QualityPage'
import DashboardPage from './pages/DashboardPage'
import EdaPage from './pages/EdaPage'
import InsightsPage from './pages/InsightsPage'
import BiPage from './pages/BiPage'
import MlPage from './pages/MlPage'
import ChatPage from './pages/ChatPage'
import RagPage from './pages/RagPage'
import ReportsPage from './pages/ReportsPage'

export default function App() {
  return (
    <BrowserRouter>
      <AuthGate>
        <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<UploadPage />} />
          <Route path="/quality" element={<QualityPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/eda" element={<EdaPage />} />
          <Route path="/insights" element={<InsightsPage />} />
          <Route path="/bi" element={<BiPage />} />
          <Route path="/ml" element={<MlPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/rag" element={<RagPage />} />
          <Route path="/reports" element={<ReportsPage />} />
        </Route>
        </Routes>
      </AuthGate>
    </BrowserRouter>
  )
}
