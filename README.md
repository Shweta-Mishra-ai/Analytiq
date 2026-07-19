# Analytiq 🔬

> **AI-powered data analytics platform — a real app.**
> FastAPI backend + React frontend with a Power BI-style dashboard,
> multimodal RAG (documents, spreadsheets, **images and video**),
> ML predictions, and senior-analyst PDF reports.

Successor to the Streamlit-based Analytiq — every feature preserved,
rebuilt as a production web application.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📥 **Smart Upload** | CSV, Excel (multi-sheet), JSON — up to 200 MB, dirty-data tolerant |
| 🧹 **Data Quality** | Per-column quality scores, outliers, one-click auto-clean with undo |
| 📊 **Power BI-style Dashboard** | Draggable/resizable tiles, KPI strip, tile builder, **cross-filtering** — click any bar or slice and every tile re-computes |
| 🔬 **Deep EDA** | Normality tests, distribution fitting, VIF, ANOVA/Kruskal-Wallis, time-series stationarity |
| 💡 **Business Insights** | Domain-aware (HR/Ecommerce/Sales) Problem → Cause → Evidence → Action → Impact cards, attrition analysis |
| 📈 **Business Intel** | Root cause analysis, cohorts, Pareto, segment health scores |
| 🤖 **ML Predictions** | Auto task detection, model leaderboard with cross-validation, feature importance |
| 💬 **AI Chat** | Plain-English questions → safe tool-calling (no code execution) → answers, charts, tables |
| 🧠 **RAG Studio** | Knowledge bases of PDFs, DOCX, CSVs, **images and video**. Gemini vision reads charts, transcribes text and narration. Ask questions with citations or generate an executive report (Markdown + PDF) |
| 📄 **Reports** | Senior-analyst PDF (cover, TOC, insight cards, benchmarks, AI chart narratives), Excel and CSV exports |

## 🛠️ Stack

```
Backend    FastAPI · pandas · scikit-learn · scipy · statsmodels · ReportLab
Frontend   React 19 · TypeScript · Tailwind 4 · Plotly · react-grid-layout · Zustand
AI         Groq (Llama 3.3) · Google Gemini (vision, video, embeddings)
RAG        Custom numpy vector store · Gemini text-embedding-004
Deploy     Docker · Render / Railway
```

## 🚀 Local development

**Backend** (Python 3.11+):
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env          # add your API keys
uvicorn app.main:app --reload --port 8000
```

**Frontend** (Node 20+):
```bash
cd frontend
npm install
npm run dev                   # http://localhost:5173 (proxies /api to :8000)
```

**Tests** — 37-check end-to-end smoke test, no API keys needed:
```bash
cd backend && python -m tests.smoke_test
```

## 🔑 API keys

| Key | Where | Used for |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) (free) | AI Chat, chart narratives |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) (free) | Image & video analysis, RAG embeddings, report generation |

The app runs without keys — AI features return clear "configure key" messages,
everything else works.

## ☁️ Deploy

**Render** — push to GitHub, create a Blueprint from `render.yaml`,
set the two API keys in the dashboard. Done.

**Railway** — new project from repo (Dockerfile auto-detected),
add the two env vars, deploy.

**Any Docker host:**
```bash
docker build -t dataforge-pro .
docker run -p 8000:8000 -e GROQ_API_KEY=... -e GEMINI_API_KEY=... \
  -v dataforge-data:/srv/data dataforge-pro
```

One container serves both the API and the built frontend.

## 📁 Structure

```
dataforge-pro/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app + static frontend serving
│   │   ├── config.py          # env-driven settings
│   │   ├── api/               # datasets, analytics, charts, ml, chat, rag, reports
│   │   ├── engines/           # analytics engines (profiler, EDA, BI, ML, story, PDF…)
│   │   ├── ai/                # LLM client, safe tool dispatcher, narrator
│   │   ├── rag/               # extractors (incl. Gemini vision/video), vector store
│   │   └── services/          # dataset store, filters, serialization
│   └── tests/smoke_test.py
├── frontend/                  # Vite + React + TS
│   └── src/pages/             # one file per page
├── Dockerfile                 # multi-stage: frontend build → single container
├── render.yaml
└── railway.json
```

## 📝 License

© 2026 Shweta Mishra. All rights reserved.

*Built by Shweta Mishra*
