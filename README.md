<p align="center">
  <img src="docs/assets/logo.svg" alt="Analytiq Logo" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-Proprietary-red.svg" alt="License: Proprietary" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688.svg?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Frontend-React%2019-61DAFB.svg?logo=react&logoColor=white" alt="React 19" />
  <img src="https://img.shields.io/badge/Language-TypeScript-3178C6.svg?logo=typescript&logoColor=white" alt="TypeScript" />
  <img src="https://img.shields.io/badge/AI-Groq%20%7C%20Gemini-orange.svg" alt="AI Services" />
  <img src="https://img.shields.io/badge/Docker-Supported-blue.svg?logo=docker&logoColor=white" alt="Docker" />
</p>

<br />

---

### 🔬 **Analytiq** is a premium, enterprise-grade AI-powered data analytics and business intelligence platform.

Rebuilt from the ground up as a production-ready application, it pairs a fast **FastAPI Python backend** with a highly interactive **React 19 + TypeScript frontend**. It features a draggable Power BI-style dashboard, multimodal RAG studio, auto-ML pipelines, statistical EDA, and auto-generated senior-analyst PDF reports.

<p align="center">
  <img src="docs/assets/dashboard_mockup.png" alt="Analytiq Dashboard Mockup" width="100%" style="border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,0.3);" />
</p>

---

## 🗺️ Architectural Flow

Here is the high-level architecture and system flow of the Analytiq ecosystem:

```mermaid
graph TD
    subgraph Frontend [React 19 Frontend]
        UI["Vite & Tailwind Dashboard"] --> Auth["Auth Gate"]
        UI --> DS_Grid["Draggable Grid (react-grid-layout)"]
        DS_Grid --> Plotly["Plotly.js Dynamic Visualizations"]
        UI --> ChatUI["AI Chat Interface"]
        UI --> RAGStudio["RAG KB Management"]
    end

    subgraph Backend [FastAPI Backend]
        API["FastAPI Router"] --> Store["Dataset Store"]
        API --> Cleaner["Auto-Cleaner Engine"]
        API --> EDA["Deep EDA Engine"]
        API --> BI["BI Root Cause & Segment Engine"]
        API --> ML["ML AutoML Leaderboard"]
        API --> RAG["RAG Service"]
        API --> PDF["ReportLab PDF Builder"]
    end

    subgraph AI_Services [AI & LLM Services]
        RAG --> GeminiEmbed["Gemini embedding-004"]
        ChatUI --> ToolDispatch["Safe Tool Dispatcher"]
        ToolDispatch --> Groq["Groq (Llama 3.3)"]
        PDF --> Narrator["AI Chart Narrator"]
        Narrator --> Groq
    end

    subgraph Data_Sources [Data Sources & KB]
        CSV["CSV / Excel / JSON Files"] --> API
        Docs["PDF / DOCX / Spreadsheets / Media"] --> RAG
    end

    style Frontend fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#fff
    style Backend fill:#0f172a,stroke:#8b5cf6,stroke-width:2px,color:#fff
    style AI_Services fill:#0f172a,stroke:#06b6d4,stroke-width:2px,color:#fff
    style Data_Sources fill:#1e293b,stroke:#475569,stroke-width:1px,color:#fff
```

---

## ✨ Enterprise Features

| Category | Feature | Description |
|---|---|---|
| 📥 **Smart Ingestion** | **Robust Upload** | Handles CSV, Multi-Sheet Excel, and JSON files up to 200 MB with strict dirty-data tolerance. |
| 🧹 **Data Sanitization** | **Data Quality Hub** | Instant per-column quality profiling, outlier detection, and one-click auto-cleaning with undo features. |
| 📊 **BI & Analytics** | **Power BI Dashboard** | Draggable, resizable layout tiles, dynamic KPI strip, custom tile builder, and automatic cross-filtering. |
| 🔬 **Statistical Suite** | **Deep EDA** | Normality testing, distribution fitting, time-series stationarity checks, ANOVA, and VIF calculations. |
| 💡 **Strategic Insights** | **Business Intel** | Domain-aware insight cards across eight domains (HR, Sales, Finance, E-commerce, Marketing, SaaS, Operations, Healthcare), Root Cause analysis, cohort tracking, and Pareto charts. See [docs/ADDING_A_DOMAIN.md](docs/ADDING_A_DOMAIN.md). |
| 🤖 **Automated ML** | **Predictive modeling** | Auto task detection (Classification/Regression), CV leaderboard scoring, and feature importance mappings. |
| 💬 **AI Copilot** | **Safe Chat Agent** | Plain-English query processor -> secure tool dispatcher (no code execution) -> interactive charts and tables. |
| 🧠 **Knowledge Store** | **RAG Studio** | Custom local vector index ingestion of PDFs, DOCX, CSVs, and video/images via Gemini Vision embeddings. |
| 📄 **Executive Reports** | **Document Generator** | Beautifully styled ReportLab PDF reports (cover page, TOC, benchmarks, and AI-narrated chart guides). |

---

## 🛠️ The Tech Stack

```
Backend ──── FastAPI · pandas · scikit-learn · scipy · statsmodels · ReportLab
Frontend ─── React 19 · TypeScript · Tailwind CSS 4 · Plotly.js · Zustand
AI ───────── Groq (Llama 3.3) · Google Gemini (Vision, Embeddings)
RAG ──────── Custom NumPy Vector Store · Gemini text-embedding-004
Deployment ─ Docker · Render / Railway
```

---

## 🚀 Local Development

### 1. Backend Setup (Python 3.11+)
```bash
# Navigate to the backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Create environment configuration
cp .env.example .env          # Update with your actual API keys

# Start the server (runs on http://localhost:8000)
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend Setup (Node.js 20+)
```bash
# Navigate to the frontend directory
cd frontend

# Install node dependencies
npm install

# Start local development server (proxies /api to port 8000)
npm run dev
```

### 3. Running Automated Tests
Run the 37-point end-to-end integration and smoke test suite:
```bash
cd backend
python tests/smoke_test.py
```

---

## 🔑 Configuration & Keys

The application remains fully functional locally without keys (AI modules degrade gracefully and show setup alerts). Add these to your `.env` to activate intelligence features:

| Environment Variable | Source | Functionality |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | Free tier. Powers the AI Chat copilot and chart narratives |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) | Powers image/video analysis, RAG indexing, and executive summaries |
| `OPENROUTER_API_KEY` | [openrouter.ai](https://openrouter.ai) | Optional. The `:free` model slugs cost nothing — Llama 3.3, DeepSeek, Qwen |
| `CEREBRAS_API_KEY` | [cloud.cerebras.ai](https://cloud.cerebras.ai) | Optional. Free tier, openly-licensed weights, very fast |
| `TOGETHER_API_KEY` | [api.together.xyz](https://api.together.xyz) | Optional. Small free credit, openly-licensed weights |
| `LOCAL_LLM_URL` | your own machine | Optional. Any OpenAI-compatible server (Ollama, llama.cpp, vLLM, LM Studio). No key, no cost, no data leaving the box |
| `LLM_ROUTING` | you choose | Which provider does which job, e.g. `executive_summary=openrouter,narrative=local`. Overrides the defaults without a code change |
| `LLM_PROVIDER_ORDER` | you choose | The fallback chain. Default `groq,openrouter,cerebras,together,gemini,local`. A provider with no key is skipped, not an error |
| `LLM_PRIVACY_MODE` | you choose | `1` refuses every cloud call outright. Only a local model may be used. Reports still build — the engines write their own findings |
| `APP_ADMIN_KEY` | you choose | Master key for account management (`/api/admin/*`). Unset + zero accounts created = open/no-auth (local dev only). `APP_PASSWORD` also works as a fallback name. |
| `DATA_TTL_DAYS` | you choose | Days before an uploaded dataset or RAG knowledge base is auto-deleted. Default `30`. Set `0` to disable expiry entirely. |
| `CLEANUP_INTERVAL_HOURS` | you choose | How often the expiry sweep runs in the background. Default `6`. A sweep also runs once at startup, and can be triggered manually via `POST /api/admin/cleanup`. |
| `APP_SECRET` | you choose | Signs client login tokens. Optional — auto-generated and persisted to `DATA_DIR/.secret_key` if unset. Set explicitly if you run multiple backend instances behind a load balancer, so they all validate the same tokens. |
| `TOKEN_TTL_DAYS` | you choose | How long a client's login session lasts before they must sign in again. Default `30`. |

### 🎬 Video-to-dataset extraction

Uploading a video of a table/spreadsheet/dashboard (`POST /api/datasets/extract-from-video`) extracts a handful of visually-distinct frames locally with **ffmpeg** — free, no API cost — then runs each frame through the same image-table-extraction path as a photo upload, merging the results. This is why the Docker image installs `ffmpeg`; if you deploy without Docker (a bare `pip install` on your own host), install `ffmpeg` yourself or this one endpoint will return a clear "ffmpeg is not installed" error while everything else keeps working.


### 👥 Multi-tenant clients

Each client gets their own account and only ever sees their own datasets and RAG knowledge bases — enforced both by an ownership check on every request and by physically separate storage directories per client. Onboard a client:

```bash
curl -X POST https://your-deployment/api/admin/users \
  -H "Authorization: Bearer $APP_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"username": "acme_corp", "password": "a-strong-password"}'
```

They then log in at `/api/auth/login` with that username/password (the app's login screen does this for them) and get a token scoped to only their own data. To offboard a client — this **permanently deletes every dataset and knowledge base they own**, not just their login:

```bash
curl -X DELETE https://your-deployment/api/admin/users/acme_corp \
  -H "Authorization: Bearer $APP_ADMIN_KEY"
```

List current clients with `GET /api/admin/users` (same admin header).

### ✅ Verifying your keys actually work

A key can be present, well-formed, and still not work — expired, from
the wrong account, out of quota, or blocked by the network the app is
deployed on. None of that is visible from the configuration, and all of
it looks identical from the outside: the reports quietly come back in
the engines' own wording instead of the model's.

The only machine that can answer "does my key work" is the one holding
the key. Secrets set in Render's dashboard, or in GitHub Actions, are
not visible anywhere else — a GitHub secret in particular is *not*
readable by the running service unless a workflow passes it through.

So the check lives in the running app. Open the **System** page, or:

```bash
curl -X POST https://your-deployment/api/admin/llm-check \
  -H "Authorization: Bearer $APP_ADMIN_KEY"
```

It makes one real call to every configured provider and reports, per
provider: whether the key is present, whether the host answered, whether
the model produced words, how long it took — and if not, the provider's
own error message plus what to do about it. `GET /api/admin/llm-status`
is the same picture without making any calls.

Locally, `python3 backend/scripts/check_api_keys.py` does the same from
a terminal.

### 🔒 Data integrity

Governance says what the data is; integrity says whether it can be
trusted. Every upload is hashed as it arrives, every change to it is
appended to a hash-chained audit trail, and the verdict is recomputed
from the stored data on every request rather than read back from a
stored claim.

`GET /api/datasets/{id}/integrity` — and the Governance page, and the
report's Data Governance section — carry the SHA-256 of the file as it
was received. Run `sha256sum` on your original file: if it matches,
every figure in that report came from that file and no other. The report
also records the library versions it was computed with, because a
quantile or a solver default can change between releases.

### 🗄️ Reading from a database instead of a CSV

`POST /api/datasets/warehouse/import` pulls a dataset straight out of
PostgreSQL, MySQL, Snowflake, BigQuery, SQL Server or SQLite, and it
lands in the same store with the same integrity record — except the
audit trail records the *query* it came from rather than a file someone
emailed.

Only `SELECT` and `WITH` are accepted, checked before the statement is
sent, and the connection is rolled back regardless. The connection URL
is supplied per request, never stored, and its password is stripped from
everything the server writes down. SQLAlchemy is included; each
database's own driver (`psycopg[binary]`, `pymysql`,
`snowflake-sqlalchemy`, `sqlalchemy-bigquery`, `pyodbc`) is optional —
`GET /api/datasets/warehouse/backends` reports which are installed and
names the package for the rest.



---

## ☁️ Deployment Guides

### **Render Deployment**
1. Push this code repository to GitHub.
2. Link your repository in Render and create a **Web Service** using the Blueprint configuration in `render.yaml`.
3. Set your environment keys (`GROQ_API_KEY`, `GEMINI_API_KEY`) in the Render Dashboard.

### **Railway Deployment**
1. Initialize a new project on Railway from your repository.
2. Railway will automatically detect the root `Dockerfile` and build a multi-stage production container.
3. Configure variables and deploy.

### **Manual Docker Execution**
```bash
# Build the container
docker build -t analytiq-platform .

# Run the container
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_key \
  -e GEMINI_API_KEY=your_key \
  -v analytiq-data:/srv/data \
  analytiq-platform
```

---

## 📝 License

This software is subject to a proprietary license. 
Copyright © 2026 Shweta Mishra. All rights reserved. Unauthorized distribution, copying, or modification is prohibited.
