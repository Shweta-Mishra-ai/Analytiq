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
| 💡 **Strategic Insights** | **Business Intel** | Domain-aware insight cards (HR/Sales/E-commerce), Root Cause analysis, cohort tracking, and Pareto charts. |
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
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | Powering the AI Chat copilot and chart generation narratives |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) | Powering image/video analysis, RAG indexing, and executive summaries |

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
