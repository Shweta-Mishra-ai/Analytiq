"""
Analytiq — FastAPI backend.
Serves the API and (in production) the built React frontend.
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pydantic import BaseModel

from app.config import config
from app.api import analytics, charts, chat, datasets, ml, reports
from app.services.auth import AuthMiddleware, check_password

logging.basicConfig(level=logging.INFO)

app = FastAPI(title=config.app_name, version=config.app_version)

origins = [o.strip() for o in config.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuthMiddleware)


class LoginRequest(BaseModel):
    password: str


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    """Exchange the workspace password for the bearer token (same value).
    Exists so the frontend can validate credentials before storing them."""
    if not config.app_password:
        return {"token": "", "auth_required": False}
    if check_password(req.password):
        return {"token": req.password, "auth_required": True}
    from fastapi import HTTPException
    raise HTTPException(401, "Wrong password")

app.include_router(datasets.router)
app.include_router(analytics.router)
app.include_router(charts.router)
app.include_router(ml.router)
app.include_router(chat.router)
app.include_router(reports.router)

try:
    from app.api import rag
    app.include_router(rag.router)
except Exception as e:  # RAG deps optional in dev
    logging.getLogger(__name__).warning(f"RAG module not loaded: {e}")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "app": config.app_name,
        "version": config.app_version,
        "auth_required": bool(config.app_password),
        "groq_configured": bool(config.groq_api_key),
        "gemini_configured": bool(config.gemini_api_key),
    }


# ── Serve built frontend (production single-container deploy) ──
_static = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(_static):
    app.mount("/assets", StaticFiles(directory=os.path.join(_static, "assets")),
              name="assets")

    @app.get("/{full_path:path}")
    async def spa(full_path: str):
        target = os.path.join(_static, full_path)
        if full_path and os.path.isfile(target):
            return FileResponse(target)
        return FileResponse(os.path.join(_static, "index.html"))
