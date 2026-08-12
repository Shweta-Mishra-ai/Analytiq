"""
Analytiq — FastAPI backend.
Serves the API and (in production) the built React frontend.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pydantic import BaseModel

from app.config import config
from app.api import advanced_analytics, analytics, charts, chat, datasets, ml, reports
from app.services.auth import AuthMiddleware
from app.services.cleanup import cleanup_loop, sweep_expired
from app.services.user_store import user_store

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Background sweep of datasets/RAG knowledge bases past DATA_TTL_DAYS.
    # Set DATA_TTL_DAYS=0 to disable. Runs once immediately, then on
    # CLEANUP_INTERVAL_HOURS. Cancelled cleanly on shutdown.
    task = asyncio.create_task(cleanup_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title=config.app_name, version=config.app_version, lifespan=lifespan)

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
    username: str
    password: str


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    """Exchange a client's username/password for their scoped bearer
    token. In single-user open mode (no admin key set, no accounts
    created yet) auth is bypassed entirely and this endpoint is unused
    by the frontend."""
    from fastapi import HTTPException
    from app.services.tokens import issue_token
    user = user_store.verify(req.username, req.password)
    if not user:
        raise HTTPException(401, "Wrong username or password")
    return {"token": issue_token(user.username), "username": user.username,
            "is_admin": user.is_admin}

app.include_router(datasets.router)
app.include_router(analytics.router)
app.include_router(advanced_analytics.router)
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
    from app.services.auth import _open_mode
    return {
        "status": "ok",
        "app": config.app_name,
        "version": config.app_version,
        "auth_required": not _open_mode(),
        "groq_configured": bool(config.groq_api_key),
        "gemini_configured": bool(config.gemini_api_key),
    }


class CreateUserRequest(BaseModel):
    username: str
    password: str


@app.post("/api/admin/users")
async def create_client_user(req: CreateUserRequest):
    """Onboard a new client account. Admin-key protected (see
    services/auth.py) — clients never see or use this endpoint."""
    from fastapi import HTTPException
    try:
        user = user_store.create(req.username, req.password)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"username": user.username, "created_at": user.created_at}


@app.get("/api/admin/users")
async def list_client_users():
    return {"users": [
        {"username": u.username, "created_at": u.created_at, "is_admin": u.is_admin}
        for u in user_store.list()
    ]}


@app.delete("/api/admin/users/{username}")
async def delete_client_user(username: str):
    """Offboard a client: removes their account AND cascades to delete
    every dataset and RAG knowledge base they own — a full data wipe,
    not just access revocation."""
    from fastapi import HTTPException
    from starlette.concurrency import run_in_threadpool
    from app.services.dataset_store import store as dataset_store
    if not user_store.exists(username):
        raise HTTPException(404, "No such user")

    def _cascade_delete():
        removed_ds = []
        for meta in dataset_store.list_meta(username):
            if dataset_store.delete(username, meta.dataset_id):
                removed_ds.append(meta.dataset_id)
        removed_kbs = []
        try:
            from app.rag.vector_store import RagStore
            rs = RagStore()
            for kb in rs.list(username):
                if rs.delete(username, kb["kb_id"]):
                    removed_kbs.append(kb["kb_id"])
        except ImportError:
            pass
        return removed_ds, removed_kbs

    removed_ds, removed_kbs = await run_in_threadpool(_cascade_delete)
    user_store.delete(username)
    return {"deleted_user": username, "datasets_removed": removed_ds,
            "knowledge_bases_removed": removed_kbs}


@app.post("/api/admin/cleanup")
async def run_cleanup():
    """Manually trigger the storage-lifecycle sweep across every client's
    data (also runs automatically every CLEANUP_INTERVAL_HOURS)."""
    from starlette.concurrency import run_in_threadpool
    result = await run_in_threadpool(sweep_expired)
    return {
        "datasets_deleted": result.datasets_deleted,
        "kbs_deleted": result.kbs_deleted,
        "errors": result.errors,
        "ttl_days": config.data_ttl_days,
    }


# ── Serve built frontend (production single-container deploy) ──
_static = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(_static):
    app.mount("/assets", StaticFiles(directory=os.path.join(_static, "assets")),
              name="assets")

    @app.get("/{full_path:path}")
    async def spa(full_path: str):
        # Never let the SPA fallback swallow API routes — a typo'd path,
        # wrong HTTP method, or missing router must surface as a real
        # 404 JSON error, not a 200 with the app's index.html. Without
        # this guard, every broken /api/... call looks "successful" to
        # monitoring, tests, and browser devtools alike.
        if full_path.startswith("api/"):
            from fastapi import HTTPException
            raise HTTPException(404, f"No API route matches /{full_path}")
        target = os.path.join(_static, full_path)
        if full_path and os.path.isfile(target):
            return FileResponse(target)
        return FileResponse(os.path.join(_static, "index.html"))
