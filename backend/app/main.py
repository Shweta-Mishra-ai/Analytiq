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
from app.api import (accounts, admin, advanced_analytics, analytics,
                     billing, charts, chat, datasets, deliver, ml, reports)
from app.services.auth import AuthMiddleware
from app.services import request_context
from app.services.cleanup import cleanup_loop
from app.services.user_store import user_store

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


# How often the ticker looks for schedules that are due. Cadences are
# daily at the finest, so a five-minute check is far more often than
# needed — which is the point: a schedule set for 07:00 should fire
# near 07:00, not at whatever moment a coarse timer happens to land on.
SCHEDULE_TICK_SECONDS = 300


async def schedule_loop():
    """Produce what is due, and sweep what has expired.

    Both live here because both are promises the product makes and
    neither should depend on somebody remembering to configure a cron
    job. The tick is idempotent — a schedule records the PERIOD it last
    produced, not the last time it ran — so a restart or an overlapping
    tick still produces exactly one report per window.
    """
    from app.services.artifacts import store as artifacts
    from app.services.jobs import reap_all
    from app.services.schedules import run_due

    while True:
        try:
            await asyncio.sleep(SCHEDULE_TICK_SECONDS)
            started = await asyncio.to_thread(run_due)
            if started:
                logging.getLogger(__name__).info(
                    "%d scheduled report(s) started", started)
            await asyncio.to_thread(artifacts.sweep)
            # A job a schedule started has nobody polling it, so a
            # restart would leave it reading "queued" for ever and the
            # schedule reporting that as its outcome.
            await asyncio.to_thread(reap_all)
        except asyncio.CancelledError:
            raise
        except Exception:
            # A ticker that dies on one bad schedule stops every other
            # one silently, which is the worst possible failure for a
            # feature whose entire promise is that it keeps happening.
            logging.getLogger(__name__).error(
                "schedule tick failed; continuing", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Background sweep of datasets/RAG knowledge bases past DATA_TTL_DAYS.
    # Set DATA_TTL_DAYS=0 to disable. Runs once immediately, then on
    # CLEANUP_INTERVAL_HOURS. Cancelled cleanly on shutdown.
    # Import the heavy engines (scikit-learn, statsmodels, ReportLab) off
    # the request path. Without this the first user to open Predict waits
    # ~5s for an endpoint whose actual work takes 70ms.
    from app.services import warmup
    warmup.start()

    # What each background job kind does. Registered here rather than
    # on first use: the scheduler reaches the runner without going
    # through the API, and a schedule that fired into an empty handler
    # table failed with "There is no job of kind 'health-report'".
    from app.services import report_jobs      # noqa: F401  (registers)

    tasks = [asyncio.create_task(cleanup_loop()),
             asyncio.create_task(schedule_loop())]
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass


app = FastAPI(title=config.app_name, version=config.app_version, lifespan=lifespan)

# Cross-origin access is off unless somebody asks for it by name. The
# app's own UI never needs it — Vite proxies /api in development and
# FastAPI serves the built frontend in production, so the browser is
# same-origin either way.
origins = [o.strip() for o in config.cors_origins.split(",") if o.strip()]
if origins:
    if "*" in origins:
        logger.warning(
            "CORS_ORIGINS is '*', so any website a user visits can call "
            "this API from their browser. With APP_ADMIN_KEY unset there "
            "is no authentication either, and that combination lets a "
            "page read every dataset on this machine. Name the origins "
            "you actually serve instead.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
app.add_middleware(AuthMiddleware)

# Outermost, so it wraps auth too: a 401 is a request worth
# finding in the log, and an id that only exists after
# authentication cannot explain a failure to authenticate.
request_context.install(app)


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

app.include_router(accounts.router)
app.include_router(billing.router)
app.include_router(datasets.router)
app.include_router(analytics.router)
app.include_router(advanced_analytics.router)
app.include_router(charts.router)
app.include_router(ml.router)
app.include_router(chat.router)
app.include_router(reports.router)
app.include_router(deliver.router)
app.include_router(admin.router)

try:
    from app.api import rag
    app.include_router(rag.router)
except Exception as e:  # RAG deps optional in dev
    logging.getLogger(__name__).warning(f"RAG module not loaded: {e}")


@app.get("/api/health")
async def health():
    from app.ai.local_llm import status as _llm_status
    from app.services.auth import _open_mode
    return {
        "status": "ok",
        "app": config.app_name,
        "version": config.app_version,
        "auth_required": not _open_mode(),
        "groq_configured": bool(config.groq_api_key),
        "gemini_configured": bool(config.gemini_api_key),
        # What will actually happen to a narrative request, so a client
        # running in privacy mode can confirm it from outside the app.
        "llm": _llm_status(),
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
