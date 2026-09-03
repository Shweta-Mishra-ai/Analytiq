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


@app.get("/api/admin/metrics")
async def app_metrics():
    """Operational counters: how long reports actually take, which engines
    are failing, and how many LLM calls the narrative cache avoided.

    In-process and reset by a restart — this answers "why is this slow"
    and "why did that section disappear", it is not a time series. Under
    /api/admin because a failure message can name a column from a
    client's data; the auth middleware gates the whole prefix.
    """
    from app.services.metrics import metrics
    return metrics.snapshot()


@app.get("/api/admin/llm-status")
async def llm_status():
    """What this deployment is configured to use, without calling
    anything. Fast, safe to poll, and never reveals a key — only whether
    one is present and, when it is not, the exact variable name to set."""
    from app.ai.llm_client import get_client
    return get_client().status()


@app.post("/api/admin/llm-check")
async def llm_check(providers: str = "", timeout: float = 12.0):
    """Actually call every configured provider and report what happened.

    This exists because a key can be present, well-formed, and still not
    work — expired, wrong account, out of quota, or blocked by the
    network the app is deployed on. None of that is visible from the
    configuration, and all of it looks identical from the outside: the
    reports quietly come back in the engines' own wording instead of the
    model's, with nothing in the UI to say why.

    It also has to be *here*, in the running service, rather than in a
    developer's terminal. The keys live in the deployment's environment
    (Render's Settings → Environment, say) and a GitHub Actions secret of
    the same name is not visible to the running service at all unless the
    workflow passes it through — so the only machine that can answer
    "does my key work" is the one holding it.

    `providers` narrows the run to a comma-separated subset; `timeout`
    caps each individual call so one stalled host cannot hold the page.
    """
    from starlette.concurrency import run_in_threadpool
    from app.ai import providers as provider_registry
    from app.ai.llm_client import get_client

    only = [n.strip() for n in providers.split(",") if n.strip()] or None
    timeout = max(1.0, min(float(timeout), 60.0))

    checks = await run_in_threadpool(
        provider_registry.check_all, only, timeout)
    rows = [c.as_dict() for c in checks]
    working = [c["name"] for c in rows if c["ok"]]
    status = get_client().status()

    return {
        "checked_at": _now_iso(),
        "providers": rows,
        "working": working,
        "any_working": bool(working),
        "routing": status["routing"],
        "order": status["order"],
        "privacy_mode": status["privacy_mode"],
        # The one line a person actually reads first.
        "summary": _llm_check_summary(rows, working, status["privacy_mode"]),
    }


class RoutingAssignment(BaseModel):
    task: str
    model_id: str = ""


class ModelDeclaration(BaseModel):
    model_id: str
    capabilities: list[str]
    label: str = ""
    tier: str = "balanced"
    context: int = 0
    free: bool = False
    notes: str = ""


def _routing_payload() -> dict:
    """The one shape every routing endpoint returns.

    Built once because the read and the write must agree: a POST that
    answers with a subset of what the GET returns leaves the caller
    holding a half-populated object, and the UI that renders it crashes
    on whichever field the write happened to omit. That is not
    hypothetical — it was a real crash, found by clicking the dropdown.
    """
    from app.ai import routing
    from app.ai.capabilities import DESCRIPTIONS
    from app.ai.settings_store import settings_store

    payload = routing.status()
    payload["capabilities"] = {c.value: text for c, text in DESCRIPTIONS.items()}
    payload["overrides"] = settings_store.as_dict()
    return payload


@app.get("/api/admin/routing")
async def get_routing():
    """Which model does which job, what each job needs, and which models
    could serve it. Everything the System page's routing table renders."""
    return _routing_payload()


@app.post("/api/admin/routing")
async def set_routing(body: RoutingAssignment):
    """Point one task at one model.

    Validated before it is written. An assignment that cannot do the job
    is refused with the reason — accepting it and skipping it at the
    point of use would look exactly like the model never being called.
    """
    from fastapi import HTTPException
    from app.ai.settings_store import RoutingRejected, settings_store
    try:
        settings_store.assign(body.task, body.model_id)
    except RoutingRejected as e:
        raise HTTPException(422, str(e))
    return _routing_payload()


@app.delete("/api/admin/routing")
async def clear_routing():
    """Back to whatever the environment says."""
    from app.ai.settings_store import settings_store
    settings_store.clear()
    return _routing_payload()


@app.post("/api/admin/models")
async def declare_model(body: ModelDeclaration):
    """Record what an operator says a model can do.

    The catalogue cannot know every model — OpenRouter alone serves
    hundreds — so an unknown one is assumed to write text and nothing
    else until someone who knows says otherwise. This is that saying.
    """
    from fastapi import HTTPException
    from app.ai.model_catalogue import catalogue
    try:
        catalogue.declare(body.model_id, body.capabilities, label=body.label,
                          tier=body.tier, context=body.context,
                          free=body.free, notes=body.notes)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return _routing_payload()


@app.delete("/api/admin/models/{model_id:path}")
async def forget_model(model_id: str):
    from fastapi import HTTPException
    from app.ai.model_catalogue import catalogue
    if not catalogue.forget(model_id):
        raise HTTPException(
            404, f"'{model_id}' was not added here. Built-in catalogue "
                 f"entries cannot be removed — declare the same id to "
                 f"change what it claims.")
    return _routing_payload()


@app.post("/api/admin/task-check")
async def task_check(task: str, timeout: float = 12.0):
    """Call the model actually assigned to one task, and report what
    happened — the per-task version of the provider check."""
    from fastapi import HTTPException
    from starlette.concurrency import run_in_threadpool
    from app.ai import providers as provider_registry
    from app.ai import routing, tasks

    spec = tasks.get(task)
    if spec is None:
        raise HTTPException(404, f"'{task}' is not a task this app has.")

    chain = routing.resolve_models(spec.name)
    if not chain:
        return {"task": spec.name, "ok": False, "model": "",
                "error": "No configured model can serve this task.",
                "hint": spec.degrades_to}

    model = chain[0]
    provider = provider_registry.get(model.provider)
    timeout = max(1.0, min(float(timeout), 60.0))
    check = await run_in_threadpool(provider.check, timeout, model.model)
    result = check.as_dict()
    result.update({"task": spec.name, "model": model.id})
    if result["ok"]:
        from app.ai.model_catalogue import catalogue
        from app.ai.capabilities import Capability
        await run_in_threadpool(catalogue.record_probe, model.id,
                                Capability.TEXT, True, "")
    return result


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _llm_check_summary(rows: list, working: list, privacy: bool) -> str:
    """Plain English, because the failure mode this endpoint exists to
    catch is someone reading a wall of JSON and concluding the wrong
    thing."""
    if privacy and not working:
        return ("Privacy mode is on and no local model answered, so every "
                "narrative will be written by the engines themselves. No "
                "data has left this machine.")
    if not rows:
        return "No providers were checked."
    configured = [r for r in rows if r["configured"]]
    if not configured:
        return ("No LLM provider is configured. Reports still build — the "
                "engines write their own wording — but nothing will be "
                "phrased by a model. Set any one of GROQ_API_KEY, "
                "OPENROUTER_API_KEY, CEREBRAS_API_KEY, TOGETHER_API_KEY or "
                "GEMINI_API_KEY, or point LOCAL_LLM_URL at a local model.")
    if not working:
        first = configured[0]
        return (f"{len(configured)} provider(s) are configured but none "
                f"answered. {first['label']}: {first['error']}")
    names = ", ".join(r["label"] for r in rows if r["ok"])
    broken = [r for r in configured if not r["ok"]]
    tail = (f" {len(broken)} configured provider(s) failed — see below."
            if broken else "")
    return f"Working: {names}.{tail}"


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
