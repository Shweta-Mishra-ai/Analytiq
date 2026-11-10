"""
Analytiq — FastAPI backend.
Serves the API and (in production) the built React frontend.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

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
        if _mcp_mount is not None:
            async with _mcp_mount.running():
                yield
        else:
            yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# Analytiq's tools for Claude (Model Context Protocol) at /mcp; see
# app/mcp_server.py. Optional: the web app runs without the mcp package.
try:
    from app.mcp_server import mount as _mcp_mount
except ImportError:                       # pragma: no cover - mcp not installed
    _mcp_mount = None
    logging.getLogger(__name__).warning("mcp package not installed; /mcp is disabled")

app = FastAPI(title=config.app_name, version=config.app_version, lifespan=lifespan)


_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    # The app's own scripts and styles only. Plotly sets inline styles.
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; "
        "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"),
}


def _caller_key(request: Request) -> str:
    """Who a request is counted against: the token when there is one, else
    the address (the first forwarded one behind a proxy)."""
    auth = request.headers.get("authorization", "")
    if auth:
        import hashlib
        return "t:" + hashlib.sha256(auth.encode("utf-8", "replace")).hexdigest()[:24]
    fwd = request.headers.get("x-forwarded-for", "")
    return "ip:" + (fwd.split(",")[0].strip() if fwd else
                    (request.client.host if request.client else "unknown"))


class GuardMiddleware(BaseHTTPMiddleware):
    """Around every request: a reference id, a request rate, a safe answer
    for any unhandled error, and the security headers.

    An unhandled error used to reach the client as a bare "Internal Server
    Error" with nothing to quote and nothing in the log to find it by. It
    is now logged with a reference the client is shown, so a report of
    "it broke" can be traced to its traceback.
    """

    async def dispatch(self, request: Request, call_next):
        rid = uuid.uuid4().hex[:12]
        request.state.request_id = rid
        path = request.url.path
        is_api = path.startswith("/api") or path.startswith("/mcp")
        started = time.time()
        if is_api and path != "/api/health" and request.method != "OPTIONS":
            from app.services.limits import rate_limiter
            wait = rate_limiter.allow(_caller_key(request))
            if wait:
                response = JSONResponse(
                    {"detail": f"Too many requests. Try again in {wait} s."},
                    status_code=429, headers={"Retry-After": str(wait)})
                response.headers["X-Request-ID"] = rid
                return response
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unhandled error ref=%s %s %s user=%s", rid, request.method,
                             path, getattr(request.state, "username", "-"))
            contact = f" at {config.support_email}" if config.support_email else ""
            response = JSONResponse(
                {"detail": "Something went wrong on our side. Please try again; if it "
                           f"keeps happening, contact support{contact} and quote "
                           f"reference {rid}.",
                 "reference": rid}, status_code=500)
        took = time.time() - started
        if is_api and (response.status_code >= 500 or took > 30):
            logger.warning("ref=%s %s %s -> %s in %.1fs user=%s", rid, request.method, path,
                           response.status_code, took,
                           getattr(request.state, "username", "-"))
        response.headers["X-Request-ID"] = rid
        for k, v in _SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if request.headers.get("x-forwarded-proto", request.url.scheme) == "https":
            response.headers.setdefault("Strict-Transport-Security",
                                        "max-age=31536000; includeSubDomains")
        return response


# Innermost first: auth, then the guard around it, then CORS outermost so
# its headers are on every answer, errors included.
app.add_middleware(AuthMiddleware)
app.add_middleware(GuardMiddleware)
origins = [o.strip() for o in config.cors_origins.split(",") if o.strip()]
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
def login(req: LoginRequest):
    """Exchange a client's username/password for their scoped bearer
    token. In single-user open mode (no admin key set, no accounts
    created yet) auth is bypassed entirely and this endpoint is unused
    by the frontend.

    Deliberately a sync route: PBKDF2 at 200k iterations is tens of
    milliseconds of CPU, and in an async route it ran on the event loop,
    stalling every other request for the duration — a login flood was a
    cheap way to freeze the server."""
    from fastapi import HTTPException
    from app.services.auth import login_throttle
    from app.services.tokens import issue_token
    wait = login_throttle.retry_after(req.username)
    if wait:
        raise HTTPException(
            429, f"Too many failed sign-in attempts. Try again in {wait} s.",
            headers={"Retry-After": str(wait)})
    user = user_store.verify(req.username, req.password)
    if not user:
        login_throttle.record_failure(req.username)
        raise HTTPException(401, "Wrong username or password")
    login_throttle.record_success(req.username)
    return {"token": issue_token(user.username), "username": user.username,
            "is_admin": user.is_admin}



class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@app.post("/api/auth/change-password")
def change_password(req: ChangePasswordRequest, request: Request):
    """Change the signed-in client's own password. Every other session of
    theirs is signed out; the answer carries a fresh token for this one."""
    from fastapi import HTTPException
    from app.services.auth import LOCAL_OWNER, _open_mode, login_throttle
    from app.services.tokens import issue_token
    username = getattr(request.state, "username", LOCAL_OWNER)
    if _open_mode() or not user_store.exists(username):
        raise HTTPException(400, "This sign-in has no password to change.")
    wait = login_throttle.retry_after(username)
    if wait:
        raise HTTPException(429, f"Too many failed attempts. Try again in {wait} s.",
                            headers={"Retry-After": str(wait)})
    if not user_store.verify(username, req.current_password):
        login_throttle.record_failure(username)
        raise HTTPException(403, "The current password is not right.")
    try:
        user_store.set_password(username, req.new_password)
    except ValueError as e:
        raise HTTPException(422, str(e))
    logger.info("password changed for %s", username)
    return {"token": issue_token(username), "username": username}


@app.get("/api/account")
def account(request: Request):
    """Who is signed in, what they store and the limits that apply."""
    from app.services.auth import LOCAL_OWNER, _open_mode
    from app.services.limits import owner_usage
    username = getattr(request.state, "username", LOCAL_OWNER)
    return {
        "username": username,
        "has_password": not _open_mode() and user_store.exists(username),
        "usage": owner_usage(username) if username != "admin" else None,
        "max_file_mb": config.max_file_mb,
        "data_ttl_days": config.data_ttl_days,
        "support_email": config.support_email,
    }

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

if _mcp_mount is not None:
    app.mount("/mcp", _mcp_mount)

    @app.get("/api/mcp-reports/{name}")
    def mcp_report_download(name: str, exp: int, sig: str):
        """A PDF the generate_report tool wrote for a remote caller. The link
        is signed and expires after an hour, so it needs no login: it is
        opened in a browser, which has no Analytiq token."""
        from fastapi import HTTPException
        from fastapi.responses import FileResponse
        from app.mcp_server import verify_download
        path = verify_download(name, exp, sig)
        if path is None:
            raise HTTPException(404, "This report link is invalid or has expired.")
        return FileResponse(path, media_type="application/pdf", filename=name)


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
        "claude_configured": bool(config.anthropic_api_key),
        "claude_model": config.claude_model if config.anthropic_api_key else None,
        "gemini_configured": bool(config.gemini_api_key),
        "max_file_mb": config.max_file_mb,
        "data_ttl_days": config.data_ttl_days,
        # What will actually happen to a narrative request, so a client
        # running in privacy mode can confirm it from outside the app.
        "llm": _llm_status(),
    }


class CreateUserRequest(BaseModel):
    username: str
    password: str


@app.post("/api/admin/users")
def create_client_user(req: CreateUserRequest):
    """Onboard a new client account. Admin-key protected (see
    services/auth.py) — clients never see or use this endpoint."""
    from fastapi import HTTPException
    try:
        user = user_store.create(req.username, req.password)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"username": user.username, "created_at": user.created_at}


@app.get("/api/admin/users")
def list_client_users():
    from app.services.limits import owner_usage
    return {"users": [
        {"username": u.username, "created_at": u.created_at, "is_admin": u.is_admin,
         "usage": owner_usage(u.username)}
        for u in user_store.list()
    ]}


class ResetPasswordRequest(BaseModel):
    password: str


@app.post("/api/admin/users/{username}/password")
def reset_client_password(username: str, req: ResetPasswordRequest):
    """Set a new password for a client who has lost theirs. Their existing
    sessions are signed out."""
    from fastapi import HTTPException
    if not user_store.exists(username):
        raise HTTPException(404, "No such user")
    try:
        user_store.set_password(username, req.password)
    except ValueError as e:
        raise HTTPException(422, str(e))
    logger.info("password reset by admin for %s", username.strip().lower())
    return {"username": username.strip().lower(), "password_reset": True}


@app.get("/api/admin/status")
def admin_status():
    """What the server is doing and how much room it has."""
    import shutil
    from app.services.dataset_store import store as dataset_store
    from app.services.limits import job_gate
    rss_mb = None
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    rss_mb = int(line.split()[1]) // 1024
    except OSError:
        pass
    disk = shutil.disk_usage(config.data_dir)
    return {"memory_mb": rss_mb, "heavy_jobs": job_gate.stats(),
            "datasets_in_memory": dataset_store.memory_stats(),
            "disk_free_mb": disk.free // 2**20, "disk_total_mb": disk.total // 2**20,
            "clients": len(user_store.list()),
            "limits": {"max_file_mb": config.max_file_mb,
                       "max_storage_mb_per_client": config.max_storage_mb_per_owner,
                       "max_datasets_per_client": config.max_datasets_per_owner,
                       "data_ttl_days": config.data_ttl_days}}


@app.get("/api/admin/backup")
def admin_backup():
    """Everything the server stores (accounts, the signing key, every
    client's data) as one .tar.gz, to keep somewhere other than this
    server. Restore by unpacking it into DATA_DIR before starting."""
    import tarfile
    import tempfile
    from starlette.background import BackgroundTask
    fd, path = tempfile.mkstemp(suffix=".tar.gz", prefix="analytiq-backup-")
    os.close(fd)

    def _keep(ti):
        # Report downloads expire within the hour; not worth backing up.
        return None if ti.name.startswith("data/mcp_reports") else ti

    with tarfile.open(path, "w:gz", compresslevel=3) as tar:
        tar.add(config.data_dir, arcname="data", filter=_keep)
    name = "analytiq-backup-{}.tar.gz".format(time.strftime("%Y%m%d-%H%M"))
    return FileResponse(path, media_type="application/gzip", filename=name,
                        background=BackgroundTask(os.unlink, path))


@app.delete("/api/admin/users/{username}")
async def delete_client_user(username: str):
    """Offboard a client: removes their account AND cascades to delete
    every dataset and RAG knowledge base they own — a full data wipe,
    not just access revocation."""
    from fastapi import HTTPException
    from starlette.concurrency import run_in_threadpool
    from app.services.dataset_store import store as dataset_store
    # Owner directories are keyed by the stored (lower-case) name; cascading
    # with the name as typed found nothing, deleted the account anyway and
    # left all of that client's data on disk with no owner to remove it.
    username = username.strip().lower()
    if not user_store.exists(username):
        raise HTTPException(404, "No such user")

    def _cascade_delete():
        removed_ds = []
        for meta in dataset_store.list_meta(username):
            if dataset_store.delete(username, meta.dataset_id):
                removed_ds.append(meta.dataset_id)
        removed_kbs = []
        try:
            # The shared instance, not a fresh RagStore(): the API serves
            # knowledge bases from that instance's in-memory cache, so
            # deleting through a second instance left every one of them
            # readable — and re-savable to disk — until the next restart.
            from app.rag.vector_store import rag_store as rs
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
        target = _static_file(full_path)
        if target:
            return FileResponse(target)
        return FileResponse(os.path.join(_static_root, "index.html"))


_static_root = os.path.realpath(_static)


def _static_file(full_path: str) -> str | None:
    """The file under the static directory that `full_path` names, or None.

    The ASGI server percent-decodes the path before routing, so a request
    for `/%2Fetc%2Fpasswd` or `/..%2F..%2Fsrv%2Fdata%2F.secret_key` arrives
    here as an absolute or parent-relative path. `os.path.join` discards
    the static root for an absolute component and `..` walks out of it, so
    without a containment check this route served any file the process
    could read — the token-signing secret included — to anyone, with no
    authentication (the auth middleware only guards /api).
    """
    if not full_path:
        return None
    target = os.path.realpath(os.path.join(_static_root, full_path))
    if os.path.commonpath([target, _static_root]) != _static_root:
        return None
    return target if os.path.isfile(target) else None
