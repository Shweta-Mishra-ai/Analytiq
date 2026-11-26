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
from app.api import advanced_analytics, analytics, billing, charts, chat, datasets, ml, reports
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
    from app.services.auth import setup_required
    if setup_required():
        logger.error("APP_ADMIN_KEY is not set and there are no accounts: every API "
                     "request is refused until it is set (REQUIRE_AUTH is on).")
    task = asyncio.create_task(cleanup_loop())
    from app.services.backups import backup_loop
    backups = asyncio.create_task(backup_loop())
    try:
        if _mcp_mount is not None:
            async with _mcp_mount.running():
                yield
        else:
            yield
    finally:
        for t in (task, backups):
            t.cancel()
            try:
                await t
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
        # checkout.razorpay.com: Razorpay's payment window (cards, UPI),
        # which loads its script and opens in a frame on api.razorpay.com.
        # challenges.cloudflare.com: the Turnstile robot check on sign-up.
        "default-src 'self'; script-src 'self' https://checkout.razorpay.com "
        "https://challenges.cloudflare.com; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob: https://*.razorpay.com; font-src 'self' data:; "
        "connect-src 'self' https://*.razorpay.com; "
        "frame-src https://api.razorpay.com https://checkout.razorpay.com "
        "https://challenges.cloudflare.com; "
        "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"),
}


def _caller_key(request: Request) -> str:
    """Who a request is counted against: the token when there is one, else
    the address.

    The address is the connecting one, or, when that is a proxy on a private
    network (Render, Railway, nginx), the last X-Forwarded-For entry: the
    one the proxy appended. The first entry was used, and the caller writes
    that one, so a new invented address on every request was never limited.
    """
    auth = request.headers.get("authorization", "")
    if auth:
        import hashlib
        return "t:" + hashlib.sha256(auth.encode("utf-8", "replace")).hexdigest()[:24]
    from app.services.limits import client_ip
    return "ip:" + client_ip(request)


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
        except Exception as e:
            user = getattr(request.state, "username", "-")
            logger.exception("unhandled error ref=%s %s %s user=%s", rid, request.method,
                             path, user)
            from app.services.alerts import record_error
            record_error(rid, request.method, path, user, e)
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
    from app.services.auth import setup_required
    from app.services.tokens import issue_token
    if setup_required():
        raise HTTPException(
            503, "This server is not set up yet: its administrator must set "
                 "APP_ADMIN_KEY and create your account.")
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



class SignupRequest(BaseModel):
    username: str
    password: str
    email: str = ""
    captcha: str = ""


class _SignupLimiter:
    """At most SIGNUPS_PER_HOUR new accounts from one address an hour. Each
    account comes with free daily credits, so unlimited sign-ups were
    unlimited free reports."""

    def __init__(self, per_hour: int = 5):
        self.per_hour = per_hour
        self._seen: dict[str, list[float]] = {}
        import threading
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            recent = [t for t in self._seen.get(key, []) if now - t < 3600]
            if len(self._seen) > 10_000:
                self._seen = {k: v for k, v in self._seen.items()
                              if v and now - v[-1] < 3600}
            if len(recent) >= self.per_hour:
                self._seen[key] = recent
                return False
            recent.append(now)
            self._seen[key] = recent
            return True


signup_limiter = _SignupLimiter()


@app.post("/api/auth/signup")
def signup(req: SignupRequest, request: Request):
    """Create an account and sign in. Every account gets the free daily
    credits; nothing else is shared between accounts."""
    from fastapi import HTTPException
    from app.services.auth import setup_required
    from app.services.tokens import issue_token
    if not config.signup_enabled:
        raise HTTPException(403, "New accounts are created by the administrator. "
                                 "Please contact support.")
    if setup_required():
        raise HTTPException(503, "This server is not set up yet: its administrator "
                                 "must set APP_ADMIN_KEY.")
    from app.services import emails
    verify = emails.verification_required()
    email = req.email.strip()
    if email or verify:
        email = emails.check_address(email)
    from app.services import captcha
    from app.services.limits import client_ip, network_of
    ip = client_ip(request)
    captcha.check(req.captcha, ip)
    if not signup_limiter.allow(_caller_key(request)):
        raise HTTPException(429, "Too many new accounts from this address. "
                                 "Try again in an hour.", headers={"Retry-After": "3600"})
    try:
        user = user_store.create(req.username, req.password, email=email,
                                 email_verified=False if verify else None)
    except ValueError as e:
        raise HTTPException(422, str(e))
    from app.services.credits import credit_store
    credit_store.set_network(user.username, network_of(ip))
    logger.info("account created by sign-up: %s", user.username)
    sent = False
    if verify:
        try:
            _send_verification(request, user.username, email)
            sent = True
        except HTTPException:
            pass   # the account stands; the page offers to send it again
    return {"token": issue_token(user.username), "username": user.username,
            "is_admin": False, "verification_sent": sent}


def _public_base(request: Request) -> str:
    if config.public_url:
        return config.public_url.rstrip("/")
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    return f"{proto}://{host}"


def _send_verification(request: Request, username: str, email: str) -> None:
    from app.services import emails
    from app.services.tokens import issue_purpose_token
    tok = issue_purpose_token("verify", username, 48 * 3600, e=email)
    emails.send_verification(email, username, f"{_public_base(request)}/verify-email?token={tok}")


@app.post("/api/auth/resend-verification")
def resend_verification(request: Request):
    from fastapi import HTTPException
    from app.services.auth import LOCAL_OWNER
    username = getattr(request.state, "username", LOCAL_OWNER)
    user = user_store.get(username)
    if user is None or not user.email:
        raise HTTPException(400, "This account has no email address to confirm.")
    if user.email_verified:
        return {"already": True}
    if not signup_limiter.allow("verify:" + username):
        raise HTTPException(429, "Several emails were already sent; check your inbox and "
                                 "spam folder, or try again in an hour.")
    _send_verification(request, username, user.email)
    return {"sent": True}


class TokenRequest(BaseModel):
    token: str


@app.post("/api/auth/verify-email")
def verify_email(req: TokenRequest):
    from fastapi import HTTPException
    from app.services.tokens import verify_purpose_token
    claims = verify_purpose_token(req.token, "verify")
    user = user_store.get(claims["u"]) if claims else None
    # The link names the address it was sent to: after a change of address
    # an old link does not confirm the new one.
    if user is None or claims.get("e") != user.email:
        raise HTTPException(400, "This link is invalid or has expired. Sign in and ask "
                                 "for a new one.")
    user_store.set_email_verified(user.username)
    logger.info("email confirmed for %s", user.username)
    return {"verified": True, "username": user.username}


class ForgotRequest(BaseModel):
    login: str


_FORGOT_REPLY = ("If an account with that username or email exists and has an email "
                 "address, a link to reset the password is on its way.")


@app.post("/api/auth/forgot-password")
def forgot_password(req: ForgotRequest, request: Request):
    """Email a reset link. The answer is the same whether or not the
    account exists, so it cannot be used to find out who has one."""
    from fastapi import HTTPException
    from app.services import emails
    from app.services.tokens import issue_purpose_token
    if not emails.enabled():
        raise HTTPException(503, "Password reset by email is not available here; "
                                 "please contact support.")
    if not signup_limiter.allow("forgot:" + _caller_key(request)):
        raise HTTPException(429, "Too many requests. Try again in an hour.")
    login = req.login.strip()
    user = user_store.find_by_email(login) if "@" in login else user_store.get(login)
    if user is not None and user.email and signup_limiter.allow("forgot-user:" + user.username):
        tok = issue_purpose_token("reset", user.username, 3600,
                                  pc=user_store.password_changed_at(user.username))
        try:
            emails.send_password_reset(user.email, user.username,
                                       f"{_public_base(request)}/reset-password?token={tok}")
        except HTTPException:
            logger.warning("password reset email for %s not sent", user.username)
    return {"message": _FORGOT_REPLY}


class ResetRequest(BaseModel):
    token: str
    password: str


@app.post("/api/auth/reset-password")
def reset_password(req: ResetRequest):
    from fastapi import HTTPException
    from app.services.tokens import issue_token, verify_purpose_token
    claims = verify_purpose_token(req.token, "reset")
    # Bound to the password it replaces: once used (or the password changed
    # any other way) the link stops working.
    if (claims is None or not user_store.exists(claims["u"])
            or claims.get("pc") != user_store.password_changed_at(claims["u"])):
        raise HTTPException(400, "This link is invalid, expired or already used. Ask "
                                 "for a new one.")
    try:
        user_store.set_password(claims["u"], req.password)
    except ValueError as e:
        raise HTTPException(422, str(e))
    # Opening the emailed link proves the mailbox too.
    if not user_store.get(claims["u"]).email_verified:
        user_store.set_email_verified(claims["u"])
    logger.info("password reset by email for %s", claims["u"])
    return {"token": issue_token(claims["u"]), "username": claims["u"]}


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
        "connector": _connector_info(request, username),
        "credits": _credits(request, username),
        "email": _user_email(username),
    }


def _user_email(username: str):
    from app.services import emails
    user = user_store.get(username)
    if user is None:
        return None
    return {"address": user.email, "verified": user.email_verified,
            "can_send": emails.enabled()}


def _credits(request: Request, username: str) -> dict:
    from app.services.credits import credit_store
    return credit_store.balance(username, bool(getattr(request.state, "is_admin", False)))


def _connector_info(request: Request, username: str):
    """Where Claude connects, and whether this client has a key for it.
    None without accounts (single-user mode serves only this machine)."""
    from app.services.auth import _open_mode
    user = None if _open_mode() else user_store.get(username)
    if _mcp_mount is None or user is None:
        return None
    base = config.public_url.rstrip("/")
    if not base:
        proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
        base = f"{proto}://{host}"
    return {"url": base + "/mcp/", "has_key": user.connector_key_at > 0,
            "key_created_at": user.connector_key_at or None}


@app.post("/api/account/connector-key")
def create_connector_key(request: Request):
    """A new key for this client's Claude connector, shown this once. Any
    earlier key stops working."""
    from fastapi import HTTPException
    from app.services.auth import LOCAL_OWNER, _open_mode
    username = getattr(request.state, "username", LOCAL_OWNER)
    if _open_mode() or not user_store.exists(username):
        raise HTTPException(400, "Connector keys need a client account.")
    key = user_store.new_connector_key(username)
    logger.info("connector key created for %s", username)
    return {"key": key, "connector": _connector_info(request, username)}


@app.delete("/api/account/connector-key")
def delete_connector_key(request: Request):
    from app.services.auth import LOCAL_OWNER
    username = getattr(request.state, "username", LOCAL_OWNER)
    removed = user_store.revoke_connector_key(username)
    if removed:
        logger.info("connector key removed for %s", username)
    return {"removed": removed}

app.include_router(datasets.router)
app.include_router(analytics.router)
app.include_router(advanced_analytics.router)
app.include_router(charts.router)
app.include_router(ml.router)
app.include_router(chat.router)
app.include_router(reports.router)
app.include_router(billing.router)
app.include_router(billing.admin_router)

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


def _chat_available() -> bool:
    from app.api.chat import chat_available
    return chat_available()


@app.get("/api/health")
async def health():
    from app.ai.local_llm import status as _llm_status
    from app.services.auth import _open_mode, setup_required
    return {
        "status": "ok",
        "setup_required": setup_required(),
        "app": config.app_name,
        "version": config.app_version,
        "auth_required": not _open_mode(),
        "groq_configured": bool(config.groq_api_key),
        "claude_configured": bool(config.anthropic_api_key),
        "claude_model": config.claude_model if config.anthropic_api_key else None,
        "gemini_configured": bool(config.gemini_api_key),
        "max_file_mb": config.max_file_mb,
        "data_ttl_days": config.data_ttl_days,
        "chat_available": _chat_available(),
        "signup_enabled": config.signup_enabled,
        "turnstile_site_key": config.turnstile_site_key if config.turnstile_secret else "",
        "email_enabled": bool(config.smtp_host),
        "email_required": bool(config.smtp_host and config.require_email_verification),
        "business": {"name": config.business_name, "address": config.business_address,
                     "phone": config.business_phone, "email": config.support_email},
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
    from app.services.credits import credit_store
    return {"users": [
        {"username": u.username, "created_at": u.created_at, "is_admin": u.is_admin,
         "email": u.email, "usage": owner_usage(u.username),
         "credits": credit_store.balance(u.username, u.is_admin)}
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
    from app.services.alerts import recent_errors
    from app.services.backups import list_backups
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
            "recent_errors": recent_errors()[:20],
            "alerts_configured": bool(config.alert_webhook_url),
            "backups": list_backups(),
            "backup_every_hours": config.backup_every_hours,
            "limits": {"max_file_mb": config.max_file_mb,
                       "max_storage_mb_per_client": config.max_storage_mb_per_owner,
                       "max_datasets_per_client": config.max_datasets_per_owner,
                       "data_ttl_days": config.data_ttl_days}}


@app.get("/api/admin/backup")
def admin_backup():
    """Everything the server stores (accounts, the signing key, every
    client's data) as one .tar.gz, to keep somewhere other than this
    server. Restore by unpacking it into DATA_DIR before starting."""
    import tempfile
    from starlette.background import BackgroundTask
    from app.services.backups import write_archive
    fd, path = tempfile.mkstemp(suffix=".tar.gz", prefix="analytiq-backup-")
    os.close(fd)
    write_archive(path)
    name = "analytiq-backup-{}.tar.gz".format(time.strftime("%Y%m%d-%H%M"))
    return FileResponse(path, media_type="application/gzip", filename=name,
                        background=BackgroundTask(os.unlink, path))


@app.post("/api/admin/backups")
def admin_take_backup():
    """Take a snapshot now, kept on the data disk with the scheduled ones."""
    from fastapi import HTTPException
    from app.services.backups import take_backup
    made = take_backup()
    if made is None:
        raise HTTPException(507, "Not enough free disk space for a snapshot.")
    return made


@app.get("/api/admin/backups/{name}")
def admin_get_backup(name: str):
    from fastapi import HTTPException
    from app.services.backups import backup_path
    path = backup_path(name)
    if path is None:
        raise HTTPException(404, "No such backup")
    return FileResponse(path, media_type="application/gzip", filename=name)


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
    from app.services.credits import credit_store
    credit_store.forget(username)
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
