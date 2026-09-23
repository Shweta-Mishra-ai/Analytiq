"""
api/admin.py — the operator's routes.

Everything under /api/admin, which the auth middleware gates as a whole
prefix: metrics and storage, the model routing editor, the LLM
self-check, client accounts, and the storage sweep. They lived in
main.py, where thirteen of its sixteen routes were these — so the file
that wires the application together was mostly one feature.

They are grouped here because they share an audience and a failure mode
rather than a subject: each one exists so the person running a
deployment can find out what it is doing, on the machine that is doing
it. A failure message here may name a column from a client's dataset,
which is why the whole prefix is behind the admin gate.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import config
from app.services.cleanup import sweep_expired
from app.services.user_store import user_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/metrics")
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


@router.get("/llm-status")
async def llm_status():
    """What this deployment is configured to use, without calling
    anything. Fast, safe to poll, and never reveals a key — only whether
    one is present and, when it is not, the exact variable name to set."""
    from app.ai.llm_client import get_client
    return get_client().status()


@router.get("/storage")
async def storage_status():
    """Where a client's data is kept, and whether it survives a deploy.

    An operator cannot tell this from the outside: everything works
    perfectly on a container that has not restarted yet. Naming it
    plainly is the difference between finding out here and finding out
    when a client opens a share link, or logs in and their uploads are
    gone.
    """
    from app.services.artifacts import store as artifacts
    from app.services.dataset_store import store as datasets

    blobs = artifacts.blobs
    reports_durable = bool(getattr(blobs, "durable", False))
    data_durable = datasets.blobs is not None

    if data_durable:
        data_detail = (
            "Uploads are written to object storage as well as this "
            "container's disk, so a restart, a redeploy or a second "
            "instance can still serve them.")
    else:
        data_detail = (
            "Uploads live only on this container's disk. They survive a "
            "request but not a deploy, and a second instance has a "
            "different disk — the same account sees its data on one "
            "request and not the next. Set S3_BUCKET, or mount a "
            "persistent volume and run a single instance.")

    return {
        "artifacts": blobs.describe(),
        "durable": reports_durable,
        "detail": (
            "Reports are kept in object storage and survive a restart."
            if reports_durable else
            "Reports are on this container's disk. They survive a request "
            "but not a deploy — set S3_BUCKET to keep them."
        ),
        "datasets": (datasets.blobs.describe() if data_durable
                     else "local directory {}".format(datasets.base_dir)),
        "datasets_durable": data_durable,
        "datasets_detail": data_detail,
    }


@router.post("/llm-check")
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


@router.get("/routing")
async def get_routing():
    """Which model does which job, what each job needs, and which models
    could serve it. Everything the System page's routing table renders."""
    return _routing_payload()


@router.post("/routing")
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


@router.delete("/routing")
async def clear_routing():
    """Back to whatever the environment says."""
    from app.ai.settings_store import settings_store
    settings_store.clear()
    return _routing_payload()


@router.post("/models")
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


@router.delete("/models/{model_id:path}")
async def forget_model(model_id: str):
    from fastapi import HTTPException
    from app.ai.model_catalogue import catalogue
    if not catalogue.forget(model_id):
        raise HTTPException(
            404, f"'{model_id}' was not added here. Built-in catalogue "
                 f"entries cannot be removed — declare the same id to "
                 f"change what it claims.")
    return _routing_payload()


@router.post("/task-check")
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


@router.post("/users")
async def create_client_user(req: CreateUserRequest):
    """Onboard a new client account. Admin-key protected (see
    services/auth.py) — clients never see or use this endpoint."""
    from fastapi import HTTPException
    try:
        user = user_store.create(req.username, req.password)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"username": user.username, "created_at": user.created_at}


@router.get("/users")
async def list_client_users():
    return {"users": [
        {"username": u.username, "created_at": u.created_at, "is_admin": u.is_admin}
        for u in user_store.list()
    ]}


@router.delete("/users/{username}")
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


@router.post("/cleanup")
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
