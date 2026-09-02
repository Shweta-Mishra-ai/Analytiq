"""
ai/routing.py — deciding which model does which job.

The rule this module exists to enforce, stated once:

    A model is only ever given work it is capable of doing.

That applies to the fallback chain as much as to the first choice, and
that is the part that matters. Two things go wrong when a model is
handed the wrong job. A text-only model given a photograph fails
outright — bad, but visible. A small chat model given a root-cause
question answers confidently and worse, and nobody notices for months.
Gating only the first pick prevents neither, because the fallback is
exactly where the wrong model gets in.

So resolution is: take the assigned model, check it can do the job,
then extend the chain with *other models that can also do the job* —
never with whatever else happens to be configured. An empty chain is a
legitimate answer, because every task here declares what it degrades
to, and for most of them the answer is that the analysis engine writes
its own wording, which was always the fallback anyway.

Precedence for the assignment, lowest to highest:

    the task's own default  →  LLM_ROUTING  →  the runtime override file

with privacy mode sitting above all of it: when no client data may
leave the machine, the chain is filtered to local models regardless of
what anyone assigned.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.ai import providers, tasks
from app.ai.capabilities import describe_gap, names
from app.ai.model_catalogue import ModelSpec, catalogue, split_id
from app.config import config

logger = logging.getLogger(__name__)


@dataclass
class RoutingProblem:
    """Something wrong with a routing assignment, phrased for whoever
    has to fix it, and surfaced rather than silently applied."""
    task: str
    model_id: str
    kind: str            # unknown_task | unknown_provider | incapable | small_context
    detail: str
    source: str = ""     # env | runtime

    def as_dict(self) -> dict:
        return {"task": self.task, "model_id": self.model_id,
                "kind": self.kind, "detail": self.detail,
                "source": self.source}


@dataclass
class Assignment:
    """What a task is set to use, and where that came from."""
    task: str
    model_id: str
    source: str          # default | env | runtime
    problem: Optional[RoutingProblem] = None


# ── parsing the environment ──────────────────────────────

def _parse_routing(raw: str) -> dict[str, str]:
    """LLM_ROUTING="task=target,task=target" → {task: target}.

    `target` is a model id (`groq/llama-3.1-8b-instant`) or — the older
    form, still supported — a bare provider name. A malformed entry is
    dropped with a warning rather than raising: one typo in one
    environment variable must not stop the service starting, and the
    System page reports the effective routing so the typo stays visible.
    """
    out: dict[str, str] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            logger.warning("LLM_ROUTING: ignoring %r (expected task=model)", part)
            continue
        task, target = part.split("=", 1)
        task, target = task.strip(), target.strip()
        if task and target:
            out[task] = target
    return out


def _model_id_for_target(target: str) -> str:
    """Turn a routing target into a model id.

    A bare provider name is the legacy form and means "that provider's
    configured default model" — which is exactly what the code did
    before models were addressable, so an existing LLM_ROUTING keeps
    its current behaviour rather than quietly changing it.
    """
    if "/" in target:
        return target
    provider = providers.get(target)
    if provider is None:
        return ""
    return f"{target}/{provider.model}" if provider.model else ""


# ── assignments ──────────────────────────────────────────

def _runtime_overrides() -> dict[str, str]:
    """Assignments saved from the System page. Empty until that store
    exists; kept as a seam so precedence is written down once."""
    try:
        from app.ai.settings_store import settings_store
    except ImportError:
        return {}
    return settings_store.routing()


def assignments() -> dict[str, Assignment]:
    """Every task, the model it is set to, and where that came from."""
    env = {tasks.resolve_name(k): v for k, v in
           _parse_routing(config.llm_routing).items()}
    runtime = {tasks.resolve_name(k): v for k, v in _runtime_overrides().items()}

    out: dict[str, Assignment] = {}
    for spec in tasks.all_tasks():
        target, source = spec.default_model, "default"
        if spec.name in env:
            target, source = env[spec.name], "env"
        if spec.name in runtime:
            target, source = runtime[spec.name], "runtime"

        model_id = _model_id_for_target(target) if target else ""
        problem = None
        if target and not model_id:
            problem = RoutingProblem(
                task=spec.name, model_id=target, kind="unknown_provider",
                source=source,
                detail=(f"'{target}' names no provider this app knows. "
                        f"Use one of: "
                        f"{', '.join(p.name for p in providers.all_providers())}, "
                        f"or a full model id like groq/llama-3.1-8b-instant."))
        out[spec.name] = Assignment(task=spec.name, model_id=model_id,
                                    source=source, problem=problem)
    return out


def deprecated_in_use() -> list[dict]:
    """Stale task names still present in LLM_ROUTING, and what they now
    control. Reported rather than silently honoured, because a name that
    works but means something else is worse than one that fails."""
    raw = _parse_routing(config.llm_routing)
    return [{"from": name, "to": tasks.DEPRECATED_ALIASES[name]}
            for name in raw if name in tasks.DEPRECATED_ALIASES]


# ── the resolver ─────────────────────────────────────────

_TIER_ORDER = {"fast": 0, "balanced": 1, "deep": 2}


def _ineligible_reason(spec: ModelSpec, task: tasks.TaskSpec) -> str:
    """Why this model cannot serve this task, or "" if it can.

    One function, three reasons, so a message never guesses. Getting
    this wrong is not cosmetic: told "its context window is too small"
    when the truth is "there is no API key", someone goes looking for a
    bigger model and never finds the problem.
    """
    provider = providers.get(spec.provider)
    if provider is None:
        return (f"'{spec.provider}' is not a provider this app knows")
    if not provider.is_credentialed():
        return (f"{provider.label} is not configured — "
                f"{provider.missing() or 'no credentials'}")
    gap = describe_gap(task.requires, spec.capabilities)
    if gap:
        return gap
    # A context window of 0 means "not recorded", not "zero" — filtering
    # those out would drop every model nobody has measured.
    if task.min_context and spec.context and spec.context < task.min_context:
        return (f"it holds {spec.context:,} tokens and this task needs "
                f"about {task.min_context:,}")
    return ""


def _eligible(spec: ModelSpec, task: tasks.TaskSpec) -> bool:
    return not _ineligible_reason(spec, task)


def candidates(task_name: str) -> list[ModelSpec]:
    """Every configured model that could serve this task, unordered."""
    task = tasks.get(task_name) or tasks.TASKS["default"]
    return [s for s in catalogue.all() if _eligible(s, task)]


def resolve_models(task_name: str, force: str = "") -> list[ModelSpec]:
    """The models this call will try, in order.

    Returning an empty list is a real answer, not an error: the task's
    `degrades_to` says what ships instead, and for most tasks that is
    the analysis engine's own wording.
    """
    from app.ai import local_llm

    task = tasks.get(task_name) or tasks.TASKS["default"]
    assigned = assignments().get(task.name)

    # An opt-in task does nothing until someone assigns a model to it.
    # Without this, the fallback fill below would find a capable model
    # and quietly turn the task on — which for generated cover artwork
    # would mean a picture appearing in a client's deliverable that
    # nobody asked for.
    if (task.opt_in and not force
            and not (assigned and assigned.model_id)):
        return []

    chain: list[ModelSpec] = []
    seen: set[str] = set()

    def _add(spec: Optional[ModelSpec]) -> None:
        if spec is None or spec.id in seen:
            return
        seen.add(spec.id)
        chain.append(spec)

    # 1. Whatever was explicitly asked for on this call.
    if force:
        forced_id = _model_id_for_target(force)
        forced = catalogue.resolve(forced_id) if forced_id else None
        if forced is not None and _eligible(forced, task):
            _add(forced)
        elif forced is not None:
            logger.warning("forced model %s cannot serve %s: %s", forced.id,
                           task.name, _ineligible_reason(forced, task))

    # 2. The assignment — but only if it can actually do the job. A
    #    misconfiguration must not silently downgrade the work; it is
    #    skipped loudly here and reported by problems().
    #
    #    A task that prefers local models yields this slot when the
    #    assignment is a cloud model and a local one can serve: the
    #    knowledge base's privacy posture is not something a default
    #    assignment should quietly undo. An explicit `force` still wins,
    #    because that is someone asking for this call specifically.
    if (task.prefers_local and assigned and assigned.source == "default"
            and _local_candidates(task)):
        assigned = None
    if assigned and assigned.model_id:
        spec = catalogue.resolve(assigned.model_id)
        if spec is not None and _eligible(spec, task):
            _add(spec)
        elif spec is not None:
            # Debug, not warning: the System page polls status(), which
            # resolves every task, so a warning here becomes a wall of
            # identical lines on every page load. problems() is the
            # surface that actually tells someone — and it tells them in
            # the UI, where they can act on it.
            logger.debug(
                "%s is assigned to %s but %s — looking for a capable model "
                "instead", task.name, spec.id, _ineligible_reason(spec, task))

    # 3. Everything else capable, cheapest-first for a task that wants
    #    speed and deepest-first for one that wants an argument held
    #    together, then by the provider order the deployment set.
    order = {name: i for i, name in enumerate(_provider_order())}
    rest = [s for s in candidates(task.name) if s.id not in seen]
    reverse = task.prefers == "deep"

    def _local_first(spec: ModelSpec) -> int:
        if not task.prefers_local:
            return 0
        provider = providers.get(spec.provider)
        return 0 if (provider is not None and provider.local) else 1

    rest.sort(key=lambda s: (
        _local_first(s),
        -_TIER_ORDER.get(s.tier, 1) if reverse else _TIER_ORDER.get(s.tier, 1),
        order.get(s.provider, len(order)),
        s.id))
    for spec in rest:
        _add(spec)

    # 4. Privacy mode is not a preference. No client data may reach a
    #    third party, so the chain is filtered rather than reordered.
    if local_llm.privacy_mode():
        chain = [s for s in chain
                 if (providers.get(s.provider) or _NullProvider()).local]

    return chain


def _local_candidates(task: tasks.TaskSpec) -> list[ModelSpec]:
    return [s for s in candidates(task.name)
            if (providers.get(s.provider) or _NullProvider()).local]


class _NullProvider:
    local = False


def _provider_order() -> list[str]:
    return [n.strip() for n in (config.llm_provider_order or "").split(",")
            if n.strip()]


def problems() -> list[dict]:
    """Every assignment that will not do what it looks like it does."""
    found: list[RoutingProblem] = []
    for name, assigned in assignments().items():
        task = tasks.TASKS[name]
        if assigned.problem is not None:
            found.append(assigned.problem)
            continue
        if not assigned.model_id:
            continue
        spec = catalogue.resolve(assigned.model_id)
        if spec is None:
            continue
        reason = _ineligible_reason(spec, task)
        if not reason:
            continue
        kind = ("not_configured" if "not configured" in reason
                else "small_context" if "tokens" in reason
                else "incapable")
        hint = ""
        if kind == "incapable" and _is_unknown(spec):
            hint = (" This model is not in the catalogue, so it is assumed "
                    "to write text only — say what else it can do to use it "
                    "here.")
        found.append(RoutingProblem(
            task=name, model_id=spec.id, kind=kind, source=assigned.source,
            detail=f"{spec.display} is assigned to {task.label}, but "
                   f"{reason}.{hint}"))
    return [p.as_dict() for p in found]


def _is_unknown(spec: ModelSpec) -> bool:
    return catalogue.get(spec.id) is None


def status() -> dict:
    """Everything the System page needs to render the routing table."""
    assigned = assignments()
    rows = []
    for task in tasks.all_tasks():
        chain = resolve_models(task.name)
        a = assigned.get(task.name)
        rows.append({
            "task": task.name,
            "label": task.label,
            "description": task.description,
            "requires": names(task.requires),
            "min_context": task.min_context,
            "degrades_to": task.degrades_to,
            "assigned": a.model_id if a else "",
            "source": a.source if a else "default",
            "resolved": [s.id for s in chain],
            "eligible": [s.id for s in candidates(task.name)],
            "served": bool(chain),
        })
    return {
        "tasks": rows,
        "models": [s.as_dict() for s in catalogue.all()],
        "problems": problems(),
        "deprecated": deprecated_in_use(),
        "unserved": [r["task"] for r in rows if not r["served"] and r["assigned"]],
    }
