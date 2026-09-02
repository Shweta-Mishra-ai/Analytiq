"""
ai/settings_store.py — routing choices made in the app, not in the
environment.

Why a file and not just environment variables: on Render, changing an
environment variable restarts the service. That makes "try the reasoning
model on the executive summary and see if it reads better" a deploy,
which means nobody does it. Routing is exactly the kind of setting
people should be able to change, look at the result, and change back.

Precedence, lowest to highest — a task's own default, then LLM_ROUTING,
then this file. The environment stays the reproducible baseline for a
deployment; this is the layer on top that a person turns. Deleting the
file returns everything to the environment, which makes "put it back how
it was" a single action.

Validation happens *before* anything is written. A capability-violating
assignment is refused with the reason, so a broken routing file cannot
be created through the UI at all — the alternative is accepting it and
skipping it silently at the point of use, which looks exactly like the
model never being called.

Modelled on services/user_store.py: a lock, JSON, an atomic replace, and
the same reason for JSON over pickle — loading a pickle executes
whatever it contains, and this file is writable by whoever holds the
admin key.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Optional

from app.config import config

logger = logging.getLogger(__name__)

SETTINGS_FILE = "llm_routing.json"
SCHEMA_VERSION = 1


class RoutingRejected(ValueError):
    """An assignment that would not do what it claims. Carries the reason
    in the words of whoever has to fix it."""


class SettingsStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(config.data_dir, SETTINGS_FILE)
        self._lock = threading.RLock()
        self._state: dict = self._load()

    def _load(self) -> dict:
        if not os.path.exists(self.path):
            return {"version": SCHEMA_VERSION, "routing": {}}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            # The environment defaults are a working configuration, so a
            # corrupt override file costs the overrides and nothing else.
            # Refusing to start over a file the operator can delete would
            # be the worse failure.
            logger.error("could not read %s — routing falls back to the "
                         "environment", self.path, exc_info=True)
            return {"version": SCHEMA_VERSION, "routing": {}}
        routing = raw.get("routing") if isinstance(raw, dict) else None
        return {"version": raw.get("version", SCHEMA_VERSION),
                "routing": dict(routing or {}),
                "updated_at": raw.get("updated_at", 0.0),
                "updated_by": raw.get("updated_by", "")}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2, sort_keys=True)
        os.replace(tmp, self.path)

    # ── reading ──────────────────────────────────────────
    def routing(self) -> dict[str, str]:
        with self._lock:
            return dict(self._state.get("routing") or {})

    def as_dict(self) -> dict:
        with self._lock:
            return {
                "routing": dict(self._state.get("routing") or {}),
                "updated_at": self._state.get("updated_at", 0.0),
                "updated_by": self._state.get("updated_by", ""),
            }

    # ── writing ──────────────────────────────────────────
    def assign(self, task_name: str, model_id: str, actor: str = "") -> None:
        """Point a task at a model, after checking it can do the job.

        An empty model_id clears the override for that task, which is
        different from assigning nothing: it hands the task back to the
        environment, or to the task's own default.
        """
        from app.ai import providers, tasks
        from app.ai.capabilities import describe_gap
        from app.ai.model_catalogue import catalogue

        task = tasks.get(task_name)
        if task is None:
            raise RoutingRejected(
                f"'{task_name}' is not a task this app has. Known tasks: "
                f"{', '.join(t.name for t in tasks.all_tasks())}.")

        model_id = (model_id or "").strip()
        if not model_id:
            with self._lock:
                self._state.setdefault("routing", {}).pop(task.name, None)
                self._state["updated_at"] = time.time()
                self._state["updated_by"] = actor
                self._save()
            return

        spec = catalogue.resolve(model_id)
        if spec is None:
            raise RoutingRejected(
                f"'{model_id}' is not a model id. Use provider/model, for "
                f"example groq/llama-3.1-8b-instant.")

        provider = providers.get(spec.provider)
        if provider is None:
            raise RoutingRejected(
                f"'{spec.provider}' is not a provider this app knows. "
                f"Known providers: "
                f"{', '.join(p.name for p in providers.all_providers())}.")

        gap = describe_gap(task.requires, spec.capabilities)
        if gap:
            known = catalogue.get(spec.id) is not None
            extra = ("" if known else
                     " It is not in the catalogue, so it is assumed to write "
                     "text only — declare what else it can do first.")
            raise RoutingRejected(
                f"{spec.display} cannot be used for {task.label}: {gap}.{extra}")

        if task.min_context and spec.context and spec.context < task.min_context:
            raise RoutingRejected(
                f"{spec.display} holds {spec.context:,} tokens and "
                f"{task.label} needs about {task.min_context:,}.")

        # Deliberately NOT rejected: a provider with no key yet. Someone
        # setting up a deployment reasonably assigns the models first and
        # adds the keys after, and the System page already reports an
        # unconfigured provider clearly.
        if not provider.is_credentialed():
            logger.info("%s assigned to %s, whose provider has no "
                        "credentials yet", task.name, spec.id)

        with self._lock:
            self._state.setdefault("routing", {})[task.name] = spec.id
            self._state["updated_at"] = time.time()
            self._state["updated_by"] = actor
            self._save()

    def clear(self, actor: str = "") -> None:
        """Back to the environment. One action, because "put it back how
        it was" needs to be one action."""
        with self._lock:
            self._state["routing"] = {}
            self._state["updated_at"] = time.time()
            self._state["updated_by"] = actor
            self._save()


settings_store = SettingsStore()
