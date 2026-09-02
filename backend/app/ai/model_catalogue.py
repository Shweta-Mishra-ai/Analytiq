"""
ai/model_catalogue.py — which models exist, and what each can do.

The catalogue's job is to answer one question: *given a task that needs
these capabilities, which models could serve it?* Everything else here
exists to keep that answer honest as the world changes underneath it.

Two design decisions do most of the work:

**A catalogue you must edit to use a new model is a catalogue that is
wrong within a month.** Providers add and retire models constantly, and
OpenRouter alone serves hundreds. So the seed list below is a
convenience, not a gate: naming a model it has never heard of is
allowed, and always has been the common case.

**An unknown model is assumed to do the least.** It gets TEXT and
nothing else. Not because that is likely true — plenty of unknown
models read images — but because the alternative is guessing a
capability the model does not have and discovering it when a client's
report comes back broken. Widening an unknown model's capabilities is
an explicit act by whoever knows: a checkbox in the UI, or a suffix in
the environment variable. The claim is theirs, and it is recorded as
theirs, which is also what makes it correctable when it turns out to be
wrong.

The seed entries carry the same caveat. They are this author's reading
of what each model does today, they will age, and `DATA_DIR/models.json`
overrides any of them without a deploy.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Optional

from app.ai.capabilities import Capability as C
from app.ai.capabilities import TEXT_ONLY, names, parse
from app.config import config

logger = logging.getLogger(__name__)

CATALOGUE_FILE = "models.json"

#: Ordering hint, not a promise about quality. `fast` is what you want
#: twenty times per report; `deep` is what you want once, for the
#: paragraph a client will actually read out loud.
TIERS = ("fast", "balanced", "deep")


@dataclass(frozen=True)
class ModelSpec:
    """One model, addressed as `provider/model`."""

    provider: str
    model: str
    label: str = ""
    capabilities: frozenset = field(default_factory=lambda: TEXT_ONLY)
    tier: str = "balanced"
    context: int = 0
    free: bool = False
    notes: str = ""
    #: True when the capabilities were asserted by an operator rather
    #: than shipped in the seed list. Surfaced in the UI so a wrong
    #: claim is traceable to a decision instead of looking like a fact.
    declared: bool = False
    #: Capabilities a real call has actually demonstrated, as
    #: {capability: {"ok": bool, "at": float, "error": str}}. Never
    #: written by hand — only by a probe. The distinction matters:
    #: everything above is a *claim* about a model, and this is the
    #: only field that is evidence. The UI shows the two differently
    #: for exactly that reason.
    verified: dict = field(default_factory=dict, compare=False)

    @property
    def id(self) -> str:
        return f"{self.provider}/{self.model}"

    @property
    def display(self) -> str:
        return self.label or self.model

    def can(self, required) -> bool:
        return set(required) <= set(self.capabilities)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "provider": self.provider,
            "model": self.model,
            "label": self.display,
            "capabilities": names(self.capabilities),
            "tier": self.tier,
            "context": self.context,
            "free": self.free,
            "notes": self.notes,
            "declared": self.declared,
            "verified": {k: v for k, v in (self.verified or {}).items()},
        }


def _spec(provider: str, model: str, label: str, caps: set, tier: str = "balanced",
          context: int = 0, free: bool = False, notes: str = "") -> ModelSpec:
    return ModelSpec(provider=provider, model=model, label=label,
                     capabilities=frozenset(caps), tier=tier,
                     context=context, free=free, notes=notes)


# ── the seed ─────────────────────────────────────────────
# Small on purpose. Enough that a fresh install routes sensibly without
# anyone typing a model name, and no more — a long list here is a long
# list to keep correct.

SEED: list[ModelSpec] = [
    # Groq — free tier, openly-licensed weights, very fast. This is the
    # right home for the high-volume short calls.
    _spec("groq", "llama-3.3-70b-versatile", "Llama 3.3 70B",
          {C.TEXT, C.JSON, C.REASONING}, tier="balanced", context=128_000,
          free=True, notes="Good general default; fast enough for bulk work."),
    _spec("groq", "llama-3.1-8b-instant", "Llama 3.1 8B Instant",
          {C.TEXT, C.JSON}, tier="fast", context=128_000, free=True,
          notes="Cheapest and fastest. Fine for one-sentence chart captions."),

    # Google — the only provider here with native video understanding,
    # and the only one currently wired for embeddings.
    _spec("gemini", "gemini-3.6-flash", "Gemini 3.6 Flash",
          {C.TEXT, C.JSON, C.REASONING, C.VISION, C.VIDEO}, tier="deep",
          context=1_000_000, notes="Reads images and video; long context."),
    _spec("gemini", "gemini-embedding-001", "Gemini Embedding 001",
          {C.EMBEDDING}, tier="balanced",
          notes="768-dimension vectors for the knowledge base."),

    # OpenRouter — one key, many models. The `:free` suffix selects the
    # no-cost pool; drop it and the same slug bills.
    _spec("openrouter", "meta-llama/llama-3.3-70b-instruct:free",
          "Llama 3.3 70B (free)", {C.TEXT, C.JSON, C.REASONING},
          tier="balanced", context=128_000, free=True),
    _spec("openrouter", "deepseek/deepseek-r1:free", "DeepSeek R1 (free)",
          {C.TEXT, C.REASONING}, tier="deep", context=64_000, free=True,
          notes="Reasoning-tuned. Slower; worth it for a summary, not a caption."),

    # Cerebras — free tier, openly-licensed weights, extremely fast.
    _spec("cerebras", "llama-3.3-70b", "Llama 3.3 70B (Cerebras)",
          {C.TEXT, C.JSON, C.REASONING}, tier="fast", context=128_000, free=True),

    # Together — small free credit, then paid.
    _spec("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
          "Llama 3.3 70B Turbo (free)", {C.TEXT, C.JSON, C.REASONING},
          tier="balanced", context=128_000, free=True),

    # Local — the client's own hardware. No key, no cost, no data
    # leaving the machine, and the only provider permitted in privacy
    # mode. Gemma 3 is multimodal, which makes a fully local vision
    # path possible.
    # Reasoning is claimed here deliberately. It is the smallest model
    # in this list to claim it, and a 70B would answer better — but
    # without it the knowledge base's local-first posture is
    # unreachable, which would mean the privacy path exists only on
    # paper. A 12B instruct model with this much context does grounded
    # answering over retrieved passages perfectly respectably, and
    # anyone who disagrees can point the task at something larger.
    _spec("local", "gemma3:12b", "Gemma 3 12B (local)",
          {C.TEXT, C.JSON, C.VISION, C.REASONING}, tier="balanced",
          context=128_000, free=True,
          notes="Runs on a laptop, reads images, and keeps everything on "
                "your own hardware."),
    # Image generation, for the report cover only — see ai/imagery.py for
    # why the scope is that narrow. Neither is assigned by default.
    _spec("gemini", "gemini-3-pro-image", "Gemini 3 Pro Image",
          {C.IMAGE_GEN}, tier="balanced",
          notes="Cover artwork only. Never used for a chart or an exhibit."),
    _spec("local", "sdxl", "Stable Diffusion XL (local)",
          {C.IMAGE_GEN}, tier="balanced", free=True,
          notes="Any OpenAI-compatible /v1/images/generations server — "
                "AUTOMATIC1111, LocalAI, SD.Next, ComfyUI. Openly licensed "
                "and nothing leaves the machine."),

    _spec("local", "nomic-embed-text", "Nomic Embed Text (local)",
          {C.EMBEDDING}, tier="balanced", free=True,
          notes="Local embeddings; keeps the knowledge base on-premises."),
]


def split_id(model_id: str) -> tuple[str, str]:
    """`provider/model` → (provider, model).

    Model names contain slashes of their own
    (`meta-llama/llama-3.3-70b-instruct:free`), so only the first
    separator is structural.
    """
    text = (model_id or "").strip()
    if "/" not in text:
        return text, ""
    provider, model = text.split("/", 1)
    return provider.strip(), model.strip()


class ModelCatalogue:
    """Seed entries plus whatever this deployment added, on disk.

    Mirrors services/user_store.py: a lock, a JSON file, an atomic
    replace. Nothing here is secret, so unlike the account file it is
    readable and editable by hand.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(config.data_dir, CATALOGUE_FILE)
        self._lock = threading.RLock()
        self._custom: dict[str, ModelSpec] = self._load()

    # ── persistence ──────────────────────────────────────
    def _load(self) -> dict[str, ModelSpec]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            # Unlike the account file, an unreadable catalogue is not
            # dangerous — it falls back to the seed, which is a working
            # configuration. Log it and carry on rather than refusing
            # to start over a file the operator can simply delete.
            logger.error("could not read the model catalogue at %s — using the "
                         "built-in list", self.path, exc_info=True)
            return {}
        out = {}
        for model_id, rec in (raw or {}).items():
            provider, model = split_id(model_id)
            if not provider or not model:
                continue
            out[model_id] = ModelSpec(
                provider=provider, model=model,
                label=rec.get("label", ""),
                capabilities=parse(rec.get("capabilities")) or TEXT_ONLY,
                tier=rec.get("tier", "balanced"),
                context=int(rec.get("context") or 0),
                free=bool(rec.get("free", False)),
                notes=rec.get("notes", ""),
                declared=bool(rec.get("declared", True)),
                verified=dict(rec.get("verified") or {}),
            )
        return out

    def _save(self) -> None:
        payload = {spec.id: {
            "label": spec.label,
            "capabilities": names(spec.capabilities),
            "tier": spec.tier,
            "context": spec.context,
            "free": spec.free,
            "notes": spec.notes,
            "declared": spec.declared,
            "verified": spec.verified,
        } for spec in self._custom.values()}
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, self.path)

    # ── reading ──────────────────────────────────────────
    def all(self) -> list[ModelSpec]:
        """Seed first, then additions; an addition with the same id wins."""
        with self._lock:
            merged = {spec.id: spec for spec in SEED}
            merged.update(self._custom)
        return sorted(merged.values(), key=lambda s: (s.provider, s.model))

    def get(self, model_id: str) -> Optional[ModelSpec]:
        """A known model, or None. Use `resolve` to accept unknowns."""
        target = (model_id or "").strip()
        for spec in self.all():
            if spec.id == target:
                return spec
        return None

    def resolve(self, model_id: str) -> Optional[ModelSpec]:
        """A spec for any well-formed `provider/model`, known or not.

        The unknown case is the point of this method. It returns a
        TEXT-only spec, so the model can write prose and can never be
        picked for vision, embeddings or JSON-critical work without
        someone saying explicitly that it can.
        """
        known = self.get(model_id)
        if known is not None:
            return known
        provider, model = split_id(model_id)
        if not provider or not model:
            return None
        return ModelSpec(
            provider=provider, model=model, label=model,
            capabilities=TEXT_ONLY, tier="balanced", declared=False,
            notes="Not in the catalogue — assumed to write text only. "
                  "Declare what else it can do to use it for other work.")

    def with_capabilities(self, required) -> list[ModelSpec]:
        """Every model that could serve a task needing `required`."""
        return [spec for spec in self.all() if spec.can(required)]

    # ── writing ──────────────────────────────────────────
    def declare(self, model_id: str, capabilities, label: str = "",
                tier: str = "balanced", context: int = 0, free: bool = False,
                notes: str = "") -> ModelSpec:
        """Record what an operator says a model can do.

        Raises ValueError on a malformed id or an empty capability set —
        a model that can do nothing is a row that will confuse whoever
        reads it later.
        """
        provider, model = split_id(model_id)
        if not provider or not model:
            raise ValueError(
                f"'{model_id}' is not a model id. Use provider/model, "
                f"for example openrouter/deepseek/deepseek-r1:free.")
        caps = parse(capabilities)
        if not caps:
            raise ValueError(
                "Say what this model can do — at minimum, that it writes text.")
        spec = ModelSpec(provider=provider, model=model,
                         label=label or model, capabilities=caps, tier=tier,
                         context=int(context or 0), free=bool(free),
                         notes=notes, declared=True)
        with self._lock:
            self._custom[spec.id] = spec
            self._save()
        return spec

    def record_probe(self, model_id: str, capability, ok: bool,
                     error: str = "") -> None:
        """Write down what a real call proved, or failed to.

        Kept separate from `declare` on purpose: a declaration is
        somebody's belief, a probe is evidence, and collapsing the two
        would lose the only signal that says which is which.
        """
        import time as _time
        spec = self.get(model_id) or self.resolve(model_id)
        if spec is None:
            return
        verified = dict(spec.verified or {})
        verified[str(getattr(capability, "value", capability))] = {
            "ok": bool(ok), "at": _time.time(), "error": (error or "")[:300]}
        with self._lock:
            self._custom[spec.id] = ModelSpec(
                provider=spec.provider, model=spec.model, label=spec.label,
                capabilities=spec.capabilities, tier=spec.tier,
                context=spec.context, free=spec.free, notes=spec.notes,
                declared=spec.declared, verified=verified)
            self._save()

    def forget(self, model_id: str) -> bool:
        """Remove an operator-added model. Seed entries cannot be removed
        — a `declare` that overrides one is how you change it."""
        with self._lock:
            if model_id not in self._custom:
                return False
            self._custom.pop(model_id)
            self._save()
        return True


catalogue = ModelCatalogue()
