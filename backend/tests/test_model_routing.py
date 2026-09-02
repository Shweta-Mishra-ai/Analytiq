"""
Capability-based model routing.

The property under test is one sentence: a model is only ever given
work it is capable of doing — including as a fallback. Most of what
follows constructs a situation where the wrong model is the convenient
one, and asserts it is not reached.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from app.ai import routing, tasks
from app.ai.capabilities import Capability as C
from app.ai.capabilities import describe_gap, names, parse
from app.ai.model_catalogue import ModelCatalogue, ModelSpec, split_id
from app.config import config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every provider credentialed, no routing overrides — so each test
    states its own situation instead of inheriting one."""
    for field in ("groq_api_key", "gemini_api_key", "openrouter_api_key",
                  "cerebras_api_key", "together_api_key"):
        monkeypatch.setattr(config, field, "test-key")
    monkeypatch.setattr(config, "local_llm_url", "http://localhost:11434")
    monkeypatch.setattr(config, "llm_routing", "")
    monkeypatch.setattr(config, "llm_privacy_mode", False)
    monkeypatch.setattr(
        config, "llm_provider_order",
        "groq,openrouter,cerebras,together,gemini,local")


def _catalogue(monkeypatch, *specs) -> ModelCatalogue:
    """Replace the whole catalogue for one test."""
    cat = ModelCatalogue(os.path.join(tempfile.mkdtemp(), "models.json"))
    monkeypatch.setattr(cat, "all", lambda: list(specs))
    monkeypatch.setattr(routing, "catalogue", cat)
    import app.ai.model_catalogue as mc
    monkeypatch.setattr(mc, "catalogue", cat)
    return cat


def _unassigned(monkeypatch, task_name: str):
    """Clear a task's default so a test can observe pure preference
    ordering rather than the assignment winning first."""
    import dataclasses
    spec = tasks.TASKS[task_name]
    monkeypatch.setitem(tasks.TASKS, task_name,
                        dataclasses.replace(spec, default_model=""))


def _model(provider, model, caps, tier="balanced", context=128_000) -> ModelSpec:
    return ModelSpec(provider=provider, model=model, label=model,
                     capabilities=frozenset(caps), tier=tier, context=context)


# ── the invariant ────────────────────────────────────────

def test_a_vision_task_never_resolves_to_a_text_only_model(monkeypatch):
    """The headline. Every configured model can write; none can see. The
    correct answer is nothing at all, not the nearest model."""
    _catalogue(monkeypatch,
               _model("groq", "llama-3.1-8b-instant", {C.TEXT, C.JSON}),
               _model("openrouter", "some-llm", {C.TEXT, C.JSON, C.REASONING}))
    assert routing.resolve_models("table_extraction") == []


def test_the_fallback_chain_is_capability_gated_too(monkeypatch):
    """Gating only the first pick prevents nothing — the fallback is
    exactly where the wrong model gets in."""
    _catalogue(monkeypatch,
               _model("gemini", "vision-a", {C.VISION, C.JSON}),
               _model("groq", "text-b", {C.TEXT, C.JSON}),
               _model("openrouter", "vision-c", {C.VISION, C.JSON}))
    chain = [s.model for s in routing.resolve_models("table_extraction")]
    assert "text-b" not in chain
    assert set(chain) == {"vision-a", "vision-c"}


def test_an_assigned_model_that_cannot_do_the_job_is_skipped(monkeypatch):
    """A misconfiguration must not silently downgrade the work. The
    assignment is refused and a capable model serves instead."""
    monkeypatch.setattr(config, "llm_routing", "table_extraction=groq/text-only")
    _catalogue(monkeypatch,
               _model("groq", "text-only", {C.TEXT}),
               _model("gemini", "sees", {C.VISION, C.JSON}))
    assert [s.model for s in routing.resolve_models("table_extraction")] == ["sees"]

    problems = {p["task"]: p for p in routing.problems()}
    assert problems["table_extraction"]["kind"] == "incapable"
    assert "reads images" in problems["table_extraction"]["detail"]


def test_an_unknown_model_is_assumed_to_write_text_and_nothing_else(monkeypatch):
    """Guessing a capability an unknown model lacks is discovered when a
    client's report comes back broken. Text is what a chat endpoint is;
    everything else has to be claimed."""
    from app.ai.model_catalogue import catalogue as real
    spec = real.resolve("openrouter/nobody/has-heard-of-this:free")
    assert names(spec.capabilities) == ["text"]
    assert not spec.declared

    monkeypatch.setattr(config, "llm_routing",
                        "table_extraction=openrouter/nobody/has-heard-of-this:free")
    problem = {p["task"]: p for p in routing.problems()}["table_extraction"]
    assert problem["kind"] == "incapable"
    assert "not in the catalogue" in problem["detail"]


# ── ineligibility says which of three things is wrong ────

def test_a_missing_key_is_not_reported_as_a_small_context_window(monkeypatch):
    """Told the wrong reason, someone goes looking for a bigger model and
    never finds the problem."""
    monkeypatch.setattr(config, "gemini_api_key", "")
    monkeypatch.setattr(config, "llm_routing", "rag_answer=gemini/gemini-3.6-flash")
    problem = {p["task"]: p for p in routing.problems()}["rag_answer"]
    assert problem["kind"] == "not_configured"
    assert "GEMINI_API_KEY" in problem["detail"]


def test_a_model_too_small_for_the_job_is_reported_as_such(monkeypatch):
    monkeypatch.setattr(config, "llm_routing", "rag_report=groq/tiny")
    _catalogue(monkeypatch,
               _model("groq", "tiny", {C.TEXT, C.REASONING}, context=8_000))
    problem = {p["task"]: p for p in routing.problems()}["rag_report"]
    assert problem["kind"] == "small_context"
    assert "8,000" in problem["detail"]


def test_an_unrecorded_context_window_does_not_disqualify(monkeypatch):
    """0 means "nobody measured it", not "zero". Filtering those out
    would drop every model the catalogue has no number for."""
    _catalogue(monkeypatch,
               _model("groq", "unmeasured", {C.TEXT, C.REASONING}, context=0))
    assert [s.model for s in routing.resolve_models("rag_report")] == ["unmeasured"]


# ── preference, not gating ───────────────────────────────

def test_a_high_volume_task_prefers_the_cheap_fast_model(monkeypatch):
    """Twenty captions per report; the expensive model is allowed but
    should not be first."""
    _unassigned(monkeypatch, "chart_caption")
    _catalogue(monkeypatch,
               _model("gemini", "deep-one", {C.TEXT}, tier="deep"),
               _model("groq", "fast-one", {C.TEXT}, tier="fast"))
    assert routing.resolve_models("chart_caption")[0].model == "fast-one"


def test_a_reasoning_task_prefers_the_deeper_model(monkeypatch):
    _unassigned(monkeypatch, "rag_report")
    _catalogue(monkeypatch,
               _model("groq", "fast-one", {C.TEXT, C.REASONING}, tier="fast"),
               _model("gemini", "deep-one", {C.TEXT, C.REASONING}, tier="deep"))
    assert routing.resolve_models("rag_report")[0].model == "deep-one"


def test_cheap_models_are_still_allowed_to_serve_a_deep_task(monkeypatch):
    """Cost is an ordering preference, never a gate — a capable cheap
    model beats no model."""
    _catalogue(monkeypatch,
               _model("groq", "fast-one", {C.TEXT, C.REASONING}, tier="fast"))
    assert [s.model for s in routing.resolve_models("rag_report")] == ["fast-one"]


# ── privacy mode ─────────────────────────────────────────

def test_privacy_mode_filters_every_task_including_vision(monkeypatch):
    monkeypatch.setattr(config, "llm_privacy_mode", True)
    _catalogue(monkeypatch,
               _model("gemini", "cloud-eyes", {C.VISION, C.JSON}),
               _model("local", "gemma3:12b", {C.TEXT, C.JSON, C.VISION}))
    assert [s.model for s in routing.resolve_models("table_extraction")] \
        == ["gemma3:12b"]
    assert [s.provider for s in routing.resolve_models("chart_caption")] == ["local"]


def test_privacy_mode_with_no_local_model_reaches_nothing(monkeypatch):
    monkeypatch.setattr(config, "llm_privacy_mode", True)
    _catalogue(monkeypatch, _model("gemini", "cloud-eyes", {C.VISION, C.JSON}))
    assert routing.resolve_models("table_extraction") == []


# ── configuration ────────────────────────────────────────

def test_a_bare_provider_name_still_works(monkeypatch):
    """The older LLM_ROUTING form. Silently changing what an existing
    deployment's configuration means is the failure this whole change
    exists to remove."""
    monkeypatch.setattr(config, "llm_routing", "chart_caption=cerebras")
    monkeypatch.setattr(config, "cerebras_model", "llama-3.3-70b")
    assert routing.resolve_models("chart_caption")[0].id == "cerebras/llama-3.3-70b"


def test_a_model_id_with_slashes_and_colons_parses(monkeypatch):
    """OpenRouter slugs contain both. Only the first slash is
    structural."""
    assert split_id("openrouter/meta-llama/llama-3.3-70b-instruct:free") == \
        ("openrouter", "meta-llama/llama-3.3-70b-instruct:free")
    monkeypatch.setattr(
        config, "llm_routing",
        "chart_caption=openrouter/meta-llama/llama-3.3-70b-instruct:free")
    assert routing.assignments()["chart_caption"].model_id == \
        "openrouter/meta-llama/llama-3.3-70b-instruct:free"


def test_a_stale_task_name_keeps_working_and_says_it_is_stale(monkeypatch):
    """Dropping the row would change behaviour with no signal."""
    monkeypatch.setattr(config, "llm_routing", "chart_analysis=cerebras")
    monkeypatch.setattr(config, "cerebras_model", "llama-3.3-70b")
    assert routing.assignments()["chart_caption"].source == "env"
    assert routing.deprecated_in_use() == [
        {"from": "chart_analysis", "to": "chart_caption"}]


def test_an_unparseable_routing_entry_costs_that_entry_only(monkeypatch):
    monkeypatch.setattr(config, "llm_routing", "nonsense,chart_caption=groq")
    monkeypatch.setattr(config, "llm_model", "llama-3.3-70b-versatile")
    assert routing.assignments()["chart_caption"].source == "env"
    assert routing.assignments()["rag_answer"].source == "default"


def test_an_unknown_provider_is_reported_not_silently_dropped(monkeypatch):
    monkeypatch.setattr(config, "llm_routing", "chart_caption=nosuchprovider")
    problem = {p["task"]: p for p in routing.problems()}["chart_caption"]
    assert problem["kind"] == "unknown_provider"
    assert "names no provider" in problem["detail"]


# ── the registry itself ──────────────────────────────────

def test_every_default_model_can_do_its_own_task():
    """Catches "someone added a task and pointed it at a model that
    cannot serve it" at commit time rather than in production."""
    from app.ai.model_catalogue import catalogue
    for task in tasks.all_tasks():
        if not task.default_model:
            continue
        spec = catalogue.get(task.default_model)
        assert spec is not None, \
            f"{task.name} defaults to {task.default_model}, which is not in the catalogue"
        assert spec.can(task.requires), (
            f"{task.name} defaults to {spec.id}, but "
            f"{describe_gap(task.requires, spec.capabilities)}")
        if task.min_context and spec.context:
            assert spec.context >= task.min_context, (
                f"{task.name} defaults to {spec.id}, which holds "
                f"{spec.context} tokens and needs {task.min_context}")


def test_every_required_capability_is_offered_by_some_model():
    """A task whose capability no model has is a task that can never
    run — except cover art, which is off by design."""
    from app.ai.model_catalogue import catalogue
    for task in tasks.all_tasks():
        if task.name == "cover_art":
            continue
        assert catalogue.with_capabilities(task.requires), \
            f"no catalogue model can serve {task.name}"


def test_every_task_says_what_happens_without_a_model():
    """A routing table that cannot answer "and if not?" is decoration."""
    for task in tasks.all_tasks():
        assert task.degrades_to, f"{task.name} does not say what it degrades to"


def test_cover_art_is_off_until_someone_turns_it_on():
    """A generated image must never appear in a deliverable by accident."""
    assert tasks.TASKS["cover_art"].default_model == ""
    assert routing.assignments()["cover_art"].model_id == ""


def test_capabilities_parse_leniently_but_never_invent():
    assert names(parse(["text", "vision", "telepathy"])) == ["text", "vision"]
    assert describe_gap({C.VISION}, {C.TEXT}) == \
        "it cannot do what this task needs: reads images"
    assert describe_gap({C.TEXT}, {C.TEXT, C.VISION}) == ""


def test_a_local_multimodal_model_can_serve_vision_without_any_cloud_key(monkeypatch):
    """Worth stating on its own: the fully-offline path is real. Gemma 3
    reads images, so a machine with Ollama and no API keys at all still
    turns a photograph of a table into data."""
    for field in ("groq_api_key", "gemini_api_key", "openrouter_api_key",
                  "cerebras_api_key", "together_api_key"):
        monkeypatch.setattr(config, field, "")
    assert [s.id for s in routing.resolve_models("table_extraction")] == \
        ["local/gemma3:12b"]


def test_status_reports_a_task_no_model_can_serve(monkeypatch):
    """The System page needs to say "this one falls back to the engines"
    rather than render a blank row."""
    # No cloud vision, and no local model either — the honest "nothing
    # here can see" case. (With a local Gemma configured, vision *is*
    # served, which is the point of having a local multimodal model.)
    monkeypatch.setattr(config, "gemini_api_key", "")
    monkeypatch.setattr(config, "local_llm_url", "")
    rows = {r["task"]: r for r in routing.status()["tasks"]}
    assert rows["table_extraction"]["served"] is False
    assert rows["table_extraction"]["degrades_to"]
    assert rows["chart_caption"]["served"] is True


# ── the narrative cache is keyed on the model that answers ──

def test_two_tasks_on_different_models_do_not_share_a_cache_entry(monkeypatch,
                                                                  tmp_path):
    """The bug this replaces: `chat()` wrote the answering provider's
    model onto the client, and the cache keyed on that attribute — so a
    chat turn could change the key a later report call was stored under,
    and two different models' wording could collide on one entry."""
    from app.ai import llm_client as mod
    from app.ai import providers as providers_mod
    from app.services import llm_cache as cache_mod

    monkeypatch.setattr(cache_mod, "llm_cache",
                        cache_mod.LLMCache(base_dir=str(tmp_path / "llm")))

    class _Provider:
        name = "groq"
        label = "Groq"
        model = "default-model"
        free = local = False
        key_env = "GROQ_API_KEY"

        def is_credentialed(self):
            return True

        def is_configured(self):
            return True

        def missing(self):
            return ""

        def generate(self, system, user, max_tokens=512, temperature=0.2,
                     timeout_sec=None, model="", json_mode=False):
            return f"answer from {model}"

    monkeypatch.setattr(providers_mod, "_providers",
                        lambda: {"groq": _Provider()})
    _catalogue(monkeypatch,
               _model("groq", "cheap", {C.TEXT}, tier="fast"),
               _model("groq", "deep", {C.TEXT, C.REASONING}, tier="deep"))
    monkeypatch.setattr(config, "llm_routing",
                        "chart_caption=groq/cheap,executive_summary=groq/deep")

    client = mod.LLMClient()
    caption = client.chat_task("same system", "same user", task="chart_caption")
    summary = client.chat_task("same system", "same user",
                               task="executive_summary")

    assert caption == "answer from cheap"
    assert summary == "answer from deep", (
        "the second task must not be served the first model's cached wording")


def test_a_chat_turn_cannot_move_a_later_cache_key(monkeypatch, tmp_path):
    """`LLMClient.model` is no longer mutated by a chat turn."""
    from app.ai import llm_client as mod
    client = mod.LLMClient()
    before = client.model
    monkeypatch.setattr(client, "chat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    client.chat_safe([{"role": "user", "content": "hi"}], fallback="{}")
    assert client.model == before


def test_the_chat_page_works_without_a_groq_key(monkeypatch):
    """It used to hard-fail on GROQ_API_KEY alone, which was wrong the
    moment a second provider existed."""
    import io
    import tempfile as tmp
    from fastapi.testclient import TestClient
    from app.services.dataset_store import DatasetStore
    import app.services.dataset_store as ds_mod
    import app.api.datasets as datasets_api
    import app.api.chat as chat_api

    monkeypatch.setattr(config, "groq_api_key", "")
    monkeypatch.setattr(config, "openrouter_api_key", "or-key")
    store = DatasetStore(tmp.mkdtemp())
    for module in (ds_mod, datasets_api, chat_api):
        monkeypatch.setattr(module, "store", store, raising=False)

    from app.main import app
    client = TestClient(app)
    up = client.post("/api/datasets/upload", files={
        "file": ("t.csv", io.BytesIO(b"a,b\n1,2\n3,4\n"), "text/csv")})
    ds_id = up.json()["meta"]["dataset_id"]

    reply = client.post(f"/api/chat/{ds_id}", json={"message": "hi", "history": []})
    assert reply.status_code != 503, reply.text


def test_the_chat_page_says_so_when_nothing_can_serve_it(monkeypatch):
    import io
    import tempfile as tmp
    from fastapi.testclient import TestClient
    from app.services.dataset_store import DatasetStore
    import app.services.dataset_store as ds_mod
    import app.api.datasets as datasets_api
    import app.api.chat as chat_api

    for field in ("groq_api_key", "gemini_api_key", "openrouter_api_key",
                  "cerebras_api_key", "together_api_key"):
        monkeypatch.setattr(config, field, "")
    monkeypatch.setattr(config, "local_llm_url", "")
    store = DatasetStore(tmp.mkdtemp())
    for module in (ds_mod, datasets_api, chat_api):
        monkeypatch.setattr(module, "store", store, raising=False)

    from app.main import app
    client = TestClient(app)
    up = client.post("/api/datasets/upload", files={
        "file": ("t.csv", io.BytesIO(b"a,b\n1,2\n3,4\n"), "text/csv")})
    ds_id = up.json()["meta"]["dataset_id"]

    reply = client.post(f"/api/chat/{ds_id}", json={"message": "hi", "history": []})
    assert reply.status_code == 503
    assert "System page" in reply.json()["detail"]


# ── the paths that used to bypass routing entirely ───────

def test_table_extraction_no_longer_demands_a_gemini_key(monkeypatch):
    """It used to refuse outright without GEMINI_API_KEY. Any model that
    can see is now enough — including one on your own hardware."""
    from app.services import table_extractor

    monkeypatch.setattr(config, "gemini_api_key", "")
    _catalogue(monkeypatch,
               _model("openrouter", "some-vision-model", {C.VISION, C.JSON}))

    seen = {}

    def _fake(image, prompt, system="", task="", mime="", max_tokens=0,
              json_mode=False, timeout_sec=None):
        seen["task"] = task
        seen["json"] = json_mode
        return ('{"found": true, "columns": ["a"], '
                '"rows": [[1], [2]]}')

    from app.ai import multimodal
    monkeypatch.setattr(multimodal, "describe_image", _fake)

    from PIL import Image
    import io as _io
    buf = _io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")

    df, _warnings = table_extractor.extract_table_from_image("t.png",
                                                             buf.getvalue())
    assert len(df) == 2
    assert seen["task"] == "table_extraction"
    assert seen["json"] is True, "extraction parses the reply, so it must ask for JSON"


def test_table_extraction_names_the_missing_capability(monkeypatch):
    """Not "GEMINI_API_KEY is required" — that pointed at one vendor and
    was wrong the moment there was a second."""
    from app.services.table_extractor import ExtractionError, extract_table_from_image

    _catalogue(monkeypatch, _model("groq", "text-only", {C.TEXT, C.JSON}))
    import io as _io
    from PIL import Image
    buf = _io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")

    with pytest.raises(ExtractionError) as excinfo:
        extract_table_from_image("t.png", buf.getvalue())
    message = str(excinfo.value)
    assert "System page" in message
    assert "GEMINI_API_KEY" not in message


def test_rag_answering_can_use_a_provider_it_never_could_before(monkeypatch):
    """OpenRouter, Cerebras and Together were unreachable for RAG: the
    old ladder only knew local, Gemini and Groq."""
    from app.rag import service

    for field in ("groq_api_key", "gemini_api_key"):
        monkeypatch.setattr(config, field, "")
    monkeypatch.setattr(config, "local_llm_url", "")
    _catalogue(monkeypatch,
               _model("openrouter", "reasoner", {C.TEXT, C.REASONING}))

    seen = {}

    class _Client:
        def chat_task(self, system, user, task="default", max_tokens=400,
                      force=""):
            seen["task"] = task
            return "an answer"

    monkeypatch.setattr("app.ai.llm_client.get_client", lambda *a, **k: _Client())
    assert service._generate("sys", "user", task="rag_answer") == "an answer"
    assert seen["task"] == "rag_answer"


def test_rag_keeps_its_privacy_wording_when_nothing_answers(monkeypatch):
    """The old message ended "No data was sent anywhere" — a claim a
    client relies on. It must survive the refactor verbatim."""
    from app.rag import service

    monkeypatch.setattr(config, "llm_privacy_mode", True)

    class _Client:
        def chat_task(self, *a, **k):
            return None

    monkeypatch.setattr("app.ai.llm_client.get_client", lambda *a, **k: _Client())
    with pytest.raises(RuntimeError) as excinfo:
        service._generate("sys", "user", task="rag_answer")
    assert "No data was sent anywhere" in str(excinfo.value)


def test_the_knowledge_base_prefers_a_model_on_your_own_hardware(monkeypatch):
    """A knowledge base holds contracts and policies. The old code tried
    local first, always; that posture is now declared on the task, so it
    cannot be undone by reordering an environment variable."""
    monkeypatch.setattr(config, "llm_provider_order", "gemini,groq,local")
    assert routing.resolve_models("rag_answer")[0].provider == "local"
    # And it is a posture for the knowledge base only — a chart caption
    # should still go to whatever is cheapest and fastest.
    assert routing.resolve_models("chart_caption")[0].provider != "local"


def test_switching_embedding_model_is_detected_rather_than_silently_mixed():
    """Two embedding models put the same sentence in different places, so
    searching a stored index with another model's query vector returns
    confident nonsense. The backend identity carries the model so the
    store's existing re-embed path catches the change."""
    from app.rag.vector_store import _family
    assert _family("gemini:gemini-embedding-001") == "gemini"
    assert _family("sentence:all-MiniLM-L6-v2") == "sentence"
    # Manifests written before the suffix existed still resolve.
    assert _family("gemini") == "gemini"
    assert _family("") == "local"
    assert "gemini:model-a" != "gemini:model-b", \
        "a model change must not compare equal, or no re-embed happens"


# ── the executive summary is rewritten, never generated ──

def _polish_with(monkeypatch, reply):
    """Assign a model to the polish task and make it answer `reply`."""
    import pandas as pd
    from app.ai import report_narrator

    _catalogue(monkeypatch,
               _model("groq", "reasoner", {C.TEXT, C.REASONING}))
    monkeypatch.setattr(config, "llm_routing", "executive_summary=groq/reasoner")

    class _Client:
        def chat_task(self, *a, **k):
            return reply

    monkeypatch.setattr("app.ai.llm_client.get_client",
                        lambda *a, **k: _Client())
    return report_narrator, pd.DataFrame({"attrition": [1, 0], "salary": [1, 2]})


ORIGINAL = "Attrition is 16.1% across 1,470 staff, concentrated in Sales."


def test_the_polish_is_off_until_a_model_is_assigned(monkeypatch):
    """The default for the most scrutinised paragraph in the deliverable
    is the wording the engine computed."""
    import pandas as pd
    from app.ai import report_narrator
    _catalogue(monkeypatch, _model("groq", "reasoner", {C.TEXT, C.REASONING}))
    assert report_narrator.polish_executive_summary(
        ORIGINAL, pd.DataFrame({"attrition": [1, 0]})) == ORIGINAL


def test_a_clean_rewrite_is_accepted(monkeypatch):
    better = "Attrition stands at 16.1% of 1,470 staff and is concentrated in Sales."
    narrator, df = _polish_with(monkeypatch, better)
    assert narrator.polish_executive_summary(ORIGINAL, df) == better


def test_a_rewrite_that_changes_a_figure_is_rejected(monkeypatch):
    """A rewrite that alters a number has stopped being a rewrite."""
    narrator, df = _polish_with(
        monkeypatch,
        "Attrition stands at 22.4% across 1,470 staff, concentrated in Sales.")
    assert narrator.polish_executive_summary(ORIGINAL, df) == ORIGINAL


def test_a_rewrite_that_drops_a_figure_is_rejected(monkeypatch):
    narrator, df = _polish_with(
        monkeypatch, "Attrition is high and concentrated in Sales.")
    assert narrator.polish_executive_summary(ORIGINAL, df) == ORIGINAL


def test_a_rewrite_naming_a_metric_the_data_lacks_is_rejected(monkeypatch):
    narrator, df = _polish_with(
        monkeypatch,
        "Attrition is 16.1% across 1,470 staff, driven by churn rate "
        "and website traffic.")
    assert narrator.polish_executive_summary(ORIGINAL, df) == ORIGINAL


def test_reordering_a_sentence_is_still_a_legitimate_rewrite(monkeypatch):
    """Numbers are compared as a set, so moving a clause is allowed."""
    reordered = "Concentrated in Sales, attrition is 16.1% across 1,470 staff."
    narrator, df = _polish_with(monkeypatch, reordered)
    assert narrator.polish_executive_summary(ORIGINAL, df) == reordered


def test_a_model_failure_leaves_the_engines_wording(monkeypatch):
    import pandas as pd
    from app.ai import report_narrator
    _catalogue(monkeypatch, _model("groq", "reasoner", {C.TEXT, C.REASONING}))
    monkeypatch.setattr(config, "llm_routing", "executive_summary=groq/reasoner")

    class _Broken:
        def chat_task(self, *a, **k):
            raise RuntimeError("provider down")

    monkeypatch.setattr("app.ai.llm_client.get_client",
                        lambda *a, **k: _Broken())
    assert report_narrator.polish_executive_summary(
        ORIGINAL, pd.DataFrame({"attrition": [1, 0]})) == ORIGINAL


# ── the runtime override ─────────────────────────────────

@pytest.fixture
def store(monkeypatch):
    from app.ai import settings_store as mod
    st = mod.SettingsStore(os.path.join(tempfile.mkdtemp(), "llm_routing.json"))
    monkeypatch.setattr(mod, "settings_store", st)
    return st


def test_a_change_made_here_beats_the_environment(store, monkeypatch):
    """The reason this file exists: on Render an environment change is a
    restart, which makes trying a different model a deploy."""
    monkeypatch.setattr(config, "llm_routing",
                        "chart_caption=groq/llama-3.3-70b-versatile")
    assert routing.assignments()["chart_caption"].source == "env"

    store.assign("chart_caption", "groq/llama-3.1-8b-instant")
    assigned = routing.assignments()["chart_caption"]
    assert assigned.model_id == "groq/llama-3.1-8b-instant"
    assert assigned.source == "runtime"


def test_clearing_returns_everything_to_the_environment(store, monkeypatch):
    monkeypatch.setattr(config, "llm_routing",
                        "chart_caption=groq/llama-3.3-70b-versatile")
    store.assign("chart_caption", "groq/llama-3.1-8b-instant")
    store.clear()
    assert routing.assignments()["chart_caption"].model_id == \
        "groq/llama-3.3-70b-versatile"


def test_an_incapable_assignment_is_refused_at_the_moment_it_is_made(store):
    """Accepting it and skipping it at the point of use looks exactly
    like the model never being called."""
    from app.ai.settings_store import RoutingRejected
    with pytest.raises(RoutingRejected) as excinfo:
        store.assign("table_extraction", "groq/llama-3.1-8b-instant")
    assert "reads images" in str(excinfo.value)
    assert store.routing() == {}, "nothing may be written when it is refused"


def test_a_model_too_small_for_the_task_is_refused(store, monkeypatch):
    """Capable of the work, but cannot hold the prompt. A separate
    refusal from "it cannot reason", and it has to say which."""
    from app.ai.settings_store import RoutingRejected
    _catalogue(monkeypatch,
               _model("groq", "short-memory", {C.TEXT, C.REASONING},
                      context=8_000))
    with pytest.raises(RoutingRejected) as excinfo:
        store.assign("rag_report", "groq/short-memory")
    message = str(excinfo.value)
    assert "8,000" in message and "64,000" in message
    assert "cannot do what this task needs" not in message


def test_an_undeclared_model_is_refused_with_the_reason_and_the_remedy(store):
    from app.ai.settings_store import RoutingRejected
    with pytest.raises(RoutingRejected) as excinfo:
        store.assign("table_extraction", "openrouter/nobody/knows-this:free")
    message = str(excinfo.value)
    assert "not in the catalogue" in message
    assert "declare" in message.lower()


def test_a_provider_without_a_key_yet_is_allowed(store, monkeypatch):
    """Someone setting up a deployment assigns models first and adds keys
    after. The System page already reports an unconfigured provider."""
    monkeypatch.setattr(config, "cerebras_api_key", "")
    store.assign("chart_caption", "cerebras/llama-3.3-70b")
    assert store.routing()["chart_caption"] == "cerebras/llama-3.3-70b"


def test_an_unknown_task_is_refused(store):
    from app.ai.settings_store import RoutingRejected
    with pytest.raises(RoutingRejected, match="is not a task"):
        store.assign("summarise_the_vibes", "groq/llama-3.1-8b-instant")


def test_a_corrupt_override_file_costs_the_overrides_and_nothing_else(monkeypatch):
    """The environment defaults are a working configuration. Refusing to
    start over a file the operator can delete would be the worse
    failure."""
    from app.ai import settings_store as mod
    path = os.path.join(tempfile.mkdtemp(), "llm_routing.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not json at all")
    st = mod.SettingsStore(path)
    assert st.routing() == {}


def test_an_assignment_survives_a_restart(store):
    from app.ai.settings_store import SettingsStore
    store.assign("chart_caption", "groq/llama-3.1-8b-instant", actor="tester")
    reopened = SettingsStore(store.path)
    assert reopened.routing()["chart_caption"] == "groq/llama-3.1-8b-instant"
    assert reopened.as_dict()["updated_by"] == "tester"


# ── generated imagery: narrow, off, and labelled ─────────

def _png(colour=(40, 60, 90)) -> bytes:
    import io as _io
    from PIL import Image
    buf = _io.BytesIO()
    Image.new("RGB", (32, 32), colour).save(buf, format="PNG")
    return buf.getvalue()


def _image_model(monkeypatch, raw=None):
    """Assign a model to cover_art and make it return an image."""
    from app.ai import providers as providers_mod

    _catalogue(monkeypatch, _model("gemini", "painter", {C.IMAGE_GEN},
                                   context=0))
    monkeypatch.setattr(config, "llm_routing", "cover_art=gemini/painter")

    calls = []

    class _Painter:
        name = "gemini"
        label = "Painter"
        model = "painter"
        free = local = False
        key_env = "GEMINI_API_KEY"

        def is_credentialed(self):
            return True

        def is_configured(self):
            return True

        def missing(self):
            return ""

        def generate_image(self, prompt, size="1024x1024", model="",
                           timeout_sec=None):
            calls.append(prompt)
            return raw if raw is not None else _png()

    monkeypatch.setattr(providers_mod, "_providers",
                        lambda: {"gemini": _Painter()})
    return calls


def test_no_image_is_generated_unless_a_model_is_assigned(monkeypatch):
    """A picture appearing in a client's deliverable that nobody asked
    for is a defect, not a convenience."""
    from app.ai import imagery
    _catalogue(monkeypatch, _model("gemini", "painter", {C.IMAGE_GEN}))
    assert imagery.is_enabled() is False
    assert imagery.generate_cover("Attrition Review") is None


def test_an_assigned_model_produces_a_captioned_image(monkeypatch):
    from app.ai import imagery
    _image_model(monkeypatch)
    result = imagery.generate_cover("Attrition Review", domain="hr")
    assert result["image"]
    assert result["model"] == "gemini/painter"
    assert "contains" in result["caption"] and "no data" in result["caption"]


def test_the_prompt_never_carries_anything_from_the_data(monkeypatch):
    """The image cannot depict the data even by accident, because the
    data never reaches the prompt — only the report title, which is
    already printed on the cover."""
    from app.ai import imagery
    calls = _image_model(monkeypatch)
    imagery.generate_cover("Attrition Review", domain="human resources")
    prompt = calls[0]
    assert "Attrition Review" in prompt
    assert "no charts" in prompt and "no text" in prompt
    for forbidden in ("monthly_income", "employee", "16.1%", "row"):
        assert forbidden not in prompt.lower()


def test_the_image_carries_its_own_provenance(monkeypatch):
    """The caption covers the report; this covers the image after
    someone has pulled it out of the PDF."""
    import io as _io
    from PIL import Image
    from app.ai import imagery

    _image_model(monkeypatch)
    result = imagery.generate_cover("Attrition Review")
    img = Image.open(_io.BytesIO(result["image"]))
    assert img.info.get("Analytiq-Generated") == "true"
    assert img.info.get("Analytiq-Model") == "gemini/painter"
    assert "no data" in img.info.get("Analytiq-Note", "").lower()


def test_generating_an_image_is_recorded_in_the_audit_trail(monkeypatch):
    """The same hash-chained trail the numbers get."""
    from app.ai import imagery
    from app.services import integrity

    _image_model(monkeypatch)
    workspace = tempfile.mkdtemp()
    import pandas as pd
    integrity.record_ingest(workspace, "ds", pd.DataFrame({"a": [1, 2]}),
                            "f.csv")
    imagery.generate_cover("Attrition Review", dataset_dir=workspace,
                           actor="tester")

    entries = integrity.read_audit(workspace)
    assert entries[-1]["event"] == "generate"
    assert entries[-1]["detail"]["contains_data"] is False
    assert entries[-1]["detail"]["placement"] == "report cover"
    ok, why = integrity.verify_chain(entries)
    assert ok, why


def test_a_failed_image_never_costs_anyone_their_report(monkeypatch):
    from app.ai import imagery
    from app.ai import providers as providers_mod

    _catalogue(monkeypatch, _model("gemini", "painter", {C.IMAGE_GEN}))
    monkeypatch.setattr(config, "llm_routing", "cover_art=gemini/painter")

    class _Broken:
        name, label, model = "gemini", "Painter", "painter"
        free = local = False
        key_env = "GEMINI_API_KEY"

        def is_credentialed(self):
            return True

        def is_configured(self):
            return True

        def missing(self):
            return ""

        def generate_image(self, *a, **k):
            raise RuntimeError("the image service is down")

    monkeypatch.setattr(providers_mod, "_providers", lambda: {"gemini": _Broken()})
    assert imagery.generate_cover("Attrition Review") is None


def test_image_generation_is_reachable_from_exactly_one_module():
    """Structural. The argument for allowing generated imagery at all
    depends on it being confined to the cover, and a rule that lives only
    in a docstring is not a rule."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "app"
    callers = set()
    for path in root.rglob("*.py"):
        if path.name in ("imagery.py", "providers.py", "gemini_client.py"):
            continue
        if "generate_image" in path.read_text(encoding="utf-8"):
            callers.add(path.name)
    assert callers == set(), (
        "generate_image is called outside ai/imagery.py: " + ", ".join(callers))


def test_the_png_provenance_does_not_survive_the_pdf_and_the_caption_does():
    """Recording a measured limitation so nobody later assumes otherwise.

    The PNG text chunks are stripped when ReportLab embeds the image, so
    the caption and the audit trail are the labelling that actually
    reaches a reader of the finished report. If this test ever starts
    failing because the chunks survive, that is good news — but it
    should be a deliberate discovery, not a silent one.
    """
    import io as _io
    from PIL import Image
    from reportlab.lib.utils import ImageReader

    from app.ai import imagery

    stamped = imagery._stamp(_png(), "gemini/painter", "a prompt")
    assert Image.open(_io.BytesIO(stamped)).info.get("Analytiq-Generated") == "true"

    # What ReportLab hands the PDF is a re-encode, and it carries none of it.
    reader = ImageReader(_io.BytesIO(stamped))
    reencoded = _io.BytesIO()
    reader._image.save(reencoded, format="PNG")
    assert "Analytiq-Generated" not in Image.open(
        _io.BytesIO(reencoded.getvalue())).info

    # The caption is what holds, so it must be non-empty and unambiguous.
    assert "no data" in imagery.CAPTION.lower()


def test_every_routing_endpoint_answers_with_the_same_shape(monkeypatch):
    """A write that answers with a subset of the read leaves the caller
    holding a half-populated object. This was a real crash: changing a
    dropdown blanked the page, because the POST omitted `capabilities`
    and the render needed it."""
    import tempfile as tmp
    from fastapi.testclient import TestClient
    from app.ai import settings_store as ss

    monkeypatch.setattr(config, "data_dir", tmp.mkdtemp())
    monkeypatch.setattr(
        ss, "settings_store",
        ss.SettingsStore(os.path.join(config.data_dir, "llm_routing.json")))

    from app.main import app
    client = TestClient(app)

    read = client.get("/api/admin/routing").json()
    written = client.post("/api/admin/routing", json={
        "task": "chart_caption",
        "model_id": "groq/llama-3.3-70b-versatile"}).json()
    cleared = client.delete("/api/admin/routing").json()

    assert set(read) == set(written) == set(cleared)
    for payload in (read, written, cleared):
        assert payload["capabilities"], "the UI renders capability names from this"
        assert payload["tasks"]
