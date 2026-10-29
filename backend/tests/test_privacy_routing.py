"""
tests/test_privacy_routing.py — privacy mode must hold on every path that
sends client data out, not only on the report narratives.

The chat route embeds the dataset's schema and sample rows in its system
prompt and called Groq unconditionally; RAG embedding sent document text
to Google. Both are covered here, along with concurrent ingest into one
knowledge base, which could pair a chunk with another chunk's vector.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from app.config import config


@pytest.fixture()
def privacy_on(monkeypatch):
    monkeypatch.setattr(config, "llm_privacy_mode", True)
    # A configured cloud key is exactly when the leak would happen.
    monkeypatch.setattr(config, "groq_api_key", "gsk-would-leak")
    monkeypatch.setattr(config, "gemini_api_key", "would-leak")


def _no_cloud(*a, **kw):
    raise AssertionError("a cloud provider was called in privacy mode")


def test_chat_never_reaches_groq_in_privacy_mode(client, uploaded_dataset_id,
                                                 privacy_on, monkeypatch):
    from app.ai import llm_client, local_llm
    monkeypatch.setattr(llm_client.LLMClient, "chat", _no_cloud)
    monkeypatch.setattr(local_llm, "generate", lambda *a, **kw: None)
    r = client.post(f"/api/chat/{uploaded_dataset_id}", json={"message": "hi"})
    assert r.status_code == 503
    assert "LLM_PRIVACY_MODE" in r.json()["detail"]


def test_chat_uses_the_local_model_in_privacy_mode(client, uploaded_dataset_id,
                                                   privacy_on, monkeypatch):
    from app.ai import llm_client, local_llm
    seen = {}

    def fake_local(system, user, **kw):
        seen["system"], seen["user"] = system, user
        return ('{"tool": "count_values", "params": {"column": "department"}, '
                '"explanation": "Counts per department."}')

    monkeypatch.setattr(llm_client.LLMClient, "chat", _no_cloud)
    monkeypatch.setattr(local_llm, "generate", fake_local)
    r = client.post(f"/api/chat/{uploaded_dataset_id}", json={
        "message": "how many per department?",
        "history": [{"role": "user", "content": "earlier question"}]})
    assert r.status_code == 200, r.text
    assert r.json()["tool"] == "count_values"
    assert "earlier question" in seen["user"]
    assert "how many per department?" in seen["user"]


def test_groq_client_refuses_directly_in_privacy_mode(privacy_on):
    from app.ai.llm_client import LLMClient
    from app.ai.local_llm import LocalLLMError
    with pytest.raises(LocalLLMError):
        LLMClient("gsk-x").chat([{"role": "user", "content": "x"}])


def test_a_provider_failure_is_reported_not_disguised_as_misunderstanding(
        client, uploaded_dataset_id, monkeypatch):
    from app.ai import llm_client, local_llm

    def down(*a, **kw):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(config, "groq_api_key", "gsk-x")
    monkeypatch.setattr(llm_client.LLMClient, "chat", down)
    monkeypatch.setattr(local_llm, "generate", lambda *a, **kw: None)
    r = client.post(f"/api/chat/{uploaded_dataset_id}", json={"message": "hi"})
    assert r.status_code == 503
    assert "rate limited" in r.json()["detail"]


def test_rag_embeds_locally_in_privacy_mode(privacy_on, monkeypatch):
    from app.ai import gemini_client
    from app.rag import vector_store
    monkeypatch.setattr(gemini_client, "embed", _no_cloud)
    vecs, backend = vector_store.embed(["quarterly revenue by region"])
    assert backend == "local"
    assert vecs.shape == (1, vector_store._LOCAL_DIM)


def test_a_gemini_indexed_kb_says_why_it_cannot_answer_in_privacy_mode(
        privacy_on, tmp_path):
    # Returning no hits here told the user their documents did not cover
    # the question, which was false.
    from app.rag.vector_store import KnowledgeBase
    kb = KnowledgeBase("k1", "kb", str(tmp_path / "k1.pkl"))
    kb.chunks = [{"text": "revenue grew", "source": "a.txt", "locator": "file"}]
    kb.vectors = np.ones((1, 768), dtype=np.float32)
    kb.embedder = "gemini"
    with pytest.raises(RuntimeError, match="LLM_PRIVACY_MODE"):
        kb.search("revenue")


def test_a_gemini_indexed_kb_without_gemini_does_not_claim_the_docs_lack_the_answer(
        monkeypatch, tmp_path):
    from app.rag.vector_store import KnowledgeBase
    monkeypatch.setattr(config, "gemini_api_key", "")
    kb = KnowledgeBase("k2", "kb", str(tmp_path / "k2.pkl"))
    kb.chunks = [{"text": "revenue grew", "source": "a.txt", "locator": "file"}]
    kb.vectors = np.ones((1, 768), dtype=np.float32)
    kb.embedder = "gemini"
    with pytest.raises(RuntimeError, match="indexed with Gemini"):
        kb.search("revenue")


def test_concurrent_uploads_cannot_both_pass_the_size_limit(tmp_path, monkeypatch):
    # The limit was checked, then the (slow) embedding ran, then chunks were
    # appended — so two uploads could each pass the check against the same
    # count and together overshoot the limit the check exists to enforce.
    from app.rag import service, vector_store
    from app.rag.vector_store import KnowledgeBase, _local_embed

    monkeypatch.setattr(config, "gemini_api_key", "")
    monkeypatch.setattr(config, "rag_max_chunks_per_kb", 30)
    kb = KnowledgeBase("race", "kb", str(tmp_path / "race.pkl"))
    kb.owner = "local"

    real = vector_store.embed
    both_checked = threading.Barrier(2, timeout=1)

    def slow_embed(texts, task="retrieval_document"):
        try:
            both_checked.wait()   # old code: both threads get here
        except threading.BrokenBarrierError:
            pass                  # fixed code: the second is still locked out
        return real(texts, task)

    monkeypatch.setattr(vector_store, "embed", slow_embed)
    errors = []

    def upload(i):
        body = "\n\n".join("Document {} paragraph {} about item{}x{}. ".format(i, j, i, j) * 30
                           for j in range(20))
        try:
            service.ingest_file(kb, f"doc{i}.txt", body.encode())
        except service.KbLimitError as e:
            errors.append(e)

    threads = [threading.Thread(target=upload, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(kb.chunks) <= config.rag_max_chunks_per_kb
    assert len(errors) == 1
    assert kb.vectors.shape[0] == len(kb.chunks)
    assert np.allclose(kb.vectors, _local_embed([c["text"] for c in kb.chunks]))
