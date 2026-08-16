"""
The knowledge base answers from the client's own documents, or it says it
cannot.

`search()` returned the k best-scoring chunks unconditionally. Ask a KB
of HR policies about warehouse throughput and it handed the model six
unrelated paragraphs, which the model duly answered from — with
citations, because the citations point at the paragraphs that were
supplied. That is the worst failure mode available to this feature: an
answer that looks sourced and is not, about a client's own material.

The limits tested here exist for a different reason: a KB is held in
memory and rewritten to disk on every ingest, so one user's document
library is everyone's outage.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.config import config
from app.rag.service import (KbLimitError, _citation_count,
                             check_ingest_limits, chunk_passages)
from app.rag.vector_store import (MIN_SCORE, KnowledgeBase,
                                  _filter_by_relevance)


def _kb(tmp_path, texts, source="policies.pdf"):
    kb = KnowledgeBase("kb1", "Test KB", str(tmp_path / "kb1.pkl"))
    kb.add_chunks(
        [{"text": t, "source": source, "locator": f"p{i+1}"}
         for i, t in enumerate(texts)],
        source, "document")
    return kb


HR_DOCS = [
    "Annual leave entitlement is 25 days per year, accrued monthly from the "
    "start date. Unused leave may be carried over up to five days.",
    "The probation period is six months, during which either party may "
    "terminate the contract with two weeks notice.",
    "Expense claims must be submitted within 30 days with an itemised "
    "receipt attached. Claims over 500 require line manager approval.",
    "Remote working is available two days a week by arrangement with the "
    "line manager, subject to role suitability.",
]


# ══════════════════════════════════════════════════════════
#  Relevance filtering
# ══════════════════════════════════════════════════════════

def test_a_question_the_documents_cover_returns_passages(tmp_path):
    kb = _kb(tmp_path, HR_DOCS)
    hits = kb.search("how many days of annual leave do staff get")
    assert hits, "a question answered verbatim in the documents found nothing"
    assert "annual leave" in hits[0]["text"].lower()


def test_a_question_the_documents_do_not_cover_returns_nothing(tmp_path):
    """The alternative is six unrelated HR paragraphs handed to a model
    that will answer from them."""
    kb = _kb(tmp_path, HR_DOCS)
    hits = kb.search("what was warehouse throughput in the third quarter")
    assert hits == [], \
        "returned unrelated passages for a question the KB cannot answer: {}".format(
            [h["text"][:60] for h in hits])


def test_padding_chunks_are_dropped(tmp_path):
    """Chunks scoring far below the best match were only returned because
    k slots had to be filled."""
    hits = [
        {"text": "a", "score": 0.90},
        {"text": "b", "score": 0.85},
        {"text": "c", "score": 0.20},
        {"text": "d", "score": 0.05},
    ]
    kept = _filter_by_relevance(hits, "gemini")
    assert [h["text"] for h in kept] == ["a", "b"]


def test_nothing_survives_when_even_the_best_hit_is_weak():
    hits = [{"text": "a", "score": 0.10}, {"text": "b", "score": 0.08}]
    assert _filter_by_relevance(hits, "gemini") == []


def test_each_embedder_has_its_own_floor():
    """Gemini and the local hashing embedder score unrelated text on
    different scales; one threshold cannot serve both."""
    assert MIN_SCORE["gemini"] > MIN_SCORE["local"]
    weak = [{"text": "a", "score": 0.20}]
    assert _filter_by_relevance(weak, "gemini") == []
    assert _filter_by_relevance(weak, "local") != []


def test_an_unknown_embedder_falls_back_to_the_stricter_local_floor():
    assert _filter_by_relevance([{"text": "a", "score": 0.001}], "") == []


def test_empty_search_result_is_not_an_error(tmp_path):
    kb = KnowledgeBase("kb0", "Empty", str(tmp_path / "kb0.pkl"))
    assert kb.search("anything") == []


def test_function_words_do_not_create_a_match(tmp_path):
    """"what was warehouse throughput in the third quarter" matched a
    passage about probation periods entirely on "the", "in" and "was" —
    enough to clear any floor low enough to admit real matches."""
    from app.rag.vector_store import _local_embed
    kb = _kb(tmp_path, HR_DOCS)
    q = _local_embed(["what was the throughput in the third quarter"])[0]
    sims = kb.vectors @ q
    assert float(np.max(sims)) < 0.05, \
        "an off-topic question still matches on function words: {}".format(
            np.round(sims, 3))


def test_a_plural_in_the_document_matches_a_singular_query(tmp_path):
    kb = _kb(tmp_path, HR_DOCS)
    hits = kb.search("expense claim deadline")
    assert hits, "'claim' did not match a passage about 'claims'"
    assert "expense" in hits[0]["text"].lower()


def test_stopword_only_queries_find_nothing(tmp_path):
    kb = _kb(tmp_path, HR_DOCS)
    assert kb.search("what is the of and to") == []


# ══════════════════════════════════════════════════════════
#  Answering
# ══════════════════════════════════════════════════════════

def test_an_empty_kb_says_so_without_calling_a_model(tmp_path, monkeypatch):
    from app.rag import service
    kb = KnowledgeBase("kb0", "Empty", str(tmp_path / "kb0.pkl"))

    def _boom(*a, **kw):
        raise AssertionError("a model was called for an empty knowledge base")
    monkeypatch.setattr(service, "_generate", _boom)

    out = service.answer_question(kb, "what is our leave policy")
    assert out["grounded"] is False
    assert out["sources"] == []


def test_an_uncovered_question_is_refused_without_calling_a_model(
        tmp_path, monkeypatch):
    """Not calling the model is the point: a model given unrelated context
    writes a confident answer from it every time."""
    from app.rag import service
    kb = _kb(tmp_path, HR_DOCS)

    def _boom(*a, **kw):
        raise AssertionError("a model was called with no relevant passages")
    monkeypatch.setattr(service, "_generate", _boom)

    out = service.answer_question(kb, "what was warehouse throughput in Q3")
    assert out["grounded"] is False
    assert "does not" in out["answer"].lower() or "nothing" in out["answer"].lower()
    assert out["sources"] == []


def test_a_covered_question_reaches_the_model_with_its_passages(
        tmp_path, monkeypatch):
    from app.rag import service
    kb = _kb(tmp_path, HR_DOCS)
    captured = {}

    def _fake(system, user, max_tokens=2048):
        captured["user"] = user
        captured["system"] = system
        return "Staff receive 25 days of annual leave [1]."
    monkeypatch.setattr(service, "_generate", _fake)

    out = service.answer_question(kb, "how much annual leave do staff get")
    assert out["grounded"] is True
    assert out["cited_sources"] == 1
    assert out["uncited"] is False
    assert "annual leave" in captured["user"].lower()
    assert "ONLY from the provided context" in captured["system"]


def test_an_answer_citing_nothing_is_flagged(tmp_path, monkeypatch):
    """An answer with no citation was not written from the passages
    supplied, whatever it says."""
    from app.rag import service
    kb = _kb(tmp_path, HR_DOCS)
    monkeypatch.setattr(
        service, "_generate",
        lambda *a, **kw: "Most companies offer around 20 days of leave.")

    out = service.answer_question(kb, "how much annual leave do staff get")
    assert out["uncited"] is True


@pytest.mark.parametrize("answer,expected", [
    ("Nothing here", 0),
    ("Revenue rose [1] and costs fell [2].", 2),
    ("Both points come from [1] and again [1].", 1),
])
def test_citation_counting(answer, expected):
    assert _citation_count(answer) == expected


def test_the_qa_prompt_forbids_outside_knowledge():
    from app.rag.service import QA_SYSTEM
    lower = QA_SYSTEM.lower()
    assert "only from the provided context" in lower
    assert "your own knowledge" in lower
    assert "never invent" in lower


# ══════════════════════════════════════════════════════════
#  Limits
# ══════════════════════════════════════════════════════════

def test_file_count_limit_is_enforced_with_a_useful_message(tmp_path):
    kb = KnowledgeBase("kb1", "K", str(tmp_path / "kb1.pkl"))
    kb.files = [{"filename": f"f{i}.pdf"} for i in range(config.rag_max_files_per_kb)]
    with pytest.raises(KbLimitError) as e:
        check_ingest_limits(kb, 1)
    assert str(config.rag_max_files_per_kb) in str(e.value)


def test_chunk_limit_refuses_rather_than_truncating(tmp_path):
    """Truncating leaves the user with a knowledge base that silently does
    not contain what they uploaded — and they will query it anyway."""
    kb = KnowledgeBase("kb1", "K", str(tmp_path / "kb1.pkl"))
    kb.chunks = [{"text": "x"}] * (config.rag_max_chunks_per_kb - 10)
    with pytest.raises(KbLimitError) as e:
        check_ingest_limits(kb, 500)
    msg = str(e.value)
    assert "nothing was added" in msg
    assert "10" in msg, "the message does not say how much room is left"


def test_an_ingest_within_the_limits_is_allowed(tmp_path):
    kb = KnowledgeBase("kb1", "K", str(tmp_path / "kb1.pkl"))
    check_ingest_limits(kb, 100)   # must not raise


def test_limits_are_configurable():
    for name in ("rag_max_kbs_per_owner", "rag_max_files_per_kb",
                 "rag_max_chunks_per_kb"):
        assert getattr(config, name) > 0


# ══════════════════════════════════════════════════════════
#  Chunking still behaves
# ══════════════════════════════════════════════════════════

def test_chunking_keeps_source_references():
    passages = [{"text": "word " * 2000, "source": "a.pdf", "locator": "p1"}]
    chunks = chunk_passages(passages)
    assert len(chunks) > 1
    assert all(c["source"] == "a.pdf" for c in chunks)
    assert all(c["locator"] == "p1" for c in chunks)


def test_chunking_drops_empty_passages():
    assert chunk_passages([{"text": "   ", "source": "a", "locator": "b"}]) == []
