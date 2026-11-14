"""
Faults found by checking the app's answers against files whose answers are
known, and by sending it what real clients and hostile callers send. Each
test pins one fault so it cannot come back.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from app.config import config


def _load(raw: bytes, name: str) -> pd.DataFrame:
    from app.engines.data_loader import load_file
    f = io.BytesIO(raw)
    f.name, f.size = name, len(raw)
    r = load_file(f)
    assert r.success, r.error
    return r.df


# ── reading files ────────────────────────────────────────

def test_a_money_column_with_blank_cells_is_read_as_money():
    rows = ['"$1,200.50",01/02/2024', ",", '$30.00,15/03/2024', '"$4,000",20/04/2024', ",", "$12,30/05/2024"]
    df = _load(("Amount ($),Order Date\n" + "\n".join(rows * 10)).encode(), "m.csv")
    assert pd.api.types.is_float_dtype(df["Amount ($)"])
    assert df["Amount ($)"].sum() == pytest.approx((1200.5 + 30 + 4000 + 12) * 10)
    assert pd.api.types.is_datetime64_any_dtype(df["Order Date"])


def test_a_word_among_numbers_is_still_kept():
    df = _load(("Status\n" + "\n".join(["10", "12", "Pending", "13", "14", "15"] * 10)).encode(), "s.csv")
    assert "Pending" in set(df["Status"].astype(str))


def test_tab_separated_files_load():
    tsv = "region\tsales\n" + "\n".join(f"{r}\t{i}" for i, r in enumerate(["N", "S"] * 20))
    for name in ("t.tsv", "t.txt"):
        df = _load(tsv.encode(), name)
        assert list(df.columns) == ["region", "sales"] and len(df) == 40


# ── figures ──────────────────────────────────────────────

def test_average_discount_is_the_plain_average_a_spreadsheet_gives():
    # It was weighted by sales after the discount, which shrinks exactly the
    # deeply discounted orders: 6.8% shown where AVERAGE() gives 7.5%.
    from app.engines.scorecard import _periods
    rng = np.random.default_rng(0)
    n = 600
    disc = rng.choice([0.0, 0.1, 0.2, 0.5], n)
    df = pd.DataFrame({"order_date": pd.date_range("2025-01-01", periods=n, freq="D"),
                       "order_id": [f"O{i}" for i in range(n)],
                       "sales": (rng.uniform(50, 500, n) * (1 - disc)).round(2), "discount": disc})
    current = _periods(df["order_date"])[0]
    expected = df.loc[current, "discount"].mean() * 100
    card = [c for c in _cards(df) if c["label"] == "Average discount"][0]
    assert card["value"] == pytest.approx(expected, abs=0.05)


def _cards(df):
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    buf = io.BytesIO(df.to_csv(index=False).encode())
    ds = c.post("/api/datasets/upload", files={"file": ("d.csv", buf, "text/csv")}).json()["meta"]["dataset_id"]
    try:
        return c.post(f"/api/charts/{ds}/kpis", json={}).json()["kpis"]
    finally:
        c.delete(f"/api/datasets/{ds}")


def test_drivers_do_not_rank_noise_by_its_number_of_values():
    # Attrition driven by overtime and satisfaction; income and age are noise
    # with many distinct values, which the forest's split counts favoured.
    from app.engines.predictive import compute_drivers
    rng = np.random.default_rng(42)
    n = 1500
    df = pd.DataFrame({"age": rng.integers(21, 60, n), "income": rng.normal(6000, 1800, n).round(0),
                       "overtime": rng.choice(["Yes", "No"], n, p=[.3, .7]),
                       "satisfaction": rng.integers(1, 5, n)})
    logit = -2.2 + 1.6 * (df.overtime == "Yes") - 0.55 * (df.satisfaction - 2.5)
    df["left"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    r = compute_drivers(df, "left")
    share = dict(r.top_drivers)
    assert [c for c, _ in r.top_drivers[:2]] == ["overtime", "satisfaction"]
    assert share["income"] < 5 and share["age"] < 5


# ── what the API accepts ─────────────────────────────────

def test_a_blank_or_path_like_dataset_id_is_not_found(client):
    ds = client.post("/api/datasets/upload",
                     files={"file": ("a.csv", io.BytesIO(b"a,b\n1,2\n3,4\n5,6\n"), "text/csv")}).json()["meta"]["dataset_id"]
    try:
        for other in ("", "../../etc/passwd", "..", "a/b"):
            r = client.post(f"/api/analytics/{ds}/compare", json={"other_dataset_id": other})
            assert r.status_code == 404, (other, r.status_code)
    finally:
        client.delete(f"/api/datasets/{ds}")
    from app.services.dataset_store import store
    assert store.get_df("local", "..") is None and store.delete("local", "") is False


def test_chat_refuses_blank_and_oversized_messages_before_any_model(client):
    ds = client.post("/api/datasets/upload",
                     files={"file": ("a.csv", io.BytesIO(b"a,b\n1,2\n3,4\n5,6\n"), "text/csv")}).json()["meta"]["dataset_id"]
    try:
        for msg in ("", "   ", "x" * 4001):
            assert client.post(f"/api/chat/{ds}", json={"message": msg}).status_code == 422
        too_long = [{"role": "user", "content": "hi"}] * 51
        assert client.post(f"/api/chat/{ds}", json={"message": "hi", "history": too_long}).status_code == 422
    finally:
        client.delete(f"/api/datasets/{ds}")


def test_without_a_model_chat_says_so_in_words_a_client_can_act_on(client, monkeypatch):
    from app.api import chat
    monkeypatch.setattr(chat, "chat_available", lambda: False)
    assert client.get("/api/health").json()["chat_available"] is False
    for key in ("anthropic_api_key", "groq_api_key", "local_llm_url"):
        monkeypatch.setattr(config, key, "")
    ds = client.post("/api/datasets/upload",
                     files={"file": ("a.csv", io.BytesIO(b"a,b\n1,2\n3,4\n5,6\n"), "text/csv")}).json()["meta"]["dataset_id"]
    try:
        r = client.post(f"/api/chat/{ds}", json={"message": "What is the total?"})
        assert r.status_code == 503
        assert "not switched on" in r.json()["detail"] and "API_KEY" not in r.json()["detail"]
    finally:
        client.delete(f"/api/datasets/{ds}")


def test_an_invented_forwarded_address_does_not_escape_the_rate_limit():
    from starlette.requests import Request
    from app.main import _caller_key

    def key(peer, fwd=None):
        headers = [(b"x-forwarded-for", fwd.encode())] if fwd else []
        return _caller_key(Request({"type": "http", "headers": headers, "client": (peer, 1)}))
    # Straight from the internet: the header is the caller's to write.
    assert key("93.184.216.34", "1.2.3.4") == key("93.184.216.34", "5.6.7.8") == "ip:93.184.216.34"
    # Through a proxy on a private network: the address it appended.
    assert key("10.0.0.5", "1.2.3.4, 198.51.100.7") == "ip:198.51.100.7"
    assert key("10.0.0.5", "9.9.9.9, 198.51.100.7") == "ip:198.51.100.7"


# ── cached results ───────────────────────────────────────

def test_cached_results_are_dropped_when_the_code_changes(client, monkeypatch):
    from app.services import dataset_store as ds_mod
    from app.services.dataset_store import store
    ds = client.post("/api/datasets/upload",
                     files={"file": ("a.csv", io.BytesIO(b"a,b\n1,2\n3,4\n5,6\n"), "text/csv")}).json()["meta"]["dataset_id"]
    try:
        store.cache_set("local", ds, "probe", {"answer": 1})
        assert store.cache_get("local", ds, "probe") == {"answer": 1}
        monkeypatch.setattr(ds_mod, "CODE_VERSION", "a-later-release")
        assert store.cache_get("local", ds, "probe") is None
    finally:
        client.delete(f"/api/datasets/{ds}")


# ── knowledge bases ──────────────────────────────────────

def test_a_file_uploaded_again_replaces_its_earlier_copy(tmp_path):
    from app.rag.service import ingest_file
    from app.rag.vector_store import KnowledgeBase
    kb = KnowledgeBase("kb1", "K", str(tmp_path / "kb1.pkl"))
    ingest_file(kb, "policy.txt", b"Refunds are accepted within 30 days of purchase.")
    ingest_file(kb, "other.txt", b"Deliveries take three to five working days.")
    r = ingest_file(kb, "policy.txt", b"Refunds are accepted within 45 days of purchase.")
    assert r["replaced"] is True
    texts = [c["text"] for c in kb.chunks]
    assert len(kb.files) == 2 and len(kb.chunks) == len(kb.vectors) == 2
    assert any("45 days" in t for t in texts) and not any("30 days" in t for t in texts)
    assert kb.search("how many days for a refund", k=3)[0]["text"].count("45") == 1


def test_a_report_on_documents_the_general_searches_miss_is_still_written(tmp_path, monkeypatch):
    from app.rag import service
    from app.rag.vector_store import KnowledgeBase
    kb = KnowledgeBase("kb1", "K", str(tmp_path / "kb1.pkl"))
    service.ingest_file(kb, "refunds.txt", b"Furniture may be returned within thirty days. "
                                           b"Electronics carry a two-year warranty.")
    monkeypatch.setattr(service, "_generate", lambda system, user, max_tokens=0: user)
    out = service.generate_report(kb, "Policies")
    assert "thirty days" in out["markdown"] and out["sources"]
    assert "## Sources" in out["markdown"] and "refunds.txt" in out["markdown"]
    with pytest.raises(ValueError, match="Nothing in this knowledge base is about"):
        service.generate_report(kb, "Policies", focus="quarterly churn by cohort")


def test_without_a_model_a_report_quotes_the_documents_instead_of_failing(tmp_path, monkeypatch):
    from app.rag import service
    from app.rag.vector_store import KnowledgeBase
    for key in ("anthropic_api_key", "groq_api_key", "gemini_api_key", "local_llm_url"):
        monkeypatch.setattr(config, key, "")
    kb = KnowledgeBase("kb1", "K", str(tmp_path / "kb1.pkl"))
    service.ingest_file(kb, "refunds.txt", b"Furniture may be returned within thirty days. "
                                           b"Electronics carry a two-year warranty.")
    out = service.generate_report(kb, "Policies")
    assert out["extractive"] is True
    assert "> Furniture may be returned within thirty days." in out["markdown"]
    assert "API_KEY" not in out["markdown"] and "quotes your documents" in out["markdown"]
    assert service.report_pdf(out["markdown"], "Policies").startswith(b"%PDF")
    answer = service.answer_question(kb, "how long is the warranty?")["answer"]
    assert "two-year warranty" in answer and "API_KEY" not in answer
