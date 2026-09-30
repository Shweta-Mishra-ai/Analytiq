"""
Regression tests for the deep-audit findings.

Each test fails against the pre-fix code and passes after. They are grouped
by the finding id from the audit (C1, C2, H1–H6, M1) so a failure points
straight at what regressed.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest


# ── C1: SPA fallback must not serve files outside the static root ──────────
def test_c1_static_path_containment():
    from app.main import _safe_static_file, _static_root

    # A real file inside the static root resolves.
    inside = os.path.join(_static_root, "index.html")
    if os.path.isfile(inside):
        assert _safe_static_file("index.html") == os.path.realpath(inside)

    # Every traversal attempt must be refused (None), never resolved to a
    # path outside the static root.
    for attack in ["../app/config.py",
                   "../../etc/passwd",
                   "../../../../../../etc/passwd",
                   "..%2f..%2fapp%2fconfig.py",   # literal, not URL-decoded here
                   "../app/../../etc/passwd"]:
        got = _safe_static_file(attack)
        assert got is None or got.startswith(_static_root + os.sep), \
            f"path {attack!r} escaped the static root: {got!r}"


# ── H1: privacy mode must not send data to a cloud LLM ─────────────────────
def test_h1_privacy_mode_blocks_groq(monkeypatch):
    from app.config import config
    from app.ai import local_llm
    from app.ai.llm_client import LLMClient

    monkeypatch.setattr(config, "llm_privacy_mode", True)
    assert local_llm.privacy_mode() is True

    client = LLMClient(api_key="unused-in-this-test")
    # chat() is the path AI-chat uses; it must refuse before any network call.
    with pytest.raises(local_llm.LocalLLMError):
        client.chat([{"role": "user", "content": "hello"}], system="rows here")


def test_h1_privacy_mode_blocks_gemini(monkeypatch):
    from app.config import config
    from app.ai import local_llm, gemini_client

    monkeypatch.setattr(config, "llm_privacy_mode", True)
    monkeypatch.setattr(config, "gemini_api_key", "fake-key")
    with pytest.raises(local_llm.LocalLLMError):
        gemini_client.get_client()


# ── H2: bool columns must stay in the attrition driver search ──────────────
def test_h2_overtime_survives_cleaning():
    from app.engines.data_cleaner import auto_clean
    from app.engines.domains.hr import _run_attrition

    rng = np.random.default_rng(0)
    n = 800
    ot = rng.random(n) < 0.3
    left = rng.random(n) < np.where(ot, 0.40, 0.12)
    df = pd.DataFrame({
        "EmployeeID": np.arange(n),
        "Department": rng.choice(["Sales", "R&D", "HR"], n),
        "OverTime":   np.where(ot, "Yes", "No"),
        "Attrition":  np.where(left, "Yes", "No"),
    })
    cleaned, _ = auto_clean(df)        # converts Yes/No → bool
    assert cleaned["OverTime"].dtype == bool

    result = _run_attrition(cleaned)
    factors = {d["factor"] for d in result.top_drivers}
    assert "OverTime" in factors, \
        "OverTime dropped out of the driver search after bool conversion"


# ── H3: bar/pie captions must aggregate the same way the chart does (sum) ──
def test_h3_bar_stats_uses_sum_matching_chart():
    from app.ai.report_narrator import _bar_stats

    # West has the most rows (highest total) but a lower per-row mean, so
    # sum and mean disagree on the leader — the exact case that made the
    # picture and its caption contradict each other.
    df = pd.DataFrame({
        "region":  ["West"] * 100 + ["East"] * 10,
        "revenue": [100] * 100 + [500] * 10,   # West sum 10000 > East 5000
    })
    s = _bar_stats(df, "region", "revenue")
    assert s["ok"]
    assert s["top"] == "West", "caption leader disagrees with the summed bar chart"
    assert s["top_val"] == pytest.approx(10000)


def test_h3_pie_shares_use_sum():
    from app.ai.report_narrator import _pie_stats
    df = pd.DataFrame({
        "region":  ["West"] * 100 + ["East"] * 10,
        "revenue": [100] * 100 + [500] * 10,
    })
    s = _pie_stats(df, "region", "revenue")
    assert s["ok"]
    # West holds 10000 / 15000 of the total.
    assert s["shares"]["West"] == pytest.approx(66.7, abs=0.2)


# ── H4: an "Over Time" chart caption must describe that chart ──────────────
def test_h4_over_time_narrative_is_not_the_wrong_column():
    from app.ai.report_narrator import generate_chart_narrative
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({
        "order_id":   np.arange(100000, 100000 + n),
        "order_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "region":     rng.choice(["N", "S", "E", "W"], n),
        "revenue":    rng.normal(1000, 200, n).round(2),
    })
    text = generate_chart_narrative(df, "revenue Over Time").lower()
    assert "order id" not in text and "order_id" not in text, \
        "time-series caption fell through to the ID column"
    assert "revenue" in text


# ── H5: the health grade must be capped when data isn't fit to analyse ─────
def test_h5_health_grade_capped_for_unfit_data():
    from app.engines.health_engine import compute_health
    # Complete, no duplicates, but revenue is stored as text → a readiness
    # blocker. The grade must not read "Excellent".
    df = pd.DataFrame({
        "region":  ["N", "S", "E", "W"] * 60,
        "revenue": [f"${v}" for v in range(1000, 1240)],
    })
    health = compute_health(df)
    assert health["ready"] is False
    assert health["blockers"], "blocker not surfaced on health payload"
    assert health["score"] <= 69, "grade not capped for unfit data"
    assert health["grade"] not in ("A+", "A")


# ── H6: auto-clean must coerce currency/number-as-text and dates ───────────
def test_h6_currency_text_coerced_to_numeric():
    from app.engines.data_cleaner import auto_clean
    df = pd.DataFrame({
        "product": [f"P{i%5}" for i in range(120)],
        "price":   [f"${1000 + i:,}.50" for i in range(120)],
    })
    cleaned, report = auto_clean(df)
    assert pd.api.types.is_numeric_dtype(cleaned["price"]), \
        "currency column left as text"
    assert cleaned["price"].iloc[0] == pytest.approx(1000.50)
    assert any("stored as text" in a.issue for a in report.actions)


def test_h6_date_text_parsed():
    from app.engines.data_cleaner import auto_clean
    df = pd.DataFrame({
        "order_date": pd.date_range("2024-01-01", periods=120).strftime("%Y-%m-%d"),
        "amount":     np.arange(120.0),
    })
    cleaned, _ = auto_clean(df)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["order_date"]), \
        "date column left as text"


# ── M1: the analysis cache must invalidate on any content change ───────────
def test_m1_cache_invalidates_beyond_row_100(tmp_path):
    from app.services.dataset_store import DatasetStore
    store = DatasetStore(str(tmp_path / "ds"))
    df = pd.DataFrame({"a": np.arange(1000.0), "b": ["x"] * 1000})
    meta = store.create("u", df, "t.csv", 0.1)
    store.cache_set("u", meta.dataset_id, "stats", {"sum_a": float(df.a.sum())})

    df2 = df.copy()
    df2.loc[500:, "a"] = 0.0          # change only rows past 100
    store.update_active("u", meta.dataset_id, df2)

    assert store.cache_get("u", meta.dataset_id, "stats") is None, \
        "stale cache served after a change beyond row 100"


# ── M5: an unreadable account store must fail closed, not open auth ────────
def test_m5_corrupt_user_store_fails_closed(tmp_path):
    from app.services.user_store import UserStore, UserStoreLoadError
    p = tmp_path / "users.json"
    p.write_text("{ this is not valid json")
    with pytest.raises(UserStoreLoadError):
        UserStore(str(p))


# ── ML pipeline: date/ID targets, XGBoost on integer classes ───────────────
def test_ml_targets_skip_dates_and_ids():
    from app.engines.ml_engine import suggest_targets
    import pandas as pd, numpy as np
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "order_id":   np.arange(n),
        "order_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "region":     rng.choice(["N", "S", "E", "W"], n),
        "revenue":    rng.normal(100, 20, n),
    })
    cols = [t["column"] for t in suggest_targets(df)]
    assert "order_id" not in cols, "ID offered as ML target"
    assert "order_date" not in cols, "datetime offered as ML target"
    assert "region" in cols and "revenue" in cols


def test_ml_targets_no_crash_on_datetime_column():
    # A parsed datetime column used to crash suggest_targets with
    # abs(Timestamp). It must return cleanly now.
    from app.engines.ml_engine import suggest_targets
    import pandas as pd, numpy as np
    df = pd.DataFrame({
        "ts":     pd.date_range("2024-01-01", periods=120, freq="h"),
        "amount": np.arange(120.0),
        "grp":    (["a", "b", "c"] * 40),
    })
    out = suggest_targets(df)          # must not raise
    assert all(s["column"] != "ts" for s in out)


def test_ml_xgboost_trains_on_integer_classes():
    from app.engines.ml_engine import run_ml_pipeline
    import pandas as pd, numpy as np
    rng = np.random.default_rng(1)
    n = 400
    x1 = rng.normal(0, 1, n)
    # integer class labels 1..4 (not 0-indexed) — the case XGBoost rejected
    y = np.clip((x1 * 1.5 + rng.normal(0, 0.5, n)).round().astype(int), 1, 4)
    df = pd.DataFrame({"f1": x1, "f2": rng.normal(0, 1, n), "rating": y})
    report = run_ml_pipeline(df, "rating")
    names = [m.name for m in report.models]
    assert "XGBoost" in names, "XGBoost was dropped on integer class target"
