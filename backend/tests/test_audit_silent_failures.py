"""
Silent failures and claims the data does not support.

Each test pins one defect found by running the engines on pure noise and on
awkward datasets with every swallowed exception made visible:

* a report section that failed was left out without a word;
* the EDA page lost its date columns, its whole multicollinearity table (one
  empty column was enough) and every normal fit (a SciPy call that always
  raised) — all logged at debug level only;
* the best of up to 30 segment tests, unadjusted, was the report's headline
  on random data, and other test families were judged pair by pair;
* an untested gap was called "statistically significant";
* the dashboard named a "largest" group, a trend and a correlation that
  chance alone produces;
* a correlation's label quoted a different p from the one it was judged on;
* every row was called a "product".
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pypdf
import pytest


def _noise(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "region": rng.choice(list("NSEW"), n),
        "plan": rng.choice(["basic", "pro", "team"], n),
        "revenue": rng.normal(1000, 200, n).round(2),
        "cost": rng.normal(600, 120, n).round(2),
        "tenure": rng.integers(1, 60, n),
        "visits": rng.poisson(5, n),
        "churned": rng.choice(["Yes", "No"], n, p=[.25, .75]),
        "date": pd.date_range("2022-01-01", periods=n, freq="D"),
    })


def _upload(client, df, name="data.csv"):
    r = client.post("/api/datasets/upload",
                    files={"file": (name, io.BytesIO(df.to_csv(index=False).encode()),
                                    "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["meta"]["dataset_id"]


def _pdf_text(pdf: bytes) -> str:
    return "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(io.BytesIO(pdf)).pages)


# ── Reports say what they left out ───────────────────────────────────────

def test_a_failed_report_section_is_named_in_the_pdf(client, monkeypatch):
    import app.engines.story_engine as story_engine

    def boom(df):
        raise RuntimeError("story failed")
    monkeypatch.setattr(story_engine, "generate_story", boom)
    ds = _upload(client, _noise())
    r = client.post(f"/api/reports/{ds}/pdf", json={})
    assert r.status_code == 200, r.text
    text = " ".join(_pdf_text(r.content).split())
    assert "Not in this report" in text
    assert "summary, risks and next steps could not be produced" in text


def test_a_requested_model_section_without_a_model_is_explained(client):
    ds = _upload(client, _noise())
    text = " ".join(_pdf_text(client.post(f"/api/reports/{ds}/pdf",
                                          json={"include_ml": True}).content).split())
    assert "No prediction model has been trained on this data" in text


def test_a_complete_report_has_no_omissions_box(client):
    ds = _upload(client, _noise())
    text = _pdf_text(client.post(f"/api/reports/{ds}/pdf", json={}).content)
    assert "Not in this report" not in text


def test_the_omissions_box_is_translated(client, monkeypatch):
    import app.engines.story_engine as story_engine
    monkeypatch.setattr(story_engine, "generate_story",
                        lambda df: (_ for _ in ()).throw(RuntimeError("x")))
    from app import i18n
    ds = _upload(client, _noise())
    i18n.clear_misses()
    r = client.post(f"/api/reports/{ds}/pdf", json={"language": "hi"})
    assert r.status_code == 200
    text = " ".join(_pdf_text(r.content).split())
    assert "Not in this report" not in text and "could not be produced" not in text
    assert not any("report" in k or "produced" in k for k in i18n.misses())


def test_the_health_preview_says_when_its_findings_failed(client, monkeypatch):
    import app.engines.health_engine as health_engine

    def boom(df, domain):
        raise RuntimeError("payload failed")
    monkeypatch.setattr(health_engine, "build_report_payload", boom)
    ds = _upload(client, _noise())
    body = client.get(f"/api/reports/{ds}/health").json()
    assert body["unavailable"]
    assert body["insights"] == []


def test_the_health_preview_is_quiet_when_it_works(client):
    ds = _upload(client, _noise())
    assert client.get(f"/api/reports/{ds}/health").json()["unavailable"] == ""


# ── EDA: nothing dropped quietly ─────────────────────────────────────────

def _eda_frame(n=300):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "normal": rng.normal(size=n), "uniform": rng.uniform(size=n),
        "when": pd.date_range("2020-01-01", periods=n), "empty": np.nan, "const": 5.0,
    })


def test_normal_data_is_fitted_as_normal():
    from app.engines.eda_engine import run_eda
    r = run_eda(_eda_frame())
    assert r.univariate["normal"].best_fit_dist == "norm"
    assert r.univariate["uniform"].best_fit_dist == "uniform"


def test_no_distribution_is_named_when_every_fit_is_rejected():
    from app.engines.eda_engine import _fit_distribution
    rng = np.random.default_rng(2)
    bimodal = pd.Series(np.concatenate([rng.normal(-6, 1, 2000), rng.normal(6, 1, 2000)]))
    assert _fit_distribution(bimodal) == (None, {})


def test_date_and_constant_columns_are_described_not_dropped():
    from app.engines.eda_engine import run_eda
    u = run_eda(_eda_frame()).univariate
    assert u["when"].interpretation.startswith("Dates from 2020-01-01")
    assert "Every value is 5" in u["const"].interpretation


def test_columns_without_variation_get_no_correlation():
    from app.engines.eda_engine import run_eda
    pairs = {(c.col_a, c.col_b) for c in run_eda(_eda_frame()).correlations}
    assert pairs == {("normal", "uniform")}


def test_an_empty_column_does_not_remove_the_multicollinearity_table():
    from app.engines.eda_engine import run_eda
    vif = {m.feature: m.verdict for m in run_eda(_eda_frame()).multicollinearity}
    assert set(vif) == {"normal", "uniform"}          # empty and constant left out
    assert all(v == "OK" for v in vif.values())


# ── Families of tests are corrected; untested is not significant ─────────

def test_segment_headline_is_rare_on_random_data():
    from app.engines.domains.general import _best_segment_difference
    found = 0
    for seed in range(25):
        df = _noise(seed=seed)
        nums = ["revenue", "cost", "tenure", "visits"]
        found += _best_segment_difference(df, nums) is not None
    assert found <= 1, found


def test_a_real_segment_gap_is_still_found_and_reports_its_adjustment():
    from app.engines.domains.general import _best_segment_difference
    df = _noise(seed=3)
    df.loc[df.region == "N", "revenue"] *= 1.6
    seg = _best_segment_difference(df, ["revenue", "cost", "tenure", "visits"])
    assert seg and seg["cat"] == "region" and seg["num"] == "revenue"
    assert seg["m"] > 1


@pytest.mark.parametrize("domain", ["sales", "ecommerce"])
def test_an_untested_gap_is_not_called_significant(monkeypatch, domain):
    from scipy import stats as sc
    from app.engines.domains.base import col_stats

    def boom(*a, **k):
        raise ValueError("test could not run")
    monkeypatch.setattr(sc, "kruskal", boom)
    rng = np.random.default_rng(4)
    n = 300
    if domain == "sales":
        from app.engines.domains.sales import _insights_sales as run
        df = pd.DataFrame({"region": rng.choice(["N", "S"], n),
                           "revenue": rng.gamma(2, 500, n)})
        df.loc[df.region == "N", "revenue"] *= 2
    else:
        from app.engines.domains.ecommerce import _insights_ecommerce as run
        df = pd.DataFrame({"category": rng.choice(["Home", "Toys"], n),
                           "rating": rng.integers(1, 6, n).astype(float)})
        df.loc[df.category == "Home", "rating"] = 5.0
    stats = {c: col_stats(df[c]) for c in df.select_dtypes("number").columns}
    out = run(df, stats, [])
    text = " ".join(str(v) for v in out["findings"] + out["risks"]) + " ".join(
        str(i) for i in out["insights"])
    assert "statistically significant" not in text
    assert "unlikely to be chance" not in text
    assert ("Regional Gap" if domain == "sales" else "Category Gap") not in text


def test_survival_pairs_are_adjusted_together():
    from app.engines.survival_engine import run_survival_analysis
    df = _noise(600, seed=3)
    rep = run_survival_analysis(df, "tenure", "churned", group_col="plan")
    assert len(rep.pairwise_comparisons) == 2
    assert all("adjusted p=" in c.verdict for c in rep.pairwise_comparisons)


def test_bi_cohorts_are_adjusted_as_a_family():
    from app.engines.bi_engine import run_bi
    rep = run_bi(_noise(seed=5))
    tested = [c for c in rep.cohorts if c.test_used != "N/A"]
    assert len(tested) > 1
    assert all("adjusted p=" in c.interpretation for c in tested)


# ── Dashboard insights ───────────────────────────────────────────────────

def test_dashboard_names_no_leader_trend_or_correlation_in_noise():
    from app.engines.insight_engine import generate_insights
    titles = " | ".join(i["title"] for i in generate_insights(_noise(600, seed=3)))
    assert "largest total" not in titles
    assert "later half" not in titles
    assert "Strong correlation" not in titles
    assert "spread about evenly" in titles


def test_dashboard_still_names_a_real_leader_and_correlation():
    from app.engines.insight_engine import generate_insights
    rng = np.random.default_rng(6)
    n = 400
    x = rng.normal(size=n)
    df = pd.DataFrame({"region": rng.choice(list("NSEW"), n, p=[.55, .15, .15, .15]),
                       "a": x * 10 + 100, "b": x * 5 + rng.normal(scale=.5, size=n) + 50})
    titles = " | ".join(i["title"] for i in generate_insights(df))
    assert "'N' has the largest total" in titles
    assert "Strong correlation" in titles


def test_a_correlation_on_a_handful_of_rows_is_not_reported():
    from app.engines.insight_engine import generate_insights
    df = pd.DataFrame({"a": [1.0, 2, 3, 4, 5], "b": [2.0, 1, 4, 3, 5]})   # r = 0.8, n = 5
    assert not any("correlation" in i["title"] for i in generate_insights(df))


# ── Labels agree with their numbers ──────────────────────────────────────

def test_correlation_labels_quote_the_adjusted_p_they_were_judged_on():
    from app.engines.stats_engine import analyze
    rep = analyze(_noise())
    assert rep.correlations
    for c in rep.correlations:
        assert "adjusted p={:.4f}".format(c.p_value) in c.label
        assert c.label.startswith("No reliable") != c.is_significant


def test_ratings_are_counted_as_ratings_not_products():
    from app.engines.health_engine import build_report_payload
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"product": rng.choice(list("ABCDE"), n),     # 5 products, 400 rows
                       "rating": rng.normal(3.0, 0.6, n).clip(1, 5).round(1),
                       "revenue": rng.normal(100, 20, n)})
    text = str(build_report_payload(df, "ecommerce"))
    assert "of 400 products" not in text and "of 400 ratings" in text


# ── ML ───────────────────────────────────────────────────────────────────

def test_gradient_boosting_trains_on_small_multiclass_data():
    from app.engines.ml_engine import run_ml_pipeline
    rng = np.random.default_rng(1)
    n = 60
    df = pd.DataFrame({"product": rng.choice(list("ABCDE"), n),
                       "revenue": rng.normal(size=n), "cost": rng.normal(size=n)})
    rep = run_ml_pipeline(df, "product")
    assert {m.name: m.error for m in rep.models}["Gradient Boosting"] == ""
