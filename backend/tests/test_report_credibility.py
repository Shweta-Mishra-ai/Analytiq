"""
Guards on the things that get a report rejected on sight.

Every check here corresponds to a defect found in a real generated
report: charts plotted on identifier columns, HR benchmark bodies cited
on a finance deliverable, the LLM named in the methodology appendix, and
internal working notes printed in the client's copy.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def finance_df():
    rng = np.random.default_rng(11)
    n = 480
    cat = rng.choice(["Ops", "Marketing", "R&D", "Admin"], n, p=[.4, .25, .2, .15])
    revenue = (np.linspace(100_000, 70_000, n) + rng.normal(0, 6000, n)).round(2)
    cost = (revenue * np.where(cat == "Marketing", 0.95, 0.62)
            + rng.normal(0, 3000, n)).round(2)
    return pd.DataFrame({
        "invoice_id": range(n),                      # identifier, must never be charted
        "period": pd.date_range("2024-01-01", periods=n),
        "category": cat,
        "revenue": revenue,
        "cost": cost,
        "profit": (revenue - cost).round(2),
        "budget": np.full(n, 85_000.0),              # constant, useless to chart
        "opex": (cost * 0.4).round(2),
    })


# ══════════════════════════════════════════════════════════
#  Charts must plot measures, never identifiers
# ══════════════════════════════════════════════════════════

def test_charts_never_plot_an_identifier_column(finance_df):
    """A finance report shipped with "invoice_id by category", a
    distribution of invoice_id, and a pie chart of summed invoice IDs —
    three of five charts on an identifier. Any analyst rejects that."""
    from app.engines.chart_exporter import generate_all_charts
    titles = [t for t, _ in generate_all_charts(finance_df, max_charts=5)]
    assert titles, "no charts produced at all"
    for t in titles:
        assert "invoice_id" not in t.lower(), f"identifier charted: {t}"


def test_chart_measure_ranking_prefers_business_metrics(finance_df):
    from app.engines.chart_exporter import _rank_measures
    ranked = _rank_measures(
        finance_df, finance_df.select_dtypes(include="number").columns.tolist())
    assert "invoice_id" not in ranked, "identifier survived measure ranking"
    assert "budget" not in ranked, "constant column survived measure ranking"
    assert ranked, "no measures ranked"
    assert set(ranked) <= {"revenue", "cost", "profit", "opex"}


def test_charts_return_empty_rather_than_meaningless(monkeypatch):
    """If the only numeric columns are identifiers, produce no charts —
    better than filling a client report with nonsense."""
    from app.engines.chart_exporter import generate_all_charts
    df = pd.DataFrame({
        "record_id": range(60),
        "customer_id": range(1000, 1060),
        "label": ["a", "b"] * 30,
    })
    assert generate_all_charts(df) == []


def test_time_series_chart_actually_renders(finance_df):
    """resample("M") was removed in pandas 3 and raised inside a
    try/except that logged at debug level — so the trend chart silently
    vanished from every report."""
    from app.engines.chart_exporter import make_line_chart
    png = make_line_chart(finance_df, "period", "revenue", "Revenue Trend")
    assert png[:4] == b"\x89PNG", "line chart did not render"
    assert len(png) > 5_000


def test_month_end_alias_is_valid_on_this_pandas():
    from app.services.dtypes import MONTH_END
    s = pd.Series([1.0, 2.0, 3.0],
                  index=pd.to_datetime(["2025-01-05", "2025-02-05", "2025-03-05"]))
    assert not s.resample(MONTH_END).mean().empty


def test_eda_time_series_section_runs(finance_df):
    """The same "M" alias broke Deep EDA's stationarity/trend section."""
    from app.engines.eda_engine import run_eda
    report = run_eda(finance_df)
    assert report.time_series, "time-series analysis produced nothing"


# ══════════════════════════════════════════════════════════
#  Nothing in the report may advertise how it was produced
# ══════════════════════════════════════════════════════════

def _main_pdf_text(df, client="Acme Corp", title="Q3 Review") -> str:
    import pypdf
    from app.engines.chart_exporter import generate_all_charts
    from app.engines.data_profiler import profile_dataset
    from app.engines.pdf_builder import build_pdf
    from app.engines.story_engine import detect_domain, generate_story

    domain, _ = detect_domain(df)
    story = generate_story(df)
    charts = [(t, b, "") for t, b in generate_all_charts(df, max_charts=3)]
    pdf = build_pdf(
        df=df,
        config={"title": title, "subtitle": "", "client_name": client,
                "confidential": True, "theme_name": "", "logo_path": None},
        profile=profile_dataset(df), cleaning_summary=None,
        stats_report=None, bi_report=None, ml_report=None, chart_data=charts,
        executive_summary=story.executive_summary, findings=story.key_findings,
        risks=story.business_risks, opportunities=story.opportunities,
        recommendations=story.recommended_actions, top_insights=[],
        attrition=None, domain=domain,
    )
    reader = pypdf.PdfReader(io.BytesIO(pdf))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def test_report_never_names_the_model_or_vendor(finance_df):
    """The methodology appendix printed "AI narratives: Groq Llama 3.3
    70B". A reader who sees that discounts every figure above it."""
    text = _main_pdf_text(finance_df)
    for banned in ("Groq", "Llama", "GPT", "OpenAI", "Gemini",
                   "AI narratives", "AI-generated", "language model"):
        assert banned.lower() not in text.lower(), \
            f"report discloses its tooling: {banned!r}"


def test_report_carries_no_internal_working_notes(finance_df):
    """"Verify with domain expert before client delivery" is an
    instruction to the analyst, not text for the client's copy."""
    text = _main_pdf_text(finance_df).lower()
    for banned in ("before client delivery", "todo", "fixme", "placeholder"):
        assert banned not in text, f"internal note leaked into the report: {banned!r}"


def test_finance_report_does_not_cite_hr_benchmark_bodies(finance_df):
    """The page footer named SHRM · Gallup · Mercer · Deloitte on every
    page of every report regardless of domain."""
    text = _main_pdf_text(finance_df)
    for hr_body in ("SHRM", "Gallup", "Mercer"):
        assert hr_body not in text, \
            f"finance report cites the HR benchmark body {hr_body!r}"


def test_finance_report_cites_finance_appropriate_sources(finance_df):
    text = _main_pdf_text(finance_df)
    assert "IFRS" in text or "CFA" in text, \
        "no domain-appropriate reference sources in the appendix"


def test_methodology_states_the_tests_actually_used(finance_df):
    """A senior reviewer reads this section to decide whether to trust the
    rest, so it must describe method, not tooling."""
    text = _main_pdf_text(finance_df)
    for expected in ("Shapiro", "Spearman", "IQR"):
        assert expected in text, f"methodology omits {expected}"
    assert "does not establish causation" in text.lower() \
        or "not establish causation" in text.lower()
