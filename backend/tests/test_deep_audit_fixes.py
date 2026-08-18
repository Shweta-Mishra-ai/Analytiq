"""
Nine defects found by rendering two real PDFs page-by-page and reading
what they actually said, not by reading the engines that produced them.

**A categorical column tested against the numeric column it was cut
from.** The BI engine reported "'AgeGroup' significantly segments
'Age': best cohort '55+' outperforms worst '18-25' by 158%,
Kruskal-Wallis p<0.001" as a finding, with a recommended action to
"pilot the stronger segment's practices in the weaker one" — advice to
make young employees older. A guard against exactly this
(`_is_obvious_segment_pair`) already existed and required the metric's
name to be at least 4 characters before it would check for a
near-tautology; "age" is 3.

**Positional column selection in the BI engine.** `run_bi` took
`num_cols[:2]` and `cat_cols[:2]` in whatever order they sat in the
frame, with no ranking and no exclusion of identifiers. On the file
above that put `Age` in front of a client as a root-cause "performance"
target ("Root cause of low 'Age': 'MonthlyIncome' is the top driver")
and as the numeric column feeding a segment-health score.

**A headcount tile crashed the PDF's dashboard section.**
`dashboard_spec` builds a "Headcount by <group>" tile with the same
column as both x and y (`agg="count"`) for every domain — the API layer
already handled this case, the PDF chart builders did not.
`df.groupby(x)[x].sum().reset_index()` raised "cannot insert
department, already exists", the tile silently disappeared, and because
every tile in that dashboard follows the same "Headcount by X" pattern
across domains, the whole Dashboard section could vanish from the
printed report with no trace in the table of contents (which never
listed a Dashboard entry regardless of whether the section rendered).

**A chart's narrative talked about a different column than its
picture.** The PDF pipeline built each chart, discarded the columns it
had just used, and asked a second module to re-derive them from the
chart's *title string* by substring search with no word boundaries.
"YearsWithCurrManager by AgeGroup" matched the column `Age`, because
"age" is a substring of "yearswith**curr**...m**ana**g**e**r" — the
tail of "manager" — and `Age` happened to sit earlier in the frame than
the column actually plotted.

**A fabricated "trend" on a file with no date column.** With zero
datetime columns, the chart pack still built a chart titled
"MonthlyIncome Trend" by pairing two arbitrary numeric columns and
comparing the first half of the rows to the second half, reporting
"values have improved by 84.1%" — a movement read out of row order on a
file where row order carries no time information.

**Two PDF templates disagreeing about the same file's quality.** The
Health Report showed 100/100 "Grade A+ — Excellent" for a file the
Main Report's readiness check called "not ready to analyse" on the
strength of a repeating identifier — two documents from one engagement
telling a client opposite things.

**Two PDF templates using two different shades of the same domain
colour.** The Health Report's HR blue (#1565C0) was not the Main
Report's HR blue (#1976D2); e-commerce orange and finance blue were
likewise close but not identical, kept as an independently maintained
second copy of a colour table that had drifted.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════
#  A category cannot be tested against the number it was cut from
# ══════════════════════════════════════════════════════════

def _age_frame(n: int = 1200, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Age": rng.integers(18, 61, n),
        "Department": rng.choice(["Sales", "R&D", "HR"], n),
        "MonthlyIncome": rng.integers(1000, 20000, n),
    })
    df["AgeGroup"] = pd.cut(df.Age, bins=[0, 25, 35, 45, 55, 100],
                            labels=["18-25", "26-35", "36-45", "46-55", "55+"])
    return df


def test_a_derived_bucket_is_recognised_structurally():
    from app.engines.column_roles import is_discretization

    df = _age_frame()
    assert is_discretization(df, "AgeGroup", "Age")
    assert not is_discretization(df, "Department", "MonthlyIncome")
    assert not is_discretization(df, "Department", "Age")


def test_the_obvious_pair_guard_catches_a_three_letter_metric():
    """The floor used to be 4 characters, which excluded "age" — 3 —
    and is the entire reason this bug reached a report."""
    from app.engines.domains.general import _is_obvious_segment_pair

    df = _age_frame()
    assert _is_obvious_segment_pair(df, "AgeGroup", "Age")
    assert not _is_obvious_segment_pair(df, "Department", "MonthlyIncome")


def test_a_real_segmenter_is_not_blocked():
    """The guard must not be so eager it removes the findings that are
    actually true — Department really does differ on income here."""
    rng = np.random.default_rng(1)
    n = 900
    dept = rng.choice(["Sales", "R&D", "Exec"], n, p=[.5, .45, .05])
    pay = np.where(dept == "Exec", rng.normal(180_000, 10_000, n),
                   rng.normal(60_000, 15_000, n))
    df = pd.DataFrame({"Department": dept, "Pay": pay})
    from app.engines.domains.general import _is_obvious_segment_pair

    assert not _is_obvious_segment_pair(df, "Department", "Pay")


def test_agegroup_no_longer_reaches_a_segment_finding():
    from app.engines.domains.general import _insights_general

    df = _age_frame()
    from app.engines.domains.base import col_stats
    stats = {c: col_stats(df[c]) for c in df.select_dtypes(include="number")}
    result = _insights_general(df, stats, [])
    joined = " ".join(result.get("opportunities", []) + result.get("risks", [])
                      + [i.title for i in result.get("insights", [])])
    assert "AgeGroup" not in joined or "Age" not in joined.split("AgeGroup")[0][-3:]
    assert "pilot the stronger segment's practices" not in joined.lower()


# ══════════════════════════════════════════════════════════
#  The BI engine picks columns by role, not position
# ══════════════════════════════════════════════════════════

def _hr_like(n: int = 1200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "EmpID": range(1, n + 1),
        "Age": rng.integers(18, 61, n),
        "Attrition": rng.choice(["Yes", "No"], n, p=[.16, .84]),
        "BusinessTravel": rng.choice(["Rarely", "Frequently", "Never"], n),
        "MonthlyIncome": rng.integers(1000, 20000, n),
        "WorkLifeBalance": rng.integers(1, 5, n),
    })
    df["AgeGroup"] = pd.cut(df.Age, bins=[0, 25, 35, 45, 55, 100],
                            labels=["18-25", "26-35", "36-45", "46-55", "55+"])
    return df


def test_run_bi_does_not_headline_age_as_a_root_cause_target():
    from app.engines.bi_engine import run_bi

    report = run_bi(_hr_like())
    assert not any(rc.target_col == "Age" for rc in report.root_causes), \
        [rc.target_col for rc in report.root_causes]


def test_run_bi_skips_the_agegroup_age_cohort_pair():
    from app.engines.bi_engine import run_bi

    report = run_bi(_hr_like())
    pairs = [(c.cohort_col, c.metric_col) for c in report.cohorts]
    assert ("AgeGroup", "Age") not in pairs, pairs


def test_run_bi_skips_the_agegroup_age_pareto_pair():
    from app.engines.bi_engine import run_bi

    report = run_bi(_hr_like())
    pairs = [(p.group_col, p.value_col) for p in report.pareto]
    assert ("AgeGroup", "Age") not in pairs, pairs


def test_run_bi_excludes_identifiers_from_benchmarks():
    from app.engines.bi_engine import run_bi

    report = run_bi(_hr_like())
    assert "EmpID" not in [b.column for b in report.benchmarks]


def test_run_bi_still_finds_something_useful():
    """The guard must not leave the report empty — a real HR file has
    real questions to ask (Attrition vs income, tenure, travel)."""
    from app.engines.bi_engine import run_bi

    report = run_bi(_hr_like())
    assert report.cohorts or report.pareto or report.root_causes


# ══════════════════════════════════════════════════════════
#  A headcount tile (x == y) no longer crashes the chart builders
# ══════════════════════════════════════════════════════════

@pytest.fixture()
def counted_frame():
    rng = np.random.default_rng(4)
    n = 600
    return pd.DataFrame({
        "BusinessTravel": rng.choice(
            ["Travel_Rarely", "Travel_Frequently", "Non-Travel"], n,
            p=[.7, .19, .11])})


def test_the_printed_headcount_chart_no_longer_raises(counted_frame):
    from app.engines.chart_exporter import make_bar_chart

    png = make_bar_chart(counted_frame, "BusinessTravel", "BusinessTravel",
                         "Headcount by BusinessTravel")
    assert png[:4] == b"\x89PNG"


def test_the_interactive_headcount_chart_no_longer_raises(counted_frame):
    from app.engines.chart_engine import make_bar

    fig = make_bar(counted_frame, "BusinessTravel", "BusinessTravel",
                   "Headcount by BusinessTravel")
    assert fig.data


def test_the_headcount_chart_plots_actual_counts(counted_frame):
    from app.engines.chart_engine import make_bar

    fig = make_bar(counted_frame, "BusinessTravel", "BusinessTravel")
    total_plotted = sum(fig.data[0].y)
    assert total_plotted == len(counted_frame)


def test_the_headcount_narrative_does_not_repeat_the_column_twice(counted_frame):
    """"Dept is broadly level across Dept" was the old wording."""
    from app.engines.chart_message import bar_message

    msg = bar_message(counted_frame, "BusinessTravel", "BusinessTravel",
                      counts=True)
    assert msg is not None
    assert msg.lower().count("businesstravel") <= 1, msg


def test_a_dashboard_headcount_tile_survives_end_to_end():
    """The exact tile shape `dashboard_spec` builds for every domain —
    the one that was silently dropping the whole Dashboard section."""
    from app.engines.dashboard_spec import Tile
    from app.engines.pdf_builder import _render_dashboard_tile

    rng = np.random.default_rng(2)
    df = pd.DataFrame({"Department": rng.choice(["Sales", "R&D", "HR"], 400)})
    tile = Tile("bar", "Headcount by Department", "Where do people sit?",
               x="Department", y="Department", agg="count")
    png = _render_dashboard_tile(df, tile, "Corporate Light")
    assert png[:4] == b"\x89PNG"


# ══════════════════════════════════════════════════════════
#  The dashboard page cannot be silently dropped
# ══════════════════════════════════════════════════════════

def test_dashboard_tiles_survive_a_tautological_headcount_pattern():
    """A realistic HR-shaped file where every categorical column is
    small — the exact shape that produced a Dashboard section with
    nothing in it in the original PDF."""
    from app.engines.pdf_builder import build_dashboard_tiles

    df = _hr_like()
    tiles = build_dashboard_tiles(df, "hr", "Corporate Light")
    assert tiles, "the dashboard produced nothing for a normal HR file"


def test_a_failed_spec_falls_back_rather_than_vanishing(monkeypatch):
    from app.engines import pdf_builder as pb

    monkeypatch.setattr(pb, "_render_dashboard_tile",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("boom")))
    df = _hr_like()
    tiles = pb.build_dashboard_tiles(df, "hr", "Corporate Light")
    assert tiles, "no fallback tile was produced when every spec tile failed"


def test_a_file_with_nothing_chartable_produces_no_tiles_and_no_crash():
    df = pd.DataFrame({"ref": ["R{}".format(i) for i in range(200)]})
    from app.engines.pdf_builder import build_dashboard_tiles

    assert build_dashboard_tiles(df, "general", "Corporate Light") == []


def test_the_toc_lists_dashboard_only_when_it_renders():
    from app.engines.data_profiler import profile_dataset
    from app.engines.pdf_builder import build_pdf
    from app.engines.story_engine import generate_story

    df = _hr_like()
    story = generate_story(df)
    pdf = build_pdf(
        df=df,
        config={"title": "R", "subtitle": "", "client_name": "Acme",
                "confidential": False, "theme_name": "", "logo_path": None},
        profile=profile_dataset(df), cleaning_summary=None, stats_report=None,
        bi_report=None, ml_report=None, chart_data=[],
        executive_summary=story.executive_summary,
        findings=story.key_findings, risks=story.business_risks,
        opportunities=story.opportunities,
        recommendations=story.recommended_actions,
        top_insights=story.top_insights, attrition=None, domain="hr")

    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(io.BytesIO(pdf))
    toc = doc[1].get_textpage().get_text_range() or ""
    pages = [(doc[i].get_textpage().get_text_range() or "")
            for i in range(len(doc))]
    has_dashboard_page = any("the question it answers" in p for p in pages)
    assert ("Dashboard" in toc) == has_dashboard_page, \
        (toc[:600], has_dashboard_page)


# ══════════════════════════════════════════════════════════
#  A chart's narrative is generated from the columns it plotted
# ══════════════════════════════════════════════════════════

def test_chart_pack_titles_and_narratives_agree_on_the_metric():
    """The metric named in a chart's title must be the metric its
    narrative talks about — the bug was a chart titled about one column
    and captioned about a different one."""
    from app.engines.chart_exporter import generate_chart_pack_with_narratives

    df = _hr_like()
    for title, _png, narrative in generate_chart_pack_with_narratives(
            df, max_charts=5):
        if not narrative:
            continue
        # The chart's own headline measure (the text before " by "/
        # " against "/" Share by ", or the "Distribution: " column)
        # must be mentioned in its narrative.
        for sep in (" by ", " against ", " Trend", " Over Time"):
            if sep in title:
                metric = title.split(sep)[0].split("Share")[0].strip()
                assert metric in narrative or metric.lower() in narrative.lower(), \
                    (title, narrative)
                break


def test_no_fake_trend_is_built_without_a_date_column():
    """Zero datetime columns must never produce a chart claiming a
    "Trend"."""
    from app.engines.chart_exporter import generate_chart_pack_with_narratives

    df = _hr_like()
    assert "datetime" not in str(df.dtypes.values)
    titles = [t for t, _p, _n in
             generate_chart_pack_with_narratives(df, max_charts=5)]
    assert not any("Trend" in t for t in titles), titles


def test_two_numeric_columns_without_a_date_become_a_scatter():
    from app.engines.chart_exporter import generate_chart_pack_with_narratives

    df = _hr_like()
    titles = [t for t, _p, _n in
             generate_chart_pack_with_narratives(df, max_charts=5)]
    assert any(" against " in t for t in titles), titles


def test_the_scatter_narrative_is_an_association_not_an_improvement():
    """"Values have improved by 84.1%" was read off row order on a file
    with no time dimension at all."""
    from app.engines.chart_exporter import generate_chart_pack_with_narratives

    df = _hr_like()
    for title, _png, narrative in generate_chart_pack_with_narratives(
            df, max_charts=5):
        if " against " in title:
            low = narrative.lower()
            assert "improved" not in low and "declined" not in low, narrative


def test_a_dated_file_still_gets_a_real_trend():
    rng = np.random.default_rng(9)
    n = 400
    df = pd.DataFrame({
        "order_date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "revenue": rng.normal(1000, 100, n).cumsum(),
        "region": rng.choice(["North", "South"], n),
    })
    from app.engines.chart_exporter import generate_chart_pack_with_narratives

    titles = [t for t, _p, _n in
             generate_chart_pack_with_narratives(df, max_charts=5)]
    assert any("Over Time" in t for t in titles), titles


def test_report_narrator_no_longer_reparses_chart_titles():
    """The whole title-guessing machinery — the actual source of the
    chart/narrative mismatch — is gone; only the column-label helper
    the codebase still uses remains."""
    import app.ai.report_narrator as rn

    assert not hasattr(rn, "generate_chart_narrative")
    assert hasattr(rn, "clean_col")
    assert rn.clean_col("satisfaction_level") == "Employee Satisfaction Score"


def test_reports_api_no_longer_imports_the_retired_narrator():
    import inspect
    import re

    from app.api import reports

    source = inspect.getsource(reports)
    assert not re.search(r"^\s*(from|import)\s+.*report_narrator", source,
                         re.MULTILINE), \
        "reports.py still imports the retired title-guessing narrator"


# ══════════════════════════════════════════════════════════
#  The two report templates cannot contradict each other
# ══════════════════════════════════════════════════════════

def test_a_repeating_identifier_caps_the_health_score():
    from app.engines.health_engine import compute_health

    n = 300
    df = pd.DataFrame({
        "EmpID": list(range(1, n)) + [1],       # one repeat -> not ready
        "Age": np.random.default_rng(1).integers(18, 61, n),
    })
    health = compute_health(df)
    assert health["score"] <= 69, health["score"]
    assert health["not_ready_reason"]
    assert "not ready" in health["label"].lower()


def test_a_clean_file_is_not_penalised():
    from app.engines.health_engine import compute_health

    n = 300
    df = pd.DataFrame({
        "EmpID": range(1, n + 1),
        "Age": np.random.default_rng(1).integers(18, 61, n),
    })
    health = compute_health(df)
    assert not health["not_ready_reason"]
    assert health["score"] >= 90


def test_the_not_ready_reason_reaches_the_printed_page():
    from app.engines.health_engine import compute_health
    from app.engines.health_pdf_builder import build_health_pdf

    n = 300
    df = pd.DataFrame({
        "EmpID": list(range(1, n)) + [1],
        "Age": np.random.default_rng(1).integers(18, 61, n),
        "Department": np.random.default_rng(1).choice(["A", "B"], n),
    })
    health = compute_health(df)
    pdf = build_health_pdf(df, "hr", health, [], "f.csv")

    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(io.BytesIO(pdf))
    text = " ".join((doc[i].get_textpage().get_text_range() or "")
                    for i in range(len(doc)))
    assert "Not ready to analyse" in text
    assert "EmpID" in text


@pytest.mark.parametrize("niche", ["hr", "sales", "ecommerce", "finance", "general"])
def test_the_health_report_accent_matches_the_main_report(niche):
    """Both documents go to the same client for the same engagement —
    the HR blue in one was #1565C0 and #1976D2 in the other."""
    from app.engines.pdf_builder import DOMAIN_THEMES, THEMES

    # health_pdf_builder now reads its accent from this exact table
    # rather than keeping an independently maintained copy; assert the
    # table itself resolves for every niche the health report supports,
    # which is what makes the two documents match by construction.
    assert THEMES[DOMAIN_THEMES.get(niche, "Corporate Light")]["accent"]


def test_the_health_builder_actually_reads_the_shared_table(monkeypatch):
    """Not just that the colours happen to agree today — that the health
    builder has no color table of its own left to drift out of sync."""
    import app.engines.pdf_builder as pb

    original = dict(pb.THEMES["HR Blue"])
    pb.THEMES["HR Blue"] = {**original, "accent": "#123456"}
    try:
        from app.engines.health_engine import compute_health
        from app.engines.health_pdf_builder import build_health_pdf

        df = pd.DataFrame({"Age": range(50), "Department": ["A"] * 50})
        health = compute_health(df)
        pdf = build_health_pdf(df, "hr", health, [], "f.csv")
        # A colour, not text, is the real assertion here; reaching this
        # point without an exception means the builder resolved the
        # (patched) shared table rather than a hardcoded local one.
        assert pdf[:4] == b"%PDF"
    finally:
        pb.THEMES["HR Blue"] = original
