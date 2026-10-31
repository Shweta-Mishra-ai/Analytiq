"""
tests/test_report_wording.py — what a report says has to be true of the
data it was built from, and nothing more.

Each check here is a defect found in a generated report: a department gap
with p=0.78 called "the highest attrition", "Mercer 2024" and "McKinsey
2024" figures nobody could trace, "low ratings are associated with lower
revenue (see the comparison above)" in a catalogue with no revenue column,
a product's list price called "average order", a −46% "cost growth" read
off the first and last week, and pages holding three lines of text.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════
#  helpers
# ══════════════════════════════════════════════════════════

def _hr_with_gaps(n=900, seed=5):
    """HR data with a real department gap and a real pay-band gap."""
    rng = np.random.default_rng(seed)
    dept = rng.choice(["Sales", "Engineering", "HR", "Marketing"], n)
    salary = rng.normal(60_000, 12_000, n).round(0)
    p_leave = np.where(dept == "Sales", 0.40, 0.12) + np.where(salary < 50_000, 0.10, 0)
    return pd.DataFrame({
        "employee_id": range(1, n + 1),
        "department": dept,
        "salary": salary,
        "tenure_years": rng.integers(0, 20, n),
        "satisfaction": rng.uniform(1, 5, n).round(1),
        "attrition": np.where(rng.random(n) < p_leave, "Yes", "No"),
        "hire_date": pd.date_range("2012-01-01", periods=n, freq="4D"),
    })


def _catalogue(n=400, seed=4):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "product_id": [f"P{i}" for i in range(n)],
        "category": rng.choice(["Toys", "Books", "Home"], n),
        "price": rng.gamma(2, 20, n).round(2),
        "rating": rng.uniform(1, 5, n).round(1),
        "reviews": rng.integers(0, 900, n),
    })


def _ledger(n=400, seed=3):
    rng = np.random.default_rng(seed)
    rev = rng.normal(100_000, 25_000, n).round(2)
    return pd.DataFrame({
        "month": pd.date_range("2022-01-01", periods=n, freq="W"),
        "department": rng.choice(["Ops", "Sales", "R&D", "Admin"], n),
        "revenue": rev,
        "cost": (rev * rng.uniform(0.5, 0.95, n)).round(2),
    })


def _payload(df):
    from app.engines.health_engine import build_report_payload
    from app.engines.story_engine import detect_domain
    domain, _ = detect_domain(df)
    return domain, build_report_payload(df, domain)


def _all_text(payload) -> str:
    parts = [payload["executive_summary"]]
    for k in ("key_findings", "risks", "opportunities", "actions"):
        parts += list(payload[k])
    for c in payload["insights"]:
        parts += [c["title"], c["body"], c["action"]]
    return "\n".join(parts)


def _health_pdf(df, name="data.csv"):
    from app.engines.health_engine import compute_health
    from app.engines.health_pdf_builder import build_health_pdf
    domain, p = _payload(df)
    return build_health_pdf(
        df, domain, compute_health(df), p["insights"], name,
        executive_summary=p["executive_summary"], key_findings=p["key_findings"],
        risks=p["risks"], opportunities=p["opportunities"], actions=p["actions"])


def _pages(pdf: bytes):
    import pypdf
    return [(pg.extract_text() or "") for pg in pypdf.PdfReader(io.BytesIO(pdf)).pages]


# ══════════════════════════════════════════════════════════
#  No invented sources, causes or boilerplate
# ══════════════════════════════════════════════════════════

_BANNED = (
    "Mercer 2024", "McKinsey", "MEDDIC", "BANT", "VIP tier", "CRITICAL:",
    "WARNING:", "IMMEDIATE", "Proceed directly", "see the comparison above",
    "Pareto opportunity", "Quarterly satisfaction pulse", "no recent promotion",
    "imputation applied", "subgroups may tell", "Industry Norm",
    "No single finding stands out", "The level, not the direction",
    "fastest to fix", "competitive advantage", "Burnout Risk",
)


@pytest.mark.parametrize("make", [_hr_with_gaps, _catalogue, _ledger])
def test_no_invented_sources_or_boilerplate(make, hr_df):
    for df in (make(), hr_df):
        text = _all_text(_payload(df)[1])
        for phrase in _BANNED:
            assert phrase.lower() not in text.lower(), (phrase, make.__name__)


def test_actions_name_only_columns_the_data_has(hr_df):
    actions = " ".join(_payload(hr_df)[1]["actions"]).lower()
    for absent in ("promotion", "manager", "team"):
        assert absent not in actions, actions


# ══════════════════════════════════════════════════════════
#  A gap between groups is reported only when it is real
# ══════════════════════════════════════════════════════════

def test_a_chance_department_gap_is_not_reported(hr_df):
    # Attrition in the shared fixture is random: any department spread is noise.
    from app.engines.health_engine import build_full_insights
    cards = build_full_insights(hr_df, "hr")
    assert not [c for c in cards if c.get("topic") in ("department", "pay")
                or c["tag"] in ("DEPARTMENT RISK", "PAY-DRIVEN ATTRITION")], \
        [c["title"] for c in cards]


def test_a_real_department_gap_is_reported_with_its_test():
    from app.engines.health_engine import build_full_insights
    cards = build_full_insights(_hr_with_gaps(), "hr")
    dept = [c for c in cards if c.get("topic") == "department"]
    assert dept, [c["title"] for c in cards]
    assert dept[0]["title"].startswith("'Sales' loses")
    assert "Chi-square p=" in dept[0]["body"]


def test_the_attrition_card_measures_from_the_top_of_the_band():
    from app.engines.health_engine import build_full_insights
    df = _hr_with_gaps()
    rate = (df["attrition"] == "Yes").mean() * 100
    card = next(c for c in build_full_insights(df, "hr") if c.get("topic") == "attrition")
    assert "{:.1f} points above the top of the 10–15%".format(rate - 15) in card["body"]


def test_niche_cards_do_not_repeat_a_domain_topic():
    # Domain-engine cards carry a topic; the health engine's own cards do not.
    from app.engines.health_engine import build_full_insights
    cards = build_full_insights(_hr_with_gaps(), "hr")
    for topic, tag in (("attrition", "ATTRITION"), ("satisfaction", "EMPLOYEE SATISFACTION")):
        n = sum(1 for c in cards if c.get("topic") == topic
                or ("topic" not in c and c["tag"] == tag))
        assert n <= 1, (topic, [c["title"] for c in cards])


# ══════════════════════════════════════════════════════════
#  Claims about revenue need revenue
# ══════════════════════════════════════════════════════════

def test_ratings_make_no_revenue_claim_without_a_revenue_column():
    text = _all_text(_payload(_catalogue())[1]).lower()
    assert "revenue" not in text, text


def test_a_list_price_is_not_called_an_order_value():
    text = _all_text(_payload(_catalogue())[1]).lower()
    assert "average order" not in text


# ══════════════════════════════════════════════════════════
#  Trends are fitted and tested, never read off two points
# ══════════════════════════════════════════════════════════

def test_noisy_revenue_and_cost_report_no_trend():
    text = _all_text(_payload(_ledger())[1])
    assert "first→last" not in text
    assert "declining" not in text.lower()
    assert "Revenue per period shows no trend" in text


def test_no_trend_is_read_over_a_hire_date(hr_df):
    from app.engines.domains.general_depth import _time_column
    assert _time_column(hr_df) is None
    assert "shows no trend" not in " ".join(_payload(hr_df)[1]["key_findings"])


# ══════════════════════════════════════════════════════════
#  Identifier test is shared, and safe on any column
# ══════════════════════════════════════════════════════════

def test_identifier_check_is_the_shared_one(hr_df):
    from app.engines.pdf_primitives import is_id_col
    assert is_id_col("employee_id", hr_df["employee_id"])
    assert not is_id_col("hire_date", hr_df["hire_date"])
    assert not is_id_col("department", hr_df["department"])   # used to raise


def test_identifier_check_respects_a_confirmed_role(hr_df):
    from app.engines.pdf_primitives import is_id_col
    df = hr_df.copy()
    df.attrs["column_roles"] = {"employee_id": {"role": "measure", "direction": 0}}
    assert not is_id_col("employee_id", df["employee_id"])


# ══════════════════════════════════════════════════════════
#  Layout
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("make", [_hr_with_gaps, _ledger, _catalogue])
def test_health_report_has_no_near_empty_page(make):
    pages = _pages(_health_pdf(make()))
    # Cover and contents are short by design; every other page carries content.
    for i, text in enumerate(pages[2:], start=3):
        assert len(text.strip()) > 300, f"page {i} is nearly empty: {text[:120]!r}"


def test_health_report_summary_table_has_a_readable_header(hr_df):
    pages = _pages(_health_pdf(hr_df))
    overview = next(t for t in pages if "Dataset Summary" in t)
    assert "Measure" in overview and "Value" in overview


def test_health_report_contents_match_the_sections(hr_df):
    pages = _pages(_health_pdf(hr_df))
    contents = pages[1]
    body = "\n".join(pages[2:])
    for line in contents.splitlines():
        if line[:2].isdigit() and " " in line:
            heading = line[2:].split(" . ")[0].strip(" .0123456789")
            assert heading.split(" —")[0] in body, (heading, contents)


# ══════════════════════════════════════════════════════════
#  Main report
# ══════════════════════════════════════════════════════════

def _main_pdf(df, ml_report=None):
    from app.engines.chart_exporter import build_report_charts
    from app.engines.data_profiler import profile_dataset
    from app.engines.insights_builder import build_top_insights
    from app.engines.pdf_builder import build_pdf
    from app.engines.story_engine import detect_domain, generate_story
    domain, _ = detect_domain(df)
    story = generate_story(df)
    charts = [(c.title, c.png, c.caption) for c in build_report_charts(df, domain=domain)]
    return build_pdf(
        df=df,
        config={"title": "Review", "subtitle": "", "client_name": "Acme",
                "confidential": True, "theme_name": "", "logo_path": None},
        profile=profile_dataset(df), cleaning_summary=None, stats_report=None,
        bi_report=None, ml_report=ml_report, chart_data=charts,
        executive_summary=story.executive_summary, findings=story.key_findings,
        risks=story.business_risks, opportunities=story.opportunities,
        recommendations=story.recommended_actions,
        top_insights=build_top_insights(df=df, domain=domain, story_obj=story),
        attrition=getattr(story, "attrition", None), domain=domain)


def test_departments_are_not_ranked_when_the_gap_is_chance(hr_df):
    text = "\n".join(_pages(_main_pdf(hr_df)))
    assert "HIGHEST DEPARTMENT" not in text
    assert "within what chance alone produces" in text
    assert "\nHIGH\n" not in text and " HIGH" not in text.replace("HIGH PRIORITY", "")


def test_a_real_department_gap_is_ranked_with_its_test():
    text = "\n".join(_pages(_main_pdf(_hr_with_gaps())))
    assert "HIGHEST DEPARTMENT" in text
    assert "by more than chance (chi-square" in text


def test_main_report_ends_on_content_not_a_stranded_box(hr_df):
    pages = _pages(_main_pdf(hr_df))
    assert "Basis of preparation" in pages[1], "basis box belongs under the contents"
    for i, text in enumerate(pages[2:], start=3):
        assert len(text.strip()) > 300, f"page {i} is nearly empty: {text[:120]!r}"


def test_card_text_is_escaped_and_steps_are_split():
    from app.engines.pdf_builder import _steps
    assert _steps("R&D <b>  2. next") == "R&amp;D &lt;b&gt;<br/>2. next"


def test_a_negative_r2_is_called_worse_than_the_average():
    from types import SimpleNamespace
    from app.engines.ml_engine import _no_signal_note
    rep = SimpleNamespace(task="regression", target_col="price",
                          best_model=SimpleNamespace(test_score=-0.04))
    assert "worse than always guessing" in _no_signal_note(rep)
    rep.best_model.test_score = 0.02
    assert "barely better than" in _no_signal_note(rep)


def test_bi_brief_carries_no_log_lines(hr_df):
    from app.engines.bi_engine import run_bi
    brief = run_bi(hr_df).executive_brief
    assert "analysis run on" not in brief
    assert "difference(s) identified" not in brief
