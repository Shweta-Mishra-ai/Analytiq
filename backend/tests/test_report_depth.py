"""
What a client actually receives.

Every check here came from building the report for thirteen domains and
reading the pages, not from reasoning about the code. The defects that
prompted them, in the order they appeared on the page:

  * a finance report with no finance page in it, because the section
    crashed summing a text column and the failure was swallowed;
  * thirty-three pages, twenty-two of them a daily revenue extract with
    day-over-day changes of +1,886%;
  * an education report headed WORKFORCE ANALYTICS and a logistics one
    headed REVENUE ANALYTICS, because the cover band came from the
    colour theme and themes are shared;
  * a CRITICAL finding that the sales team hit 6% of quota, arrived at
    by dividing average deal size by an annual team target;
  * twelve of the thirteen domains with no domain page at all.
"""
import io
import warnings

import numpy as np
import pandas as pd
import pypdf
import pytest

warnings.filterwarnings("ignore")


# ── fixtures ──────────────────────────────────────────────

@pytest.fixture()
def logistics():
    r = np.random.default_rng(11)
    n = 1500
    lane = r.choice(["NE-Corridor", "Midwest", "West"], n, p=[.4, .35, .25])
    transit = np.where(lane == "West", r.gamma(7, 1.1, n), r.gamma(4, .9, n))
    return pd.DataFrame({
        "shipment_id": range(1, n + 1),
        "lane": lane,
        "carrier": r.choice(["Carrier A", "Carrier B"], n),
        "transit_days": transit.round(1),
        "freight_cost": (transit * r.normal(210, 45, n)).round(2),
        "on_time": (r.random(n) > np.where(lane == "West", .34, .09))
                   .astype(int),
    })


@pytest.fixture()
def management_accounts():
    """The shape a finance team exports: account, budget, actual, daily."""
    r = np.random.default_rng(5)
    n = 1500
    account = r.choice(["Payroll", "Marketing", "Cloud"], n)
    base = {"Payroll": 52000, "Marketing": 18000, "Cloud": 26000}
    planned = np.array([base[a] for a in account])
    actual = planned * r.normal(1.0, .22, n)
    return pd.DataFrame({
        "txn_id": range(1, n + 1),
        "account": account,
        "cost_center": r.choice(["CC-100", "CC-200"], n),
        "period": pd.to_datetime("2024-01-01")
                  + pd.to_timedelta(r.integers(0, 700, n), "D"),
        "budget": planned,
        "actual": actual.round(0),
        "revenue": (actual * r.normal(2.1, .4, n)).round(0),
    })


def _render(df, domain, title="Review"):
    from app.engines.data_profiler import profile_dataset
    from app.engines.pdf.builder import build_pdf
    out = build_pdf(df, {"title": title, "client_name": "Northwind"},
                    profile=profile_dataset(df), domain=domain)
    data = out if isinstance(out, (bytes, bytearray)) else out.getvalue()
    return pypdf.PdfReader(io.BytesIO(data))


def _pages(reader):
    return [" ".join((p.extract_text() or "").split()) for p in reader.pages]


# ── the finance section that was silently dropped ─────────

def test_a_text_column_is_never_chosen_as_a_money_column(management_accounts):
    """`cost_center` holds "CC-100". It matched the word `cost`, was
    summed, raised ValueError, and the whole Finance Analysis section
    was dropped — leaving a finance report with no finance page."""
    from app.engines.pdf.domain_sections import _finance_page
    from app.engines.pdf.theme import THEMES, _styles

    T = THEMES["Corporate Light"]
    story = []
    _finance_page(story, _styles(T), T, management_accounts, {}, 460.0)
    assert story, "the finance page produced nothing"


def test_the_finance_report_contains_a_finance_page(management_accounts):
    text = " ".join(_pages(_render(management_accounts, "finance")))
    assert "P&L Summary" in text or "Finance Analysis" in text


# ── the twenty-two page data dump ─────────────────────────

def test_a_daily_column_is_rolled_up_before_it_is_tabulated(
        management_accounts):
    """Two years of daily rows are not reporting periods. Summed per day
    they produced hundreds of lines whose period-over-period change is
    noise — nobody reads "Tuesday was 1,886% up on Monday"."""
    from app.engines.pdf.domain_sections import _revenue_by_period
    series, grain = _revenue_by_period(management_accounts, "period",
                                       "revenue")
    assert grain, "a daily column was left at daily grain"
    assert len(series) <= 24, len(series)


def test_a_long_label_column_is_capped_rather_than_dumped():
    from app.engines.pdf.domain_sections import (PERIOD_ROLLUP_AT,
                                                 _revenue_by_period)
    r = np.random.default_rng(7)
    n = 2000
    df = pd.DataFrame({
        "period": ["label-{:04d}".format(i) for i in r.integers(0, 400, n)],
        "revenue": r.normal(1000, 200, n),
    })
    series, grain = _revenue_by_period(df, "period", "revenue")
    # Nothing to roll up to, so the caller's row cap is the only defence
    # and the grain note must not claim a rollup happened.
    assert grain == ""
    assert len(series) > PERIOD_ROLLUP_AT


def test_the_report_does_not_run_to_dozens_of_pages(management_accounts):
    """It was 33 pages against 11 for every other domain."""
    assert len(_render(management_accounts, "finance").pages) <= 20


# ── the cover band ────────────────────────────────────────

@pytest.mark.parametrize("domain,band", [
    ("education", "EDUCATION ANALYTICS"),
    ("logistics", "SUPPLY CHAIN ANALYTICS"),
    ("healthcare", "CLINICAL ANALYTICS"),
    ("finance", "FINANCIAL ANALYTICS"),
    ("insurance", "UNDERWRITING ANALYTICS"),
])
def test_the_cover_names_the_readers_discipline(domain, band, logistics):
    """It came from the colour theme, and themes are shared, so an
    education report was headed WORKFORCE ANALYTICS."""
    cover = _pages(_render(logistics, domain))[0]
    assert band in cover, cover[:200]


def test_every_domain_declares_a_cover_band():
    from app.engines.domains.registry import REGISTRY
    missing = [k for k, spec in REGISTRY.items() if not spec.cover_label]
    assert missing == [], missing


# ── the false quota finding ───────────────────────────────

def test_a_constant_quota_yields_no_attainment_figure():
    """Average deal size over an annual team quota gave "6% achievement,
    94pp below target", marked CRITICAL — on a file whose own data
    quality page said that quota column was constant and should be
    dropped."""
    from app.engines.domains.sales import _quota_attainment
    r = np.random.default_rng(3)
    n = 800
    df = pd.DataFrame({"deal_size": r.lognormal(9.4, .7, n).round(0),
                       "quota": 250000})
    value, reason = _quota_attainment(df, "deal_size", "quota")
    assert value is None
    assert "every row" in reason.lower()


def test_attainment_is_computed_when_the_grain_lines_up():
    from app.engines.domains.sales import _quota_attainment
    r = np.random.default_rng(3)
    n = 600
    target = r.choice([40000, 60000, 80000], n)
    df = pd.DataFrame({"revenue": target * r.normal(.92, .18, n),
                       "target": target})
    value, basis = _quota_attainment(df, "revenue", "target")
    assert value is not None
    assert 80 <= value <= 105, value
    assert "row by row" in basis


def test_mismatched_grain_is_refused_even_when_the_target_varies():
    """Per-deal revenue against a per-quarter target is still a category
    error when the target happens to have three distinct values."""
    from app.engines.domains.sales import _quota_attainment
    r = np.random.default_rng(9)
    n = 600
    df = pd.DataFrame({"deal_size": r.normal(15000, 4000, n),
                       "quota": r.choice([2_000_000, 2_500_000, 3_000_000],
                                         n)})
    value, _ = _quota_attainment(df, "deal_size", "quota")
    assert value is None


def test_the_sales_report_makes_no_false_target_claim():
    from app.engines.story_engine import generate_story
    r = np.random.default_rng(3)
    n = 1200
    rep = r.choice(["Rep 01", "Rep 02", "Rep 03"], n)
    df = pd.DataFrame({
        "opportunity_id": range(1, n + 1), "sales_rep": rep,
        "region": r.choice(["North", "South"], n),
        "deal_size": r.lognormal(9.4, .7, n).round(0),
        "quota": 250000,
        "won": (r.random(n) < .25).astype(int),
    })
    titles = " ".join(i.title for i in generate_story(df).top_insights)
    assert "Below Target" not in titles, titles


# ── the page twelve domains never had ─────────────────────

@pytest.mark.parametrize("domain", [
    "hr", "sales", "saas", "logistics", "healthcare", "insurance",
    "ecommerce", "marketing", "operations", "education", "realestate",
    "energy", "general",
])
def test_every_domain_renders_a_performance_page(domain, logistics):
    from app.engines.pdf.domain_sections import _domain_performance_page
    from app.engines.pdf.theme import THEMES, _styles
    T = THEMES["Corporate Light"]
    story = []
    _domain_performance_page(story, _styles(T), T, logistics, {}, 460.0,
                             domain=domain)
    assert story, domain


def test_the_performance_page_breaks_the_outcome_down_by_group(logistics):
    from app.engines.pdf.domain_sections import _domain_performance_page
    from app.engines.pdf.theme import THEMES, _styles
    T = THEMES["Corporate Light"]
    story = []
    _domain_performance_page(story, _styles(T), T, logistics, {}, 460.0,
                             domain="logistics")
    text = " ".join(getattr(f, "text", "") for f in story)
    assert "Lane" in text or "Carrier" in text


def test_the_heading_never_stands_over_an_empty_section():
    """A page with a title and nothing under it is worse than no page."""
    from app.engines.pdf.domain_sections import _domain_performance_page
    from app.engines.pdf.theme import THEMES, _styles
    T = THEMES["Corporate Light"]
    bare = pd.DataFrame({"note": ["a", "b", "c"] * 40})
    story = []
    _domain_performance_page(story, _styles(T), T, bare, {}, 460.0,
                             domain="general")
    text = " ".join(getattr(f, "text", "") for f in story)
    assert "cover" in text.lower() or "no measure" in text.lower(), text[:300]


def test_a_binary_flag_is_not_offered_as_a_total(logistics):
    """"Total On Time 1,355 — 54.1% share" is a count of shipments
    wearing the word Total."""
    from app.engines.pdf.domain_sections import _primary_measure
    from app.engines.domains.registry import spec_for
    measure = _primary_measure(logistics, spec_for("logistics"))
    assert measure != "on_time"


# ── the scorecard verdicts ────────────────────────────────

def test_a_kpi_with_a_published_range_is_read_against_it():
    """Every row of every scorecard said "No published range" because
    the benchmark KEY was passed to a lookup that expects a column
    name."""
    from app.engines.kpi_engine import compute_kpis
    from app.engines.pdf.domain_sections import _verdict_against
    r = np.random.default_rng(1)
    n = 1200
    df = pd.DataFrame({
        "EmployeeID": range(n),
        "Department": r.choice(["Sales", "Support"], n),
        "MonthlyIncome": r.normal(6500, 1500, n),
        "Attrition": np.where(r.random(n) < .19, "Yes", "No"),
    })
    verdicts = [_verdict_against(c, "hr") for c in compute_kpis(df, "hr")]
    assert any("typical" in v for v in verdicts), verdicts
