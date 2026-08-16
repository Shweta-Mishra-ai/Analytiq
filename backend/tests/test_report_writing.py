"""
How the report reads, as distinct from whether its arithmetic is right.

Four things a reader notices in the first paragraph.

**Scientific notation.** `%g` shortens a float by switching to exponent
form, which is right for a p-value and wrong for money. A financial
review opened with "Median 'revenue' ranges from 7.72e+04 in 'Support' to
8.89e+05 in 'Retail'". Nobody writes revenue that way.

**A generic cross-tab as the headline.** The segment-difference check runs
on any dataset and was marked "high", so it outranked the finance
engine's own work: the opening sentence of a P&L review was the revenue
gap between a trading cost centre and a support one, ahead of margin,
cost structure and budget variance.

**Urgency asserted by position.** The first two recommendations in the
list were stamped CRITICAL whatever they said, so a report with nothing
critical in it closed with "[CRITICAL] Review attainment on the same
cadence the targets are set".

**Arithmetic reported as discovery.** "Notable relationship: 'revenue' vs
'cogs' (r=+0.98)". Gross profit is revenue minus cost of sales.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from app.engines.story_engine import _is_tautological_pair, generate_story
from app.services.numfmt import human_number


@pytest.fixture()
def ledger():
    rng = np.random.default_rng(5)
    months = pd.date_range("2023-01-31", periods=30, freq="ME")
    base = {"Retail": 900_000, "Wholesale": 600_000,
            "Services": 350_000, "Support": 80_000}
    cost_ratio = {"Retail": .62, "Wholesale": .78,
                  "Services": .45, "Support": .90}
    rows = []
    for m in months:
        for cc, amount in base.items():
            rev = rng.normal(amount, 60_000)
            cogs = rev * rng.normal(cost_ratio[cc], .03)
            rows.append({"period": m, "cost_centre": cc,
                         "revenue": round(rev, 2), "cogs": round(cogs, 2),
                         "gross_profit": round(rev - cogs, 2),
                         "opex": round(rev * rng.normal(.18, .02), 2),
                         "budget": round(rev * rng.normal(1.05, .06), 2)})
    df = pd.DataFrame(rows)
    df["ebitda"] = (df.gross_profit - df.opex).round(2)
    return df


def _all_text(story):
    return " ".join([story.executive_summary, story.headline]
                    + story.key_findings + story.business_risks
                    + story.opportunities + story.recommended_actions)


# ══════════════════════════════════════════════════════════
#  Numbers are written, not printed
# ══════════════════════════════════════════════════════════

def test_no_figure_reaches_the_report_in_scientific_notation(ledger):
    text = _all_text(generate_story(ledger))
    offenders = re.findall(r"\d\.?\d*e[+-]\d+", text)
    assert not offenders, offenders


def test_the_executive_summary_is_free_of_exponents(ledger):
    assert "e+0" not in generate_story(ledger).executive_summary


@pytest.mark.parametrize("value,expected", [
    (77_200, "77k"),
    (889_000, "889k"),
    (57_974_807, "58.0m"),
    (0.0, "0"),
])
def test_figures_are_written_as_a_person_writes_them(value, expected):
    assert human_number(value) == expected


def test_a_non_number_does_not_raise():
    assert human_number(float("nan")) == "n/a"
    assert human_number("not a number") == "not a number"


# ══════════════════════════════════════════════════════════
#  The domain's own finding leads
# ══════════════════════════════════════════════════════════

def test_a_financial_review_opens_on_the_financial_position(ledger):
    """It opened on the revenue gap between a trading cost centre and a
    support one — true, trivial, and not what a finance reader turns to
    the first page for."""
    summary = generate_story(ledger).executive_summary
    opening = summary.split(".")[0].lower()
    assert "margin" in opening or "profit" in opening or "revenue" in opening
    assert "segments" not in opening, summary


def test_a_generic_cross_tab_does_not_take_the_headline(ledger):
    """Severity still orders the insight list — a warning outranks a
    healthy-margin note, and should. What it must not do is decide what
    the report is about."""
    story = generate_story(ledger)
    assert "segmentation" not in story.headline.lower()
    assert any(i.category.startswith("finance") for i in story.top_insights), \
        [i.category for i in story.top_insights]


def test_the_domain_engine_ranks_first_within_its_severity_band():
    """Two warnings, one from the domain engine and one generic."""
    from app.engines.story_engine import _is_tautological_pair  # noqa: F401

    rng = np.random.default_rng(24)
    n = 600
    df = pd.DataFrame({
        "cost_centre": rng.choice(["A", "B", "C"], n),
        "revenue": rng.normal(50_000, 8_000, n),
        "cogs": rng.normal(46_000, 8_000, n),
    })
    story = generate_story(df)
    same_band = [i for i in story.top_insights if i.severity == "warning"]
    if len(same_band) >= 2 and any(i.category.startswith("finance")
                                   for i in same_band):
        assert same_band[0].category.startswith("finance"), \
            [i.category for i in same_band]


def test_the_segment_gap_is_stated_without_calling_it_underperformance(ledger):
    """A support cost centre earning less revenue than a trading one is
    what a support cost centre is for."""
    risks = " ".join(generate_story(ledger).business_risks).lower()
    assert "underperforms on 'revenue'" not in risks, risks


# ══════════════════════════════════════════════════════════
#  Urgency has to be earned
# ══════════════════════════════════════════════════════════

def test_nothing_is_marked_critical_without_a_critical_finding():
    rng = np.random.default_rng(21)
    n = 400
    df = pd.DataFrame({
        "region": rng.choice(["North", "South"], n),
        "units": rng.integers(1, 50, n),
        "revenue": rng.normal(5_000, 400, n),
    })
    story = generate_story(df)
    if not story.critical_issues:
        flagged = [a for a in story.recommended_actions if "[CRITICAL]" in a]
        assert not flagged, flagged


def test_a_critical_finding_still_produces_critical_actions():
    """The fix must not remove urgency where it was measured."""
    rng = np.random.default_rng(22)
    n = 400
    df = pd.DataFrame({
        "category": rng.choice(["A", "B", "C"], n),
        "revenue": rng.normal(10_000, 2_000, n),
        "profit": rng.normal(-500, 2_000, n),      # widely loss-making
    })
    story = generate_story(df)
    if story.critical_issues:
        assert any("[CRITICAL]" in a for a in story.recommended_actions)


# ══════════════════════════════════════════════════════════
#  Arithmetic is not a discovery
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("a,b", [
    ("revenue", "cogs"),
    ("revenue", "gross_profit"),
    ("budget", "variance"),
    ("actual", "plan"),
    ("total_sales", "turnover"),
])
def test_a_definitional_pair_is_not_a_notable_relationship(a, b):
    assert _is_tautological_pair(a, b), (a, b)


@pytest.mark.parametrize("a,b", [
    ("marketing_spend", "signups"),
    ("cycle_days", "win_rate"),
    ("headcount", "defect_rate"),
])
def test_a_real_relationship_is_still_reportable(a, b):
    assert not _is_tautological_pair(a, b), (a, b)


def test_the_summary_does_not_report_a_p_and_l_identity(ledger):
    summary = generate_story(ledger).executive_summary
    if "Notable relationship" in summary:
        tail = summary.split("Notable relationship")[1]
        assert not ("revenue" in tail and "cogs" in tail), summary


# ══════════════════════════════════════════════════════════
#  The same finding is not counted twice
# ══════════════════════════════════════════════════════════

def test_a_headline_warning_is_not_also_an_additional_warning():
    """"…This is the most material finding. 1 additional warning requires
    review." — with one warning in the file, in consecutive sentences."""
    rng = np.random.default_rng(23)
    n = 500
    df = pd.DataFrame({
        "region": rng.choice(["North", "South", "East"], n),
        "revenue": rng.lognormal(9, 1.1, n),
    })
    story = generate_story(df)
    n_warn = sum(1 for i in story.top_insights if i.severity == "warning")
    stated = re.search(r"(\d+) additional warning", story.executive_summary)
    if stated and n_warn:
        assert int(stated.group(1)) < n_warn or n_warn == 0, \
            (story.executive_summary, n_warn)


# ══════════════════════════════════════════════════════════
#  Trivial movement is not a trend
# ══════════════════════════════════════════════════════════

def test_a_third_of_a_percent_is_not_growth(ledger):
    """"Revenue growing 0.3%" invites the reader to act on noise."""
    findings = " ".join(generate_story(ledger).key_findings)
    assert not re.search(r"[Rr]evenue (rose|fell|growing|declining) 0\.\d%",
                         findings), findings
    assert "broadly flat" in findings, findings
