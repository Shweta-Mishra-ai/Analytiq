"""
The two numbers a sales report exists to give, and the ways it got them
wrong.

**Quota attainment was not attainment.** The engine divided the mean of
the revenue column by the mean of the quota column. On any normal export
that is a per-deal amount over a per-period-per-rep target — 18,420
against 250,000 — reported as "Target Gap: 7% Achievement — 93pp Below
Target", marked CRITICAL, and placed at the top of the report as its
headline finding. Attainment only means something per quota-holder:
bookings summed for that person, against their own quota.

**Win rate was never computed.** `find_outcome_col` required at least
half the rows to be decided, which no live pipeline export satisfies —
most opportunities are open by definition. So on a file carrying a
`deal_stage` column full of "Closed Won" and "Closed Lost", the single
most-asked-for sales number never appeared.

The report also closed with five fixed recommendations regardless of the
file, including a quarterly pricing review by product category on data
with no product and no margin column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.domains.sales import _insights_sales
from app.engines.domains.base import col_stats


@pytest.fixture()
def pipeline():
    """An opportunity export: mostly open, quota repeated per row."""
    rng = np.random.default_rng(3)
    n = 900
    reps = ["A. Rao", "B. Chen", "C. Diaz", "D. Ekwe", "E. Fry"]
    return pd.DataFrame({
        "opportunity_id": range(1, n + 1),
        "created_date": pd.to_datetime("2024-01-01")
                        + pd.to_timedelta(rng.integers(0, 540, n), "D"),
        "sales_rep": rng.choice(reps, n, p=[.3, .25, .2, .15, .10]),
        "territory": rng.choice(["EMEA", "AMER", "APAC"], n),
        "deal_stage": rng.choice(
            ["Prospect", "Qualified", "Proposal", "Closed Won", "Closed Lost"],
            n, p=[.2, .2, .2, .22, .18]),
        "deal_amount": rng.lognormal(9.5, 0.8, n).round(2),
        "quota": 250_000.0,
        "forecast_category": rng.choice(["Commit", "Best Case", "Pipeline"], n),
    })


def _run(df):
    stats = {c: col_stats(df[c])
             for c in df.select_dtypes(include="number").columns}
    return _insights_sales(df, {c: s for c, s in stats.items() if s}, [])


def _titles(out):
    return [i.title for i in out["insights"]]


def _text(out):
    parts = []
    for i in out["insights"]:
        parts += [i.title, i.problem, i.cause, i.evidence, i.action, i.impact]
    return " ".join(parts + out["findings"] + out["risks"]
                    + out["opportunities"] + out["actions"])


# ══════════════════════════════════════════════════════════
#  Quota attainment
# ══════════════════════════════════════════════════════════

def test_attainment_is_not_a_per_deal_mean_over_a_period_target(pipeline):
    """The defect exactly: 18,420 / 250,000 reported as 7% attainment."""
    out = _run(pipeline)
    assert not [t for t in _titles(out) if "7%" in t], _titles(out)
    assert "93pp Below Target" not in _text(out)


def test_attainment_is_measured_per_quota_holder(pipeline):
    out = _run(pipeline)
    hits = [t for t in _titles(out) if "Quota-Holders" in t]
    assert hits, _titles(out)
    assert "5" in hits[0], hits


def test_the_attainment_figure_is_arithmetically_right():
    """A at 100% of quota, B at 50%, C at 75% — median 75%, one at
    target."""
    df = pd.DataFrame({
        "sales_rep": ["A"] * 40 + ["B"] * 40 + ["C"] * 40,
        "deal_stage": ["Closed Won"] * 120,
        "deal_amount": [2_500.0] * 40 + [1_250.0] * 40 + [1_875.0] * 40,
        "quota": [100_000.0] * 120,
    })
    out = _run(df)
    text = _text(out)
    assert "1 of 3" in text, text
    assert "75%" in text, text


def test_a_target_with_nothing_to_group_by_is_refused():
    """A per-row amount and a per-period target are not comparable, and
    saying so is better than dividing them."""
    df = pd.DataFrame({
        "deal_amount": np.linspace(1_000, 5_000, 200),
        "quota": [250_000.0] * 200,
        "territory": ["EMEA"] * 100 + ["AMER"] * 100,
    })
    out = _run(df)
    text = " ".join(out["findings"])
    assert "cannot be calculated" in text or "not comparable" in text, text
    assert not [t for t in _titles(out) if "Achievement" in t]


def test_attainment_states_the_period_it_covers(pipeline):
    """A quota is for a period; bookings summed over a different one make
    the ratio meaningless without the caveat."""
    out = _run(pipeline)
    hit = next(i for i in out["insights"] if "Quota-Holders" in i.title)
    assert "months" in hit.evidence, hit.evidence


# ══════════════════════════════════════════════════════════
#  Win rate
# ══════════════════════════════════════════════════════════

def test_win_rate_is_computed_on_a_mostly_open_pipeline(pipeline):
    """Requiring half the rows to be decided rejected every real
    pipeline export."""
    out = _run(pipeline)
    assert [t for t in _titles(out) if "Win Rate" in t], _titles(out)


def test_open_deals_are_not_counted_as_losses(pipeline):
    """Folding the pipeline into the denominator halves the rate."""
    decided = pipeline["deal_stage"].isin(["Closed Won", "Closed Lost"]).sum()
    won = (pipeline["deal_stage"] == "Closed Won").sum()
    expected = round(won / decided * 100)
    out = _run(pipeline)
    hit = next(i for i in out["insights"] if "Win Rate" in i.title)
    assert "{:.0f}%".format(expected) in hit.title, hit.title
    assert "still open" in hit.problem or "open" in hit.problem


def test_the_win_rate_names_its_source_column(pipeline):
    out = _run(pipeline)
    hit = next(i for i in out["insights"] if "Win Rate" in i.title)
    assert "deal_stage" in hit.evidence, hit.evidence


def test_win_rate_is_not_claimed_without_both_outcomes():
    """A stage column with no losses in it is a pipeline snapshot."""
    df = pd.DataFrame({
        "sales_rep": ["A", "B"] * 100,
        "deal_stage": ["Prospect", "Qualified"] * 100,
        "deal_amount": np.linspace(1_000, 9_000, 200),
    })
    out = _run(df)
    assert not [t for t in _titles(out) if "Win Rate" in t]


def test_the_rate_is_not_stated_twice_in_the_findings(pipeline):
    """Two engines both compute it. Only one should state the figure —
    the other's finding is about the spread between reps, which is a
    different claim."""
    import re
    stated = [f for f in _run(pipeline)["findings"]
              if re.search(r"win rate (is )?\d", f.lower())]
    assert len(stated) == 1, stated


# ══════════════════════════════════════════════════════════
#  Recommendations the data can carry
# ══════════════════════════════════════════════════════════

def test_no_recommendation_names_a_column_that_is_not_there(pipeline):
    """It closed with "quarterly pricing review — ensure margins are
    healthy per product category" on a file with neither."""
    actions = " ".join(_run(pipeline)["actions"]).lower()
    assert "margin" not in actions, actions
    assert "pricing" not in actions, actions
    assert "customer" not in actions, actions


def test_a_forecast_band_is_not_treated_as_a_product_line(pipeline):
    """`forecast_category` holds Commit / Best Case / Pipeline."""
    assert "forecast_category" not in " ".join(_run(pipeline)["actions"])


def test_recommendations_reference_the_columns_that_exist(pipeline):
    actions = " ".join(_run(pipeline)["actions"])
    assert "sales_rep" in actions, actions


def test_a_bare_frame_asks_for_what_is_missing():
    """No opportunity to recommend anything is itself worth saying."""
    df = pd.DataFrame({"value": np.linspace(1, 100, 50)})
    actions = _run(df)["actions"]
    assert actions
    assert "cannot be measured" in " ".join(actions)


# ══════════════════════════════════════════════════════════
#  Titles read as findings
# ══════════════════════════════════════════════════════════

def test_no_insight_is_titled_like_a_spreadsheet_header(pipeline):
    """"Revenue Overview: Mean 18420 | Median 13450 | Range 1314-186265"
    is a column header, not a finding."""
    for title in _titles(_run(pipeline)):
        assert "|" not in title, title
        assert "Mean " not in title, title
