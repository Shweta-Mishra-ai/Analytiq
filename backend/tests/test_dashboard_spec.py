"""
Whether the dashboard is about this business or about no business.

Every dataset got the same five tiles — a line, a bar, a pie, a
histogram and a correlation matrix — whatever was in it. That is a chart
grid. A finance director opens a P&L expecting margin, cost structure and
budget variance; an HR director expects headcount and attrition by
department; a sales director expects the funnel and quota attainment. One
fixed set serves none of them, and a reader can tell in two seconds that
nobody decided what the file was about.

Three things are being tested here.

**The tiles follow the domain.** A finance dashboard and an HR dashboard
built from the same code should have almost nothing in common.

**The mark follows the question.** Composition is a donut, a trend is a
line, a comparison across a few groups is a bar, a relationship between
two measures is a scatter. Reaching for a pie because a categorical
column happened to be there is how a chart pack becomes noise — and a
donut of forty slices is not composition, it is a colour wheel.

**A tile the data cannot support is dropped.** Six tiles that mean
something beat twenty that fill a grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.dashboard_spec import MAX_TILES, build_spec, layout_tiles
from app.engines.domain_detect import detect


@pytest.fixture()
def finance():
    rng = np.random.default_rng(5)
    rows = []
    level = {"Retail": 900_000, "Wholesale": 600_000,
             "Services": 350_000, "Support": 80_000}
    for m in pd.date_range("2023-01-31", periods=30, freq="ME"):
        for cc, v in level.items():
            rev = rng.normal(v, v * .08)
            cogs = rev * rng.normal(.6, .03)
            rows.append({"period": m, "cost_centre": cc,
                         "revenue": round(rev, 2), "cogs": round(cogs, 2),
                         "gross_profit": round(rev - cogs, 2),
                         "opex": round(rev * .18, 2),
                         "budget": round(rev * 1.06, 2)})
    df = pd.DataFrame(rows)
    df["ebitda"] = (df.gross_profit - df.opex).round(2)
    return df


@pytest.fixture()
def hr():
    rng = np.random.default_rng(5)
    n = 600
    df = pd.DataFrame({
        "employee_id": range(n),
        "department": rng.choice(["Sales", "Eng", "Ops", "HR"], n),
        "job_title": rng.choice(["Analyst", "Manager"], n),
        "salary": rng.normal(60_000, 12_000, n).round(),
        "tenure_years": rng.integers(0, 20, n),
        "manager_id": rng.integers(1, 40, n),
        "region": rng.choice(["North", "South"], n),
    })
    df["attrition"] = np.where(
        (df.tenure_years < 4) & (rng.random(n) < .5), "Yes", "No")
    return df


@pytest.fixture()
def sales():
    rng = np.random.default_rng(3)
    n = 900
    return pd.DataFrame({
        "opportunity_id": range(n),
        "created_date": pd.to_datetime("2024-01-01")
                        + pd.to_timedelta(rng.integers(0, 540, n), "D"),
        "sales_rep": rng.choice(list("ABCDE"), n),
        "territory": rng.choice(["EMEA", "AMER", "APAC"], n),
        "deal_stage": rng.choice(["Prospect", "Closed Won", "Closed Lost"], n),
        "deal_amount": rng.lognormal(9.5, .8, n).round(2),
        "quota": 250_000.0,
        "product_line": rng.choice(["Core", "Plus", "Enterprise"], n),
    })


def _spec(df):
    return build_spec(df, detect(df).domain)


def _titles(df):
    return [t.title for t in _spec(df)]


# ══════════════════════════════════════════════════════════
#  The tiles follow the domain
# ══════════════════════════════════════════════════════════

def test_a_finance_dashboard_is_about_the_p_and_l(finance):
    joined = " ".join(_titles(finance)).lower()
    assert "budget" in joined, _titles(finance)
    assert "cost_centre" in joined, _titles(finance)
    assert "profit" in joined, _titles(finance)


def test_an_hr_dashboard_is_about_people(hr):
    joined = " ".join(_titles(hr)).lower()
    assert "attrition" in joined, _titles(hr)
    assert "headcount" in joined, _titles(hr)


def test_a_sales_dashboard_is_about_the_number(sales):
    joined = " ".join(_titles(sales)).lower()
    assert "sales_rep" in joined or "territory" in joined, _titles(sales)


def test_two_domains_do_not_get_the_same_dashboard(finance, hr):
    """The defect: the same five tiles whatever the file."""
    assert set(_titles(finance)).isdisjoint(set(_titles(hr)))


def test_every_tile_states_the_question_it_answers(finance, hr, sales):
    for df in (finance, hr, sales):
        for tile in _spec(df):
            assert tile.question.strip(), tile.title
            assert tile.question.rstrip().endswith("?"), tile.question


# ══════════════════════════════════════════════════════════
#  The mark follows the question
# ══════════════════════════════════════════════════════════

def test_a_trend_is_a_line(finance):
    trend = [t for t in _spec(finance) if "over time" in t.title]
    assert trend
    assert all(t.type == "line" for t in trend), [t.type for t in trend]


def test_composition_is_a_donut(finance):
    share = [t for t in _spec(finance) if "split" in t.title.lower()]
    assert share
    assert all(t.type == "pie" for t in share)


def test_a_donut_is_not_used_for_many_categories():
    """Forty slices is a colour wheel, not a composition."""
    rng = np.random.default_rng(8)
    n = 800
    df = pd.DataFrame({
        "order_date": pd.date_range("2024-01-01", periods=n, freq="h"),
        "category": rng.choice([f"Cat {i}" for i in range(30)], n),
        "revenue": rng.normal(100, 20, n),
        "quantity": rng.integers(1, 5, n),
    })
    assert not [t for t in build_spec(df, "ecommerce") if t.type == "pie"]


def test_a_comparison_carries_two_measures(finance):
    """"revenue against budget" plotted revenue alone: the builder took
    one `y`, so the tile promised a comparison and delivered one series."""
    comparisons = [t for t in _spec(finance) if t.type == "comparison"]
    assert comparisons, [t.type for t in _spec(finance)]
    assert comparisons[0].y and comparisons[0].y2
    assert comparisons[0].y != comparisons[0].y2


def test_a_flag_is_compared_as_a_bar_not_scattered(hr):
    """A scatter of pay against a Yes/No column is two vertical strips."""
    for tile in _spec(hr):
        if tile.type == "scatter":
            assert tile.y != "attrition", tile.title


# ══════════════════════════════════════════════════════════
#  Refusing
# ══════════════════════════════════════════════════════════

def test_a_dashboard_never_runs_past_what_a_reader_reads(finance, hr, sales):
    for df in (finance, hr, sales):
        assert len(_spec(df)) <= MAX_TILES


def test_a_thin_file_gets_a_short_dashboard_not_a_padded_one():
    df = pd.DataFrame({"grp": ["a", "b"] * 50,
                       "revenue": np.linspace(1, 100, 100)})
    tiles = build_spec(df, "general")
    assert 1 <= len(tiles) <= 4, [t.title for t in tiles]


def test_a_frame_with_nothing_chartable_returns_nothing():
    df = pd.DataFrame({"row_id": range(200),
                       "ref": [f"R{i}" for i in range(200)]})
    assert build_spec(df, "general") == []


def test_no_tile_is_built_twice(finance, hr, sales):
    for df in (finance, hr, sales):
        keys = [(t.type, t.x, t.y) for t in _spec(df)]
        assert len(keys) == len(set(keys)), keys


def test_an_unknown_domain_falls_back_rather_than_raising(finance):
    assert build_spec(finance, "astrology")


# ══════════════════════════════════════════════════════════
#  Layout
# ══════════════════════════════════════════════════════════

def test_no_row_is_left_half_empty(finance, hr, sales):
    """A six-wide tile followed by a seven-wide one does not fit twelve
    columns, so the seven wrapped and the six sat beside white space.
    Half an empty row reads as a chart that failed to load."""
    for df in (finance, hr, sales):
        placed = layout_tiles(_spec(df))
        rows: dict = {}
        for tile in placed:
            rows.setdefault(tile["gy"], 0)
            rows[tile["gy"]] += tile["w"]
        for gy, total in rows.items():
            assert total == 12, (gy, total, [t["title"] for t in placed])


def test_tiles_never_overlap(finance):
    placed = layout_tiles(_spec(finance))
    seen = set()
    for tile in placed:
        cells = {(tile["gx"] + dx, tile["gy"]) for dx in range(tile["w"])}
        assert not cells & seen, tile["title"]
        seen |= cells


def test_the_layout_carries_what_the_builder_needs(finance):
    for tile in layout_tiles(_spec(finance)):
        assert tile["type"]
        assert "question" in tile
        assert tile["w"] >= 3
        if tile["type"] == "comparison":
            assert tile["y2"], tile
