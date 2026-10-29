"""
tests/test_report_charts.py — every chart in a report shows a meaningful
figure, says what it shows, and is described by its own numbers.

Before: "tenure by department" summed tenure to 1,300 years; the text under
"tenure over time" described "Employee Id across 4 Department groups"; a
correlation matrix included employee_id; charts were drawn in a blue that
had nothing to do with the report's colours.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from app.engines import report_theme
from app.engines.chart_exporter import build_report_charts, fmt


def _hr(n=400, seed=7):
    rng = np.random.default_rng(seed)
    dept = rng.choice(["Sales", "HR", "Engineering", "Marketing"], n)
    return pd.DataFrame({
        "employee_id": np.arange(1, n + 1),
        "department": dept,
        "salary": rng.normal(60_000, 12_000, n).round(0),
        "tenure_years": rng.integers(0, 20, n),
        "hire_date": pd.date_range("2010-01-01", periods=n, freq="9D"),
        "attrition": np.where(rng.random(n) < np.where(dept == "Sales", 0.35, 0.12), "Yes", "No"),
    })


def _orders(n=900, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "order_id": [f"O{i}" for i in range(n)],
        "order_date": pd.date_range("2025-01-01", periods=n, freq="7h"),
        "region": rng.choice(["North", "South", "East"], n),
        "amount": rng.gamma(2, 80, n).round(2),
    })


def test_no_chart_plots_an_identifier():
    for c in build_report_charts(_hr(), domain="hr") + build_report_charts(_orders(), domain="sales"):
        assert not {"employee_id", "order_id"} & set(c.columns), (c.title, c.columns)
        assert "employee" not in c.caption.lower() or "employee_id" not in c.caption


def test_averages_are_averaged_and_totals_totalled():
    hr = {c.title: c for c in build_report_charts(_hr(), domain="hr", max_charts=6)}
    assert any(t.startswith("Average salary by") for t in hr), list(hr)
    assert not any(t.startswith("Total salary") or t.startswith("Total tenure") for t in hr)
    orders = {c.title: c for c in build_report_charts(_orders(), domain="sales", max_charts=6)}
    assert any(t.startswith("Total amount by") for t in orders), list(orders)


def test_each_caption_is_computed_from_the_plotted_figures():
    df = _hr()
    chart = next(c for c in build_report_charts(df, domain="hr", max_charts=6)
                 if c.title.startswith("Average salary by"))
    means = df.groupby("department")["salary"].mean().sort_values(ascending=False)
    assert f"highest in {means.index[0]} ({fmt(means.iloc[0])})" in chart.caption
    assert f"lowest in {means.index[-1]} ({fmt(means.iloc[-1])})" in chart.caption


def test_an_outcome_is_shown_as_a_rate_with_its_real_overall_figure():
    df = _hr()
    rate = next(c for c in build_report_charts(df, domain="hr") if c.kind == "rate")
    overall = (df["attrition"] == "Yes").mean() * 100
    assert rate.title == "Attrition rate by department"
    assert "{:.0f}%".format(overall) in rate.caption
    assert "Sales" in rate.caption.split("highest in")[1][:12]      # the planted effect
    assert "statistically significant" in rate.caption


def test_no_trend_is_drawn_over_a_start_date():
    # Tenure falls with hire date by construction; plotting it is an artefact.
    assert not [c for c in build_report_charts(_hr(), domain="hr", max_charts=6)
                if c.kind == "trend"]


def test_a_trend_leaves_out_part_months_and_names_real_ones():
    df = _orders()          # starts 1 Jan, ends part-way through a month
    trend = next(c for c in build_report_charts(df, domain="sales") if c.kind == "trend")
    last = df["order_date"].max()
    assert last.strftime("%b %Y") not in trend.caption.split("The highest")[0]
    assert "Jan 2025" in trend.caption


def test_no_correlation_grid_when_nothing_correlates():
    assert not [c for c in build_report_charts(_hr(), domain="hr", max_charts=6)
                if c.kind == "correlation"]


@pytest.mark.parametrize("domain", ["hr", "sales", "finance", "ecommerce"])
def test_charts_are_drawn_in_the_report_colours(domain):
    accent = report_theme.get("", domain)["accent"].lstrip("#")
    target = np.array([int(accent[i:i + 2], 16) for i in (0, 2, 4)])
    chart = build_report_charts(_orders(), domain=domain)[0]
    px = np.asarray(Image.open(io.BytesIO(chart.png)).convert("RGB")).reshape(-1, 3)
    share = (np.abs(px.astype(int) - target).sum(axis=1) < 30).mean()
    assert share > 0.03, f"accent {accent} covers only {share:.1%} of the chart"
