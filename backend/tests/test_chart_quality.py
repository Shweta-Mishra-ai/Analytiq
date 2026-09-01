"""
Chart selection and rendering quality.

These pin the behaviours recovered from dataforge-ai in the chart port, and
the two Analytiq guarantees that port initially broke: never chart an
identifier, and return nothing rather than something meaningless.
"""
import numpy as np
import pandas as pd
import pytest

from app.engines.chart_exporter import (
    _agg_for_metric, _best_metric_by_category, _is_grouping_column,
    _pick_best_metric, _pretty, _rank_measures, generate_all_charts,
)
from app.engines import chart_engine


@pytest.fixture
def sales_df():
    r = np.random.default_rng(3)
    n = 400
    region = r.choice(["North", "South", "East", "West"], n)
    return pd.DataFrame({
        "order_id": np.arange(n),
        "region": region,
        "revenue": np.round(r.uniform(100, 9000, n) *
                            np.where(region == "North", 2.5, 1.0), 2),
        "units": r.integers(1, 60, n),
        "satisfaction": r.integers(1, 6, n),
        "gender": r.choice(["M", "F"], n),
    })


# ── labels ────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("EnvironmentSatisfaction", "Environment Satisfaction"),
    ("monthly_income", "Monthly Income"),
    ("YearsWithCurrManager", "Years With Curr Manager"),
    ("MRP", "MRP"),
])
def test_pretty_makes_human_labels(raw, expected):
    """Chart titles reach a client PDF. Raw column names read as a bug."""
    assert _pretty(raw) == expected


def test_chart_titles_contain_no_raw_snake_case(sales_df):
    for title, _, _spec in generate_all_charts(sales_df, max_charts=6):
        assert "_" not in title, f"raw column name in chart title: {title!r}"


# ── aggregation ───────────────────────────────────────────

@pytest.mark.parametrize("col,expected_agg", [
    ("revenue", "sum"), ("units", "sum"), ("spend", "sum"),
    ("satisfaction", "mean"), ("rating", "mean"), ("nps", "mean"),
    # Per-entity attributes: a group total of an age says nothing.
    ("age", "mean"), ("tenure_years", "mean"),
])
def test_metric_aggregation_matches_metric_type(col, expected_agg):
    """Summing a rating produces 'Total Satisfaction: 4,410'. Analytiq's
    bar chart summed unconditionally before this port."""
    agg, _ = _agg_for_metric(col)
    assert agg == expected_agg


def test_score_metrics_are_flagged_as_scores():
    assert _agg_for_metric("satisfaction")[1] is True
    assert _agg_for_metric("revenue")[1] is False


# ── identifiers ───────────────────────────────────────────

def test_no_charts_when_every_numeric_column_is_an_identifier():
    df = pd.DataFrame({
        "record_id": range(60),
        "customer_id": range(1000, 1060),
        "label": ["a", "b"] * 30,
    })
    assert generate_all_charts(df) == []


def test_pick_best_metric_returns_none_rather_than_an_identifier():
    df = pd.DataFrame({"record_id": range(80), "order_id": range(80)})
    assert _pick_best_metric(list(df.columns), df=df, cat_cols=[]) is None


def test_rank_measures_puts_business_metrics_first(sales_df):
    ranked = _rank_measures(sales_df, ["order_id", "satisfaction", "revenue"])
    assert "order_id" not in ranked
    assert ranked[0] == "revenue"


def test_identifier_columns_never_become_chart_metrics(sales_df):
    for title, _, _spec in generate_all_charts(sales_df, max_charts=6):
        assert "Order Id" not in title


# ── dimension selection ───────────────────────────────────

def test_near_unique_column_is_not_a_grouping_dimension():
    """On a small frame an ID column can land inside a 'reasonable'
    cardinality window and win spread-based selection outright."""
    df = pd.DataFrame({"emp_id": [f"E{i}" for i in range(18)],
                       "value": range(18)})
    assert not _is_grouping_column(df, "emp_id")


def test_sensitive_dimensions_are_not_auto_charted(sales_df):
    """An automatically generated 'revenue by gender' comparison is
    spurious and can be discriminatory."""
    pair = _best_metric_by_category(
        sales_df, ["revenue", "units"], ["region", "gender"])
    assert pair is not None
    assert pair[0] != "gender"


def test_best_pair_finds_the_dimension_that_actually_varies(sales_df):
    """North is built to earn 2.5x. Selection should find region."""
    cat, metric, spread = _best_metric_by_category(
        sales_df, ["revenue", "units"], ["region"])
    assert cat == "region"
    assert spread > 0.1


# ── rendering ─────────────────────────────────────────────

def test_charts_render_to_real_png_bytes(sales_df):
    charts = generate_all_charts(sales_df, max_charts=6)
    assert charts
    for title, data, spec in charts:
        assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{title} is not a PNG"
        assert len(data) > 5000, f"{title} rendered suspiciously small"
        # The spec is what stops a chart being captioned with another
        # chart's narrative, so every chart must carry one.
        assert spec.kind, f"{title} has no chart spec"


def test_correlation_heatmap_survives_a_read_only_frame():
    """np.fill_diagonal on .values raises on a read-only array; dataforge's
    version caught it and silently degraded to the first ten columns."""
    from app.engines.chart_exporter import make_correlation_heatmap
    n = 200
    df = pd.DataFrame({f"m{i}": np.random.rand(n) * (i + 1) for i in range(14)})
    for c in df.columns:
        df[c].values.flags.writeable = False
    assert len(make_correlation_heatmap(df, "Correlation Matrix",
                                        "Corporate Light")) > 5000


@pytest.mark.parametrize("theme", ["Corporate Light", "Dark Tech"])
def test_charts_render_in_every_theme(sales_df, theme):
    assert generate_all_charts(sales_df, theme, max_charts=3)


# ── interactive (plotly) selection ────────────────────────

def test_interactive_charts_exclude_identifiers(sales_df):
    for title, _ in chart_engine.recommend_charts(sales_df, "sales"):
        assert "order_id" not in title


def test_interactive_pie_only_for_additive_metrics():
    """A pie of an average is not a part-of-whole relationship."""
    assert chart_engine._is_pie_valid("revenue")
    assert chart_engine._is_pie_valid("units")
    assert not chart_engine._is_pie_valid("satisfaction_score")
    assert not chart_engine._is_pie_valid("churn_rate")


def test_interactive_titles_carry_no_emoji(sales_df):
    for title, fig in chart_engine.recommend_charts(sales_df, "sales"):
        blob = title + str(fig.layout.title.text or "")
        assert all(ord(ch) < 0x2190 for ch in blob), \
            f"emoji in chart title: {title!r}"


def test_domain_metric_priority_comes_from_the_registry(sales_df):
    """A sales dataset should lead with revenue, not the first numeric
    column. The priority list lives on DomainSpec.chart_metrics."""
    cols = chart_engine._get_analysis_columns(sales_df)
    assert chart_engine._pick_primary_metric(
        cols, "sales", df_ref=sales_df) == "revenue"


def test_safe_pct_gap_refuses_undefined_comparisons():
    assert chart_engine.safe_pct_gap(50, 0) == "n/a"
    assert chart_engine.safe_pct_gap(50, float("nan")) == "n/a"
    assert chart_engine.safe_pct_gap(1e9, 0.0001) == "n/a"
    assert chart_engine.safe_pct_gap(110, 100) == "+10.0%"
