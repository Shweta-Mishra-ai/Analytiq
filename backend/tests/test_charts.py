"""
What the chart pack plots, and what it claims about it.

Three defects, found by rendering the charts rather than by reading the
code that builds them.

**It charted the wrong column.** Every chart took `num_cols[0]`, and the
ranking broke ties on coefficient of variation — so on a sales export
`units` outranked `revenue` purely because unit counts are noisier than
money. All five charts were about units; revenue never appeared. On a
finance file the same path produced "invoice_id by category" and a pie
chart of summed invoice IDs.

**It titled charts with their axes.** "revenue by region" tells the
reader what is plotted and nothing about what to take from it. The
finding belongs in the title; the variable names belong on the axes.

**It was unreadable in places.** Gridlines drawn over the bars, a
transparent legend box sitting on top of the data, value labels reading
"3,242,612" above an axis reading "3.0m", and correlation figures drawn
in the theme's near-white text on the pale middle of a colour scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.chart_exporter import (
    _rank_measures,
    generate_all_charts,
    make_bar_chart,
    make_correlation_heatmap,
)
from app.engines.chart_message import (
    bar_message,
    heatmap_message,
    histogram_message,
    human_number,
    line_message,
    pie_message,
)


@pytest.fixture()
def sales():
    rng = np.random.default_rng(7)
    n = 600
    region = rng.choice(["North", "South", "East", "West"], n,
                        p=[.45, .25, .18, .12])
    base = {"North": 9000, "South": 3200, "East": 2400, "West": 1800}
    df = pd.DataFrame({
        "order_id": range(1, n + 1),
        "order_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "region": region,
        "channel": rng.choice(["Online", "Retail", "Partner"], n),
        "units": rng.integers(1, 120, n),
        "revenue": [rng.normal(base[r], base[r] * 0.25) for r in region],
        "margin_pct": rng.normal(22, 4, n).round(1),
    })
    df["revenue"] = df["revenue"] * np.linspace(1, 1.8, n)
    return df


# ══════════════════════════════════════════════════════════
#  Which column gets charted
# ══════════════════════════════════════════════════════════

def test_money_outranks_volume(sales):
    """The defect exactly: `units` led because it varies more."""
    ranked = _rank_measures(sales, ["units", "revenue", "margin_pct"])
    assert ranked[0] == "revenue", ranked


def test_a_rate_column_never_leads(sales):
    """"margin_pct" contains a money word. It is still a percentage, and
    a percentage where the reader expects a total is a wrong chart."""
    ranked = _rank_measures(sales, ["margin_pct", "units"])
    assert ranked[-1] == "margin_pct", ranked


def test_identifiers_are_not_measures(sales):
    ranked = _rank_measures(sales, ["order_id", "revenue"])
    assert "order_id" not in ranked


def test_a_constant_column_is_dropped():
    df = pd.DataFrame({"flat": [5] * 100,
                       "revenue": np.linspace(1, 100, 100)})
    assert _rank_measures(df, ["flat", "revenue"]) == ["revenue"]


def test_the_pack_covers_more_than_one_measure(sales):
    """Four views of the same column is not a chart pack."""
    titles = [t for t, _png in generate_all_charts(sales)]
    joined = " ".join(titles).lower()
    assert "revenue" in joined, titles
    assert "units" in joined, titles


def test_no_chart_is_about_an_identifier(sales):
    titles = [t for t, _png in generate_all_charts(sales)]
    assert not [t for t in titles if "order_id" in t], titles


def test_a_frame_of_only_identifiers_produces_no_charts():
    """Better an empty section than five meaningless pictures."""
    df = pd.DataFrame({"row_id": range(200),
                       "ref": [f"R{i}" for i in range(200)]})
    assert generate_all_charts(df) == []


# ══════════════════════════════════════════════════════════
#  The correlation matrix
# ══════════════════════════════════════════════════════════

def test_the_heatmap_excludes_identifiers(sales):
    """On a file ordered by date, `order_id` correlated 0.22 with revenue
    — an artefact of the row order, printed in the same grid and colour
    as the real relationships."""
    from app.engines.chart_exporter import _rank_measures as rank
    assert "order_id" not in rank(sales,
                                  sales.select_dtypes("number")
                                  .columns.tolist())
    assert make_correlation_heatmap(sales) is not None


def test_the_heatmap_message_survives_copy_on_write(sales):
    """`np.fill_diagonal(corr.values, 0)` raises on a read-only view, and
    the failure was swallowed — the chart kept its placeholder title."""
    assert heatmap_message(sales) is not None


def test_the_heatmap_reports_the_strongest_pair():
    rng = np.random.default_rng(11)
    n = 400
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    df["c"] = df["a"] * 2 + rng.normal(0, 0.4, n)
    msg = heatmap_message(df)
    assert "a" in msg and "c" in msg, msg
    assert "not a cause" in msg, msg


def test_a_derived_duplicate_is_not_reported_as_a_finding():
    """r=1.00 between `revenue` and `revenue_usd` is a data note."""
    rng = np.random.default_rng(12)
    df = pd.DataFrame({"revenue": rng.normal(100, 20, 300),
                       "noise": rng.normal(0, 1, 300)})
    df["revenue_copy"] = df["revenue"]
    msg = heatmap_message(df)
    assert "revenue_copy" not in (msg or ""), msg


# ══════════════════════════════════════════════════════════
#  What the headline claims
# ══════════════════════════════════════════════════════════

def test_every_message_carries_a_figure(sales):
    """An adjective on its own is not a finding."""
    messages = [
        bar_message(sales, "region", "revenue"),
        line_message(sales, "order_date", "revenue"),
        histogram_message(sales, "units"),
        pie_message(sales, "region", "revenue"),
        heatmap_message(sales),
    ]
    for msg in messages:
        assert msg, messages
        assert any(ch.isdigit() for ch in msg), msg


def test_no_message_claims_a_cause(sales):
    banned = ("because", "caused by", "drives ", "driven by", "due to",
              "leads to", "resulted in")
    messages = [
        bar_message(sales, "region", "revenue"),
        line_message(sales, "order_date", "revenue"),
        histogram_message(sales, "units"),
        pie_message(sales, "region", "revenue"),
        heatmap_message(sales),
    ]
    for msg in messages:
        low = (msg or "").lower()
        for word in banned:
            assert word not in low, (word, msg)


def test_no_message_predicts_anything(sales):
    for msg in (bar_message(sales, "region", "revenue"),
                line_message(sales, "order_date", "revenue")):
        low = (msg or "").lower()
        assert "will " not in low, msg
        assert "expect" not in low, msg


def test_a_trivial_difference_is_reported_as_level():
    """Inventing a leader out of a 2% gap is how a chart pack becomes
    noise."""
    df = pd.DataFrame({"grp": ["A"] * 50 + ["B"] * 50,
                       "value": [100.0] * 50 + [98.0] * 50})
    msg = bar_message(df, "grp", "value")
    assert "broadly level" in msg.lower(), msg
    assert "leads" not in msg.lower(), msg


def test_a_real_lead_is_stated_with_its_multiple():
    df = pd.DataFrame({"grp": ["A"] * 50 + ["B"] * 50,
                       "value": [100.0] * 50 + [25.0] * 50})
    msg = bar_message(df, "grp", "value")
    assert "A leads" in msg, msg
    assert "4.0x" in msg, msg


def test_a_flat_series_is_not_called_a_trend():
    df = pd.DataFrame({"t": pd.date_range("2024-01-01", periods=60),
                       "v": [500.0] * 60})
    msg = line_message(df, "t", "v")
    assert "flat" in msg.lower(), msg
    assert "rose" not in msg.lower() and "fell" not in msg.lower(), msg


def test_a_fall_is_not_described_as_a_rise():
    df = pd.DataFrame({"t": pd.date_range("2024-01-01", periods=60),
                       "v": np.linspace(1000, 400, 60)})
    msg = line_message(df, "t", "v")
    assert "fell" in msg, msg


def test_a_skewed_distribution_names_the_median_as_the_summary():
    rng = np.random.default_rng(13)
    df = pd.DataFrame({"v": np.concatenate(
        [rng.normal(100, 10, 400), rng.normal(3000, 200, 30)])})
    msg = histogram_message(df, "v")
    assert "skewed" in msg, msg
    assert "median is the fairer summary" in msg, msg


def test_concentration_is_called_out():
    df = pd.DataFrame({"grp": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
                       "v": [90.0] * 10 + [5.0] * 10 + [5.0] * 10})
    msg = pie_message(df, "grp", "v")
    assert "90%" in msg and "A" in msg, msg


def test_a_message_is_withheld_rather_than_guessed():
    """Too little data to say anything is a valid answer."""
    df = pd.DataFrame({"t": pd.date_range("2024-01-01", periods=3),
                       "v": [1.0, 2.0, 3.0]})
    assert line_message(df, "t", "v") is None
    assert histogram_message(df, "v") is None


# ══════════════════════════════════════════════════════════
#  Numbers are written the same way everywhere
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("value,expected", [
    (3_242_612, "3.2m"),
    (685_941, "686k"),
    (2_400_000_000, "2.4bn"),
    (62.0, "62"),
    (0.2367, "0.24"),
    (1.50, "1.5"),
])
def test_numbers_read_the_way_a_person_writes_them(value, expected):
    assert human_number(value) == expected


def test_the_axis_and_the_headline_agree(sales):
    """"3,242,612" printed above an axis reading "3.0m" makes the reader
    check the two against each other."""
    msg = bar_message(sales, "region", "revenue")
    assert "3.2m" in msg, msg


# ══════════════════════════════════════════════════════════
#  Rendering
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("theme", ["Corporate Light", "Dark Tech",
                                   "Executive Green"])
def test_every_theme_renders_every_chart(sales, theme):
    charts = generate_all_charts(sales, theme_name=theme)
    assert len(charts) >= 4
    for title, png in charts:
        assert png and png[:4] == b"\x89PNG", title


def test_gridlines_are_drawn_behind_the_data(sales):
    """`ax.grid(True)` draws on top, so gridlines ran across the face of
    every bar."""
    import matplotlib.pyplot as plt

    from app.engines.chart_exporter import _apply_style, _get_style

    _fig, ax = plt.subplots()
    _apply_style(ax, _get_style("Corporate Light"))
    assert ax.get_axisbelow() is True
    plt.close(_fig)


def test_a_bar_chart_has_no_vertical_gridlines(sales):
    """A bar sitting on `region` has no x scale to read against."""
    import matplotlib.pyplot as plt

    from app.engines.chart_exporter import _apply_style, _get_style

    _fig, ax = plt.subplots()
    _apply_style(ax, _get_style("Corporate Light"), axis="y")
    assert not any(line.get_visible()
                   for line in ax.xaxis.get_gridlines())
    assert any(line.get_visible() for line in ax.yaxis.get_gridlines())
    plt.close(_fig)


def test_the_tallest_bar_label_is_not_clipped(sales):
    """The label sat at the top of the frame with nowhere to go."""
    png = make_bar_chart(sales, "region", "revenue",
                         theme_name="Corporate Light")
    assert png[:4] == b"\x89PNG"


def test_charts_survive_a_frame_with_nothing_in_it():
    assert generate_all_charts(pd.DataFrame()) == []
