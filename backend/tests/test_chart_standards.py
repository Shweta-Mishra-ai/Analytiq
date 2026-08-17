"""
The chart conventions the pack is read by.

Two of them are house style; one changes what a reader concludes.

**Semantic notation (IBCS).** A bar of revenue and a bar of budget were
drawn as the accent colour and the second series colour — the same
treatment two product categories would get. That says "two things",
not "what happened against what was committed", and it sends the reader
to the legend to find out which is which. The standard fixes the
appearance of each kind of series: an actual is solid, a plan is an
outline, a forecast is hatched, a prior period is grey. It is the same
on every chart in the pack, so it is learned once.

**Zero baseline.** Four bars between 980 and 1,020, left to autoscale,
fill the plot: the shortest looks a third of the tallest and the
difference is four per cent. That is not a styling preference. A reader
comparing bar lengths — which is the only way a bar chart is read —
takes a false conclusion off the page, and the axis labels that would
correct it are the part nobody checks. Bars start at zero. Bars that go
negative keep an autoscaled floor, because clamping those hides them.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from app.engines.chart_engine import notation, series_kind


# ══════════════════════════════════════════════════════════
#  What a series is, from its name
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("column,kind", [
    ("revenue", "actual"),
    ("net_sales", "actual"),
    ("budget", "plan"),
    ("planned_spend", "plan"),
    ("sales_target", "plan"),
    ("quota", "plan"),
    ("annual_goal", "plan"),
    ("forecast_revenue", "forecast"),
    ("projected_spend", "forecast"),
    ("estimated_cost", "forecast"),
    ("revenue_prior", "previous"),
    ("previous_quarter", "previous"),
    ("sales_last_year", "previous"),
    ("revenue_ly", "previous"),
    ("revenue_yoy", "previous"),
])
def test_a_series_is_classified_from_its_column_name(column, kind):
    assert series_kind(column) == kind


@pytest.mark.parametrize("column", [
    "monthly_revenue",   # "ly" inside "monthly"
    "supply_cost",       # "py" inside "supply"
    "quarterly_units",
    "employee_count",    # "ee"/"ploy" near-misses
    "target_audience_size",
])
def test_a_word_inside_another_word_is_not_a_match(column):
    """The first version matched substrings, so `monthly_revenue` came
    out as a prior period and was drawn in grey — a column of actuals
    presented as last year's."""
    assert series_kind(column) in ("actual", "plan"), column


def test_monthly_revenue_is_an_actual():
    assert series_kind("monthly_revenue") == "actual"


def test_an_unrecognised_name_is_treated_as_an_actual():
    """Drawing an unknown series as a hollow plan bar asserts something
    the column name does not support."""
    assert series_kind("xyz_column_42") == "actual"
    assert series_kind("") == "actual"


def test_a_forecast_of_a_budget_reads_as_the_forecast():
    """Both words are present and only one appearance is available; the
    projection is the weaker claim, so it wins."""
    assert series_kind("forecast_budget") == "forecast"


# ══════════════════════════════════════════════════════════
#  What each kind looks like
# ══════════════════════════════════════════════════════════

def test_a_plan_is_an_outline_not_a_fill():
    marker = notation("plan", "#1B4FD8")
    assert marker["color"] in ("rgba(0,0,0,0)", "rgba(0, 0, 0, 0)")
    assert marker["line"]["color"] == "#1B4FD8"
    assert marker["line"]["width"] >= 1


def test_a_forecast_is_hatched_and_lighter():
    marker = notation("forecast", "#1B4FD8")
    assert marker["pattern"]["shape"], marker
    assert marker["opacity"] < 1


def test_a_prior_period_is_grey_not_a_second_accent():
    """A second accent colour reads as a second category. Last year is
    background against which this year is read."""
    marker = notation("previous", "#1B4FD8")
    assert marker["color"].lower() != "#1b4fd8"
    r, g, b = (int(marker["color"].lstrip("#")[i:i + 2], 16)
               for i in (0, 2, 4))
    assert max(r, g, b) - min(r, g, b) < 25, marker["color"]


def test_an_actual_is_a_plain_solid_fill():
    marker = notation("actual", "#1B4FD8")
    assert marker["color"] == "#1B4FD8"
    assert "pattern" not in marker


def test_the_four_kinds_are_all_distinguishable():
    import json

    looks = [json.dumps(notation(k, "#1B4FD8"), sort_keys=True)
             for k in ("actual", "plan", "forecast", "previous")]
    assert len(set(looks)) == 4, looks


# ══════════════════════════════════════════════════════════
#  The notation reaches the figure
# ══════════════════════════════════════════════════════════

@pytest.fixture()
def against_budget():
    rng = np.random.default_rng(11)
    n = 300
    region = rng.choice(["North", "South", "East", "West"], n)
    revenue = rng.normal(50_000, 8_000, n).round(2)
    return pd.DataFrame({"region": region, "revenue": revenue,
                         "budget": (revenue * 1.05).round(2)})


def test_the_plan_bar_in_the_figure_is_unfilled(against_budget):
    from app.engines.chart_engine import make_comparison

    fig = make_comparison(against_budget, "region", "revenue", "budget")
    plan = next(t for t in fig.data if "budget" in str(t.name))
    assert plan.marker.color in ("rgba(0,0,0,0)", "rgba(0, 0, 0, 0)")
    assert plan.marker.line.color


def test_the_actual_bar_in_the_figure_is_solid(against_budget):
    from app.engines.chart_engine import make_comparison

    fig = make_comparison(against_budget, "region", "revenue", "budget")
    actual = next(t for t in fig.data if str(t.name) == "revenue")
    assert actual.marker.color not in ("rgba(0,0,0,0)", None)


def test_the_legend_says_which_series_is_the_commitment(against_budget):
    """The notation carries it for a reader who knows the standard; the
    name carries it for one who does not."""
    from app.engines.chart_engine import make_comparison

    fig = make_comparison(against_budget, "region", "revenue", "budget")
    assert any("plan" in str(t.name) for t in fig.data), \
        [t.name for t in fig.data]


# ══════════════════════════════════════════════════════════
#  The variance is written the way it is spoken
# ══════════════════════════════════════════════════════════

def test_a_shortfall_is_written_as_below_not_as_a_minus_sign():
    """"revenue came in -7% against budget" makes the reader decode a
    sign before they can read the sentence, and the sign is the point."""
    from app.engines.chart_message import comparison_message

    text = comparison_message("revenue", "budget", 19_900_000, 21_500_000)
    assert "below budget" in text, text
    assert "-7" not in text and "+" not in text, text


def test_an_overshoot_says_above():
    from app.engines.chart_message import comparison_message

    text = comparison_message("revenue", "budget", 21_500_000, 19_900_000)
    assert "above budget" in text, text


def test_landing_on_plan_is_not_dressed_up_as_a_variance():
    """Rounding 0.2% to "0% below budget" invites the reader to look for
    a gap that is not there."""
    from app.engines.chart_message import comparison_message

    text = comparison_message("revenue", "budget", 1_000_000, 1_000_800)
    assert "landed on budget" in text, text


def test_the_worst_group_is_named_when_one_is_behind():
    from app.engines.chart_message import comparison_message

    text = comparison_message("revenue", "budget", 900, 1_000, "South")
    assert "South furthest behind" in text, text


def test_a_plan_of_zero_says_nothing_rather_than_dividing_by_it():
    from app.engines.chart_message import comparison_message

    assert comparison_message("revenue", "budget", 900, 0) == ""


def test_both_chart_paths_use_the_same_sentence(against_budget):
    """The page and the screen describing one comparison differently is
    the defect a shared helper exists to prevent."""
    from app.engines.chart_engine import make_comparison

    fig = make_comparison(against_budget, "region", "revenue", "budget")
    assert "below budget" in str(fig.layout.title.text), fig.layout.title.text


# ══════════════════════════════════════════════════════════
#  Zero baseline — the interactive charts
# ══════════════════════════════════════════════════════════

def _tight() -> pd.DataFrame:
    """Four groups whose totals differ by four per cent."""
    return pd.DataFrame({
        "region": ["North", "South", "East", "West"],
        "revenue": [980.0, 1000.0, 1010.0, 1020.0],
    })


def test_a_bar_chart_starts_at_zero():
    from app.engines.chart_engine import make_bar

    fig = make_bar(_tight(), "region", "revenue")
    assert fig.layout.yaxis.range[0] == 0, fig.layout.yaxis.range


def test_the_four_per_cent_difference_looks_like_four_per_cent():
    from app.engines.chart_engine import make_bar

    fig = make_bar(_tight(), "region", "revenue")
    low, high = fig.layout.yaxis.range
    drawn = (1020 - low) / (980 - low)      # ratio of drawn bar lengths
    assert drawn < 1.1, (fig.layout.yaxis.range, drawn)


def test_negative_bars_are_not_clamped_away():
    """A cost centre that gave money back has a bar below the line.
    Forcing the floor to zero would draw it as nothing."""
    from app.engines.chart_engine import make_bar

    df = pd.DataFrame({"centre": ["A", "B", "C"],
                       "variance": [-4000.0, 1500.0, 3000.0]})
    fig = make_bar(df, "centre", "variance")
    lo = fig.layout.yaxis.range
    assert lo is None or lo[0] < 0, lo


def test_a_comparison_starts_at_zero(against_budget):
    from app.engines.chart_engine import make_comparison

    fig = make_comparison(against_budget, "region", "revenue", "budget")
    assert fig.layout.yaxis.range[0] == 0, fig.layout.yaxis.range


def test_the_headroom_leaves_the_tallest_bar_inside_the_frame():
    from app.engines.chart_engine import make_bar

    fig = make_bar(_tight(), "region", "revenue")
    assert fig.layout.yaxis.range[1] > 1020


# ══════════════════════════════════════════════════════════
#  A truncated trend says so
# ══════════════════════════════════════════════════════════

def _trend(values) -> pd.DataFrame:
    return pd.DataFrame({
        "month": pd.date_range("2024-01-31", periods=len(values), freq="ME"),
        "revenue": [float(v) for v in values],
    })


def test_a_truncated_axis_is_disclosed():
    """A 4% rise drawn from a floor of 975 climbs the height of the tile.
    The picture is not wrong, but it is not readable without knowing
    where the axis starts."""
    from app.engines.chart_engine import make_line

    fig = make_line(_trend([980, 995, 1005, 1010, 1015, 1020]),
                    "month", "revenue")
    notes = [a.text for a in fig.layout.annotations]
    assert any("not zero" in str(n) for n in notes), notes


def test_a_chart_that_reaches_zero_says_nothing():
    """The note is a disclosure, not decoration; on an axis that already
    starts at zero it is one more thing to read for no information."""
    from app.engines.chart_engine import make_line

    fig = make_line(_trend([120, 400, 90, 780, 300, 950]), "month", "revenue")
    notes = [str(a.text) for a in fig.layout.annotations]
    assert not any("not zero" in n for n in notes), notes


@pytest.mark.parametrize("values,disclosed", [
    ([980, 1000, 1020], True),        # 4% of the level
    ([100, 300, 900], False),         # the axis reaches zero by itself
    ([-50, 20, 90], False),           # crosses zero, so zero is on screen
    ([0, 10, 20], False),
])
def test_the_disclosure_rule(values, disclosed):
    from app.engines.chart_engine import truncated_scale

    assert (truncated_scale(values) is not None) is disclosed, values


def test_the_disclosure_survives_unusable_values():
    from app.engines.chart_engine import truncated_scale

    assert truncated_scale([]) is None
    assert truncated_scale(["a", "b"]) is None


# ══════════════════════════════════════════════════════════
#  Zero baseline — the printed charts
# ══════════════════════════════════════════════════════════

def _bar_pixel_heights(png: bytes, theme: str = "Corporate Light") -> list[int]:
    """Measure the bars as drawn, in pixels.

    Asserting on `set_ylim` would test the call, not the picture. The
    picture is what the reader takes the conclusion from, so it is what
    is measured. The fill is the theme accent at 85% over the plot
    background, which is a colour we can compute rather than guess at,
    and each run of columns carrying it is one bar.
    """
    from PIL import Image

    from app.engines.chart_exporter import _get_colors, _get_style

    def _rgb(value: str) -> np.ndarray:
        value = value.lstrip("#")
        return np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)], float)

    accent = _rgb(_get_colors(theme)[0])
    paper = _rgb(_get_style(theme)["axes.facecolor"])
    fill = np.floor(accent * 0.85 + paper * 0.15).astype(int)

    arr = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)
    mask = np.all(np.abs(arr - fill) <= 1, axis=-1)
    assert mask.any(), "no bars found at {}".format(fill)
    heights = []
    run = []
    for col in range(mask.shape[1]):
        rows = np.flatnonzero(mask[:, col])
        if rows.size > 2:
            run.append(int(rows.max() - rows.min() + 1))
        elif run:
            heights.append(max(run))
            run = []
    if run:
        heights.append(max(run))
    return heights


def test_the_printed_bars_are_drawn_in_proportion():
    """The chart that goes in the PDF is a different code path from the
    one on screen, and it had the same defect."""
    from app.engines.chart_exporter import make_bar_chart

    png = make_bar_chart(_tight(), "region", "revenue")
    heights = _bar_pixel_heights(png)
    assert len(heights) == 4, heights
    ratio = max(heights) / min(heights)
    assert ratio < 1.1, (heights, ratio)


def test_a_printed_bar_chart_of_unequal_values_still_separates_them():
    """Zero-basing must not flatten a real difference into nothing."""
    from app.engines.chart_exporter import make_bar_chart

    df = pd.DataFrame({"region": ["A", "B", "C", "D"],
                       "revenue": [100.0, 400.0, 700.0, 1000.0]})
    heights = _bar_pixel_heights(make_bar_chart(df, "region", "revenue"))
    assert len(heights) == 4, heights
    assert max(heights) / min(heights) > 5, heights


def test_a_printed_chart_with_negatives_keeps_its_floor():
    from app.engines.chart_exporter import make_bar_chart

    df = pd.DataFrame({"centre": ["A", "B", "C"],
                       "variance": [-4000.0, 1500.0, 3000.0]})
    png = make_bar_chart(df, "centre", "variance")
    assert png[:4] == b"\x89PNG"


def test_the_printed_comparison_carries_the_same_notation(against_budget):
    from app.engines.chart_exporter import make_comparison_chart

    png = make_comparison_chart(against_budget, "region", "revenue", "budget")
    assert png[:4] == b"\x89PNG"


def test_the_printed_trend_is_drawn_against_zero(monkeypatch):
    """The shaded area under the printed trend line is filled down to
    zero, which puts zero in the axis limits. That is the behaviour the
    reader depends on, so it is asserted rather than assumed."""
    from app.engines import chart_exporter

    limits = []
    original = chart_exporter.fig_to_bytes

    def spy(fig, *args, **kwargs):
        limits.append(fig.axes[0].get_ylim())
        return original(fig, *args, **kwargs)

    monkeypatch.setattr(chart_exporter, "fig_to_bytes", spy)
    df = pd.DataFrame({
        "month": pd.date_range("2024-01-31", periods=12, freq="ME"),
        "revenue": [980, 990, 1000, 1005, 1010, 1008,
                    1012, 1015, 1018, 1020, 1019, 1021],
    })
    chart_exporter.make_line_chart(df, "month", "revenue")
    assert limits, "chart was not rendered"
    assert limits[0][0] <= 0, limits[0]


def test_the_printed_comparison_labels_the_plan(against_budget, monkeypatch):
    """Same legend wording as the interactive chart, so the page and the
    screen do not describe the same series differently."""
    import matplotlib.axes

    labels = []
    original = matplotlib.axes.Axes.bar

    def spy(self, *args, **kwargs):
        labels.append(kwargs.get("label"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "bar", spy)
    from app.engines.chart_exporter import make_comparison_chart

    make_comparison_chart(against_budget, "region", "revenue", "budget")
    assert any(lbl and "plan" in lbl for lbl in labels), labels
