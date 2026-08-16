"""
Whether a chart looks like it belongs to the document it is printed in.

**The report.** There are five report themes, each with its own accent —
`#1B4FD8` for Corporate Light, `#F4511E` for Ecommerce Orange, `#2E7D32`
for Sales Green. The chart exporter carried three palettes of its own,
keyed on theme names that mostly did not exist: `"Executive Green"` is
not a theme (the green one is `"Sales Green"`), so that palette was dead
code and every light report got the same `#1a4a8a` blue. An e-commerce
report therefore had an orange cover, orange headings, orange section
badges — and blue bars. The HR report drew `#1976D2` headings above
`#1a4a8a` bars: two blues close enough to read as a mistake rather than
a choice.

**The dashboard.** Every Plotly figure hardcoded
`paper_bgcolor="#07080f"` and `plot_bgcolor="#0e0f1a"` — a near-black
that is not the app's panel colour (`#101215`) — in `JetBrains Mono`,
which the interface does not use anywhere. Each tile showed a dark
rectangle sitting on a slightly different dark panel.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.io as pio
import pytest

from app.engines import chart_engine
from app.engines.chart_exporter import _get_colors, _get_style
from app.engines.pdf_builder import THEMES


@pytest.fixture()
def frame():
    rng = np.random.default_rng(7)
    n = 400
    region = rng.choice(["North", "South", "East"], n)
    level = {"North": 9000, "South": 3200, "East": 2400}
    return pd.DataFrame({
        "order_id": range(n),
        "order_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "region": region,
        "channel": rng.choice(["Online", "Retail"], n),
        "units": rng.integers(1, 120, n),
        "revenue": [rng.normal(level[r], level[r] * .2) for r in region],
    })


# ══════════════════════════════════════════════════════════
#  The report
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("theme", sorted(THEMES))
def test_the_series_colour_is_the_theme_accent(theme):
    """A single-series chart — most of them — must draw in exactly the
    colour the headings use."""
    assert _get_colors(theme)[0] == THEMES[theme]["accent"], theme


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_the_figure_is_drawn_on_the_theme_page(theme):
    """Otherwise a dark theme shows a white rectangle per chart."""
    assert _get_style(theme)["figure.facecolor"] == THEMES[theme]["page_bg"]


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_chart_text_uses_the_theme_text_colour(theme):
    style = _get_style(theme)
    assert style["text.color"] == THEMES[theme]["text"]
    assert style["xtick.color"] == THEMES[theme]["text_muted"]


def test_two_themes_with_different_accents_produce_different_charts():
    """The check that would have caught the dead green palette."""
    assert _get_colors("Ecommerce Orange")[0] != _get_colors("HR Blue")[0]
    assert _get_colors("Sales Green")[0] != _get_colors("Corporate Light")[0]


def test_an_unknown_theme_falls_back_rather_than_raising():
    assert _get_colors("Executive Green")[0] == THEMES["Corporate Light"]["accent"]


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_every_theme_renders_a_chart(frame, theme):
    from app.engines.chart_exporter import generate_all_charts

    charts = generate_all_charts(frame, theme_name=theme)
    assert charts
    for _title, png in charts:
        assert png[:4] == b"\x89PNG"


# ══════════════════════════════════════════════════════════
#  The dashboard
# ══════════════════════════════════════════════════════════

def _layout(fig):
    return json.loads(pio.to_json(fig))["layout"]


def test_a_tile_does_not_paint_its_own_background(frame):
    """The tile already has a background, a border and a radius. A figure
    that paints its own rectangle inside that shows a seam."""
    for _name, fig in chart_engine.recommend_charts(frame):
        layout = _layout(fig)
        assert layout["paper_bgcolor"] == "rgba(0,0,0,0)"
        assert layout["plot_bgcolor"] == "rgba(0,0,0,0)"


def test_charts_use_the_interface_font(frame):
    """`JetBrains Mono` is not loaded by the app; the figures asked for
    it anyway, so every chart fell back to a system monospace."""
    for _name, fig in chart_engine.recommend_charts(frame):
        family = _layout(fig)["font"]["family"]
        assert "IBM Plex" in family, family


def test_charts_use_the_interface_palette(frame):
    """The accent token, not a fourth blue."""
    assert chart_engine.PALETTE[0] == "#5b8def"
    for _name, fig in chart_engine.recommend_charts(frame):
        assert _layout(fig)["colorway"][0] == "#5b8def"


def test_no_title_carries_an_emoji(frame):
    """"📊 revenue by region" is not what a BI tool looks like."""
    for _name, fig in chart_engine.recommend_charts(frame):
        title = (_layout(fig).get("title") or {}).get("text", "")
        assert not any(ord(ch) > 0x2100 for ch in title), title


def test_a_bar_is_not_coloured_by_its_own_height(frame):
    """`color=y` on a continuous scale adds a legend that repeats the
    y-axis and makes every bar chart look like a heatmap."""
    fig = chart_engine.make_bar(frame, "region", "revenue")
    assert "coloraxis" not in _layout(fig), _layout(fig).keys()


def test_tiles_are_titled_with_their_finding(frame):
    fig = chart_engine.make_bar(frame, "region", "revenue")
    title = (_layout(fig).get("title") or {}).get("text", "")
    assert "leads revenue" in title, title
    assert "revenue by region" in title, title


# ══════════════════════════════════════════════════════════
#  Both paths agree about what the measure is
# ══════════════════════════════════════════════════════════

def test_the_dashboard_and_the_export_rank_measures_the_same(frame):
    from app.engines.chart_exporter import _rank_measures

    cols = frame.select_dtypes(include="number").columns.tolist()
    assert chart_engine.rank_measures(frame) == _rank_measures(frame, cols)


def test_neither_path_charts_an_identifier(frame):
    assert "order_id" not in chart_engine.rank_measures(frame)


def test_the_recommended_pack_covers_more_than_one_measure(frame):
    titles = " ".join(
        (_layout(f).get("title") or {}).get("text", "")
        for _n, f in chart_engine.recommend_charts(frame)
    ).lower()
    assert "revenue" in titles
    assert "units" in titles


def test_a_frame_of_only_identifiers_produces_no_tiles():
    df = pd.DataFrame({"row_id": range(200),
                       "ref": [f"R{i}" for i in range(200)]})
    assert chart_engine.recommend_charts(df) == []
