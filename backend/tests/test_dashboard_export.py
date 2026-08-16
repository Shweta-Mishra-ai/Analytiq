"""
The dashboard as one file the client can keep.

A PDF is finished: the reader sees the cuts we chose, and wanting the
same view for one region means coming back to ask. Half of what a client
commissions analysis for is the ability to poke at it.

The constraint that makes this worth anything is that it has to work
from a file:// path with no server, no network and no install — that is
what makes it something you attach to an email rather than something you
host. So the tests here are mostly about what the file must NOT contain:
no script src, no stylesheet link, no font fetch, nothing that turns
into a blank page on a plane or behind a corporate proxy.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from app.engines.chart_engine import recommend_charts
from app.engines.dashboard_export import MAX_EMBEDDED_ROWS, build_dashboard_html


@pytest.fixture()
def frame():
    rng = np.random.default_rng(7)
    n = 600
    region = rng.choice(["North", "South", "East", "West"], n,
                        p=[.45, .25, .18, .12])
    level = {"North": 9000, "South": 3200, "East": 2400, "West": 1800}
    df = pd.DataFrame({
        "order_id": range(n),
        "order_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "region": region,
        "channel": rng.choice(["Online", "Retail", "Partner"], n),
        "units": rng.integers(1, 120, n),
        "revenue": [rng.normal(level[r], level[r] * .25) for r in region],
    })
    df["revenue"] = df.revenue * np.linspace(1, 1.8, n)
    return df


@pytest.fixture()
def page(frame):
    tiles = [{"title": t, "figure": f, "w": 12 if i == 0 else 6}
             for i, (t, f) in enumerate(recommend_charts(frame))]
    kpis = [{"label": "Σ revenue", "value": float(frame.revenue.sum()),
             "format": "num"},
            {"label": "Rows", "value": len(frame), "format": "int"},
            {"label": "Missing %", "value": 0.0, "format": "pct"}]
    return build_dashboard_html(frame, tiles, kpis, title="Q1 Review",
                                subtitle="600 orders",
                                prepared_by="S. Mishra")


# ══════════════════════════════════════════════════════════
#  It opens with nothing else present
# ══════════════════════════════════════════════════════════

def test_nothing_is_loaded_from_a_url(page):
    """A CDN reference makes the file useless offline, and tells whoever
    runs that CDN every time the client opens their report."""
    assert not re.findall(r"<script[^>]*\ssrc=", page)
    assert not re.findall(r"<link[^>]*\shref=", page)
    assert not re.findall(r"<img[^>]*\ssrc=", page)


def test_the_plotting_runtime_is_inlined(page):
    assert "Plotly" in page
    assert "Plotly.newPlot" in page
    # The bundle, not a stub.
    assert len(page) > 500_000


def test_it_is_a_complete_document(page):
    assert page.lstrip().startswith("<!doctype html>")
    assert page.rstrip().endswith("</html>")
    assert "<title>" in page


# ══════════════════════════════════════════════════════════
#  What it shows
# ══════════════════════════════════════════════════════════

def test_every_tile_is_rendered(frame, page):
    """Counted by tile container, not by searching the whole document —
    the inlined Plotly bundle contains its own reference to newPlot."""
    expected = len(recommend_charts(frame))
    assert len(re.findall(r'<div id="t\d+"', page)) == expected


def test_the_kpi_values_are_written_not_printed(page):
    """"4523891.44" in a KPI card is a number nobody reads."""
    assert "4.5m" in page
    assert "4523891" not in page


def test_slicers_are_built_from_the_categorical_fields(page):
    assert 'data-col="region"' in page
    assert 'data-value="North"' in page


def test_identifier_columns_are_not_offered_as_slicers(page):
    assert 'data-col="order_id"' not in page


def test_the_footer_carries_the_provenance(page):
    """A snapshot that stopped matching its source with nothing to say so
    is worse than no export."""
    assert "600 rows" in page
    assert "Generated" in page


def test_the_preparer_is_named_and_the_tool_is_not(page):
    """The person delivering the work signs it. The app never names
    itself or the model that produced the figures.

    Checked against the visible document only. The minified plotting
    bundle contains identifiers like `gpt` as ordinary variable names,
    and matching those is a test that fails on a library upgrade rather
    than on anything a reader would ever see."""
    body = re.sub(r"<script>.*?</script>", "", page, flags=re.S)
    assert "S. Mishra" in body
    for name in ("claude", "anthropic", "openai", "gemini", "chatgpt"):
        assert name not in body.lower(), name


# ══════════════════════════════════════════════════════════
#  Honest about what a static file can do
# ══════════════════════════════════════════════════════════

def test_a_slicer_says_what_it_does_not_recompute(page):
    """A control that silently does nothing is worse than no control.
    A static file cannot re-aggregate, so selecting a value highlights
    and says as much."""
    assert "Highlighting" in page
    assert "full dataset" in page


def test_the_title_is_escaped(frame):
    """A filename is user input and lands in the document head."""
    tiles = [{"title": t, "figure": f}
             for t, f in recommend_charts(frame)][:1]
    page = build_dashboard_html(
        frame, tiles, [], title='<script>alert(1)</script>')
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_a_slicer_value_is_escaped():
    rng = np.random.default_rng(2)
    n = 100
    df = pd.DataFrame({
        "grp": ['<img onerror=alert(1)>', "safe"] * (n // 2),
        "revenue": rng.normal(100, 10, n),
    })
    page = build_dashboard_html(df, [], [], title="t")
    assert "<img onerror" not in page


# ══════════════════════════════════════════════════════════
#  Degrading
# ══════════════════════════════════════════════════════════

def test_a_frame_with_no_slicable_field_still_exports(frame):
    df = frame[["order_id", "revenue"]]
    page = build_dashboard_html(df, [], [], title="t")
    assert "<aside>" not in page
    assert page.rstrip().endswith("</html>")


def test_an_empty_tile_list_does_not_raise(frame):
    assert build_dashboard_html(frame, [], [], title="t")


def test_a_large_frame_caps_what_it_embeds():
    rng = np.random.default_rng(5)
    n = MAX_EMBEDDED_ROWS + 5_000
    df = pd.DataFrame({"grp": rng.choice(list("abcde"), n),
                       "revenue": rng.normal(100, 10, n)})
    page = build_dashboard_html(df, [], [], title="t")
    assert "{:,}".format(n) in page          # the true row count is stated
    assert "limited to the first" in page


# ══════════════════════════════════════════════════════════
#  The trend is legible
# ══════════════════════════════════════════════════════════

def test_a_dense_date_axis_is_rolled_up(frame):
    """600 daily points on a 400px tile is a noise floor with the finding
    only in the title."""
    import json

    import plotly.io as pio

    from app.engines.chart_engine import make_line

    fig = make_line(frame, "order_date", "revenue")
    points = json.loads(pio.to_json(fig))["data"][0]["x"]
    assert len(points) < 200, len(points)


def test_a_short_series_is_not_rolled_up():
    from app.engines.chart_engine import make_line
    import json
    import plotly.io as pio

    df = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=30),
                       "v": np.linspace(1, 30, 30)})
    fig = make_line(df, "d", "v")
    assert len(json.loads(pio.to_json(fig))["data"][0]["x"]) == 30
