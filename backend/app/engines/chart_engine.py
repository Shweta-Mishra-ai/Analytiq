"""
engines/chart_engine.py — the interactive charts: dashboard tiles, EDA,
anything rendered by Plotly in the browser.

Three things made these read as a hobby project rather than a BI tool.

**They painted their own background.** Every figure hardcoded
`paper_bgcolor="#07080f"` and `plot_bgcolor="#0e0f1a"` — a near-black
that is not the app's panel colour (`#101215`), in `JetBrains Mono`,
which the app does not use anywhere. So each tile showed a dark
rectangle sitting on a slightly different dark panel, with a
terminal-style font inside it. The figures are now transparent and take
the app's own tokens, so a chart looks like part of the page instead of
an image pasted onto it.

**They charted whatever column came first.** `num_cols[0]` on a sales
export is `units` or, worse, `order_id` — the same defect that put an
identifier on three of five pages of the PDF. Ranking is shared with the
export path so the dashboard and the document agree about what the
business measure is.

**They were titled with their axes**, and decorated: "📊 revenue by
region", with the bars coloured by their own height on a continuous blue
scale, which adds a legend that repeats the y-axis and makes every bar
chart look like a heatmap. Titles now carry the finding, and colour is
used to distinguish series — never to restate a value the axis already
gives.
"""
import logging
from typing import List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.engines.chart_message import (
    bar_message,
    heatmap_message,
    histogram_message,
    human_number,
    line_message,
    pie_message,
)
from app.services.dtypes import text_columns

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  Theme — the same tokens the interface uses
# ══════════════════════════════════════════════════════════
# Mirrors frontend/src/index.css. Kept as literals rather than fetched
# from the client, because a figure is serialised server-side and the two
# have to agree; if the interface palette changes, change it here in the
# same commit.

INK = "#f2f4f7"        # --color-ink
INK2 = "#c9d1dc"       # --color-ink2
MUTE = "#8b93a1"       # --color-mute
GRID = "#22262e"       # --color-edge
AXIS = "#2d323c"       # --color-edge2

# Categorical series. Ordered so the first two — the ones most charts
# actually use — are the app's accent and teal.
PALETTE = ["#5b8def", "#2dd4a7", "#f0a44a", "#a78bfa",
           "#f4676b", "#7aa5f5", "#38bdf8", "#fbbf24"]

# A single-hue ramp for the one case that needs a continuous scale: a
# correlation matrix, where the value genuinely maps to colour.
DIVERGING = [[0.0, "#f4676b"], [0.5, "#161a1f"], [1.0, "#5b8def"]]

FONT = "IBM Plex Sans, system-ui, -apple-system, Segoe UI, sans-serif"
MONO = "IBM Plex Mono, ui-monospace, SFMono-Regular, monospace"

TEMPLATE = "plotly_dark"


def _style(fig: go.Figure, message: str = "", subtitle: str = "") -> go.Figure:
    """App chrome, and the finding as the title.

    Transparent paper is the important part: the tile already has a
    background, a border and a radius. A figure that paints its own
    rectangle inside that shows a seam on every tile.
    """
    title = None
    if message:
        title = dict(
            text=("<b>{}</b>".format(message)
                  + ("<br><span style='font-size:11px;color:{}'>{}</span>"
                     .format(MUTE, subtitle) if subtitle else "")),
            x=0, xanchor="left", y=0.97, yanchor="top",
            font=dict(size=13, color=INK),
        )
    elif subtitle:
        title = dict(text=subtitle, x=0, xanchor="left",
                     font=dict(size=13, color=INK))

    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK2, size=11),
        margin=dict(l=8, r=12, t=54 if title else 12, b=8),
        hoverlabel=dict(font=dict(family=MONO, size=11),
                        bgcolor="#161a1f", bordercolor=AXIS),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22,
                    xanchor="left", x=0,
                    font=dict(size=10, color=MUTE),
                    bgcolor="rgba(0,0,0,0)"),
        colorway=PALETTE,
        separators=".,",
    )
    # Horizontal rules only. Vertical gridlines between categories mark
    # nothing — a bar sitting on "region" has no x scale to read against.
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=AXIS,
                     tickfont=dict(family=MONO, size=10, color=MUTE),
                     title=dict(font=dict(size=10, color=MUTE)))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False,
                     tickfont=dict(family=MONO, size=10, color=MUTE),
                     title=dict(font=dict(size=10, color=MUTE)))
    return fig


def _human_ticks(fig: go.Figure, axis: str = "y") -> None:
    """Axis labels as "3.2m", not "3242612" or "3.242612e+6"."""
    updater = fig.update_yaxes if axis == "y" else fig.update_xaxes
    updater(tickformat="~s", exponentformat="none")


def rank_measures(df: pd.DataFrame, cols: Optional[List[str]] = None
                  ) -> List[str]:
    """Shared with the PDF export so both agree what the measure is."""
    from app.engines.chart_exporter import _rank_measures

    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
    return _rank_measures(df, cols)


def _cat_columns(df: pd.DataFrame, limit: int = 30) -> List[str]:
    from app.engines.domains.base import is_id_column

    return [c for c in text_columns(df)
            if not is_id_column(c, df[c]) and 2 <= df[c].nunique() <= limit]


# ══════════════════════════════════════════════════════════
#  The recommended set
# ══════════════════════════════════════════════════════════

def recommend_charts(df: pd.DataFrame) -> List[Tuple[str, go.Figure]]:
    """A starting pack that covers more than one measure.

    Every chart used to take `num_cols[0]`, so a sales file produced five
    views of whichever column happened to be first — on a real export,
    the order ID.
    """
    num_cols = rank_measures(df)
    cat_cols = _cat_columns(df)
    date_cols = df.select_dtypes(include="datetime").columns.tolist()
    charts: List[Tuple[str, go.Figure]] = []

    if not num_cols:
        logger.info("no non-identifier numeric columns to chart")
        return []

    primary = num_cols[0]
    secondary = num_cols[1] if len(num_cols) > 1 else primary

    if date_cols:
        try:
            charts.append(("Trend", make_line(df, date_cols[0], primary)))
        except Exception:
            logger.debug("recommend: line failed", exc_info=True)

    if cat_cols:
        try:
            charts.append(("Top Categories",
                           make_bar(df, cat_cols[0], primary)))
        except Exception:
            logger.debug("recommend: bar failed", exc_info=True)

    try:
        charts.append(("Distribution", make_histogram(df, secondary)))
    except Exception:
        logger.debug("recommend: histogram failed", exc_info=True)

    if len(cat_cols) > 1:
        try:
            charts.append(("Share", make_pie(df, cat_cols[1], primary)))
        except Exception:
            logger.debug("recommend: pie failed", exc_info=True)

    if len(num_cols) >= 3:
        try:
            charts.append(("Correlations", make_heatmap(df)))
        except Exception:
            logger.debug("recommend: heatmap failed", exc_info=True)

    return charts


# ══════════════════════════════════════════════════════════
#  Individual builders
# ══════════════════════════════════════════════════════════

def make_bar(df, x, y, title="", top_n: int = 25):
    agg = (df.groupby(x, dropna=True)[y].sum()
             .reset_index()
             .sort_values(y, ascending=False)
             .head(top_n))
    fig = px.bar(agg, x=x, y=y)
    # One colour. Colouring bars by their own height restates the y-axis
    # and adds a legend for a variable already on the chart.
    fig.update_traces(
        marker_color=PALETTE[0],
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>" + str(y) + ": %{y:,.0f}<extra></extra>",
    )
    _style(fig, bar_message(df, x, y), title or "{} by {}".format(y, x))
    _human_ticks(fig)
    return fig


def make_line(df, x, y, title=""):
    work = df[[x, y]].dropna().sort_values(x)
    fig = px.line(work, x=x, y=y)
    fig.update_traces(
        line=dict(color=PALETTE[0], width=2),
        hovertemplate="%{x}<br>" + str(y) + ": %{y:,.0f}<extra></extra>",
    )
    _style(fig, line_message(df, x, y), title or "{} over {}".format(y, x))
    _human_ticks(fig)
    return fig


def make_scatter(df, x, y, color=None, title=""):
    fig = px.scatter(df.head(3000), x=x, y=y, color=color,
                     color_discrete_sequence=PALETTE, opacity=0.65)
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    _style(fig, "", title or "{} against {}".format(y, x))
    _human_ticks(fig)
    _human_ticks(fig, "x")
    return fig


def make_histogram(df, col, nbins=40, title=""):
    fig = px.histogram(df, x=col, nbins=nbins)
    fig.update_traces(marker_color=PALETTE[0], marker_line_width=0)
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s):
        # The median, not the mean: on a skewed column the mean sits
        # where few records are, and the reader takes the marked line as
        # "typical".
        median = float(s.median())
        fig.add_vline(
            x=median, line_width=1.5, line_dash="dash",
            line_color=PALETTE[2],
            annotation_text="median {}".format(human_number(median)),
            annotation_position="top",
            annotation_font=dict(size=10, color=MUTE, family=MONO),
        )
    _style(fig, histogram_message(df, col),
           title or "Distribution of {}".format(col))
    _human_ticks(fig, "x")
    return fig


def make_pie(df, names_col, values_col, title=""):
    agg = (df.groupby(names_col, dropna=True)[values_col].sum()
             .reset_index()
             .sort_values(values_col, ascending=False)
             .head(10))
    fig = px.pie(agg, names=names_col, values=values_col, hole=0.55)
    fig.update_traces(
        marker=dict(colors=PALETTE, line=dict(color="#101215", width=2)),
        textposition="inside", textinfo="percent",
        insidetextfont=dict(color="#0b0d10", size=11),
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} (%{percent})"
                      "<extra></extra>",
    )
    _style(fig, pie_message(df, names_col, values_col),
           title or "{} by {}".format(values_col, names_col))
    return fig


def make_heatmap(df):
    import numpy as np

    cols = sorted(rank_measures(df)[:10])
    if len(cols) < 2:
        raise ValueError("need two non-identifier numeric columns")
    corr = df[cols].corr().round(2)
    matrix = corr.to_numpy(dtype=float, copy=True)
    # 1.00 down the diagonal by definition — left in, it takes the
    # strongest colour on the scale and the eye lands on the one thing
    # carrying no information.
    np.fill_diagonal(matrix, np.nan)
    fig = px.imshow(matrix, x=cols, y=cols, text_auto=".2f",
                    color_continuous_scale=DIVERGING, zmin=-1, zmax=1)
    fig.update_traces(
        xgap=2, ygap=2,
        hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
    )
    fig.update_layout(coloraxis_colorbar=dict(
        thickness=8, len=0.6, outlinewidth=0,
        tickfont=dict(size=9, color=MUTE, family=MONO)))
    _style(fig, heatmap_message(df, cols), "Correlation matrix")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig
