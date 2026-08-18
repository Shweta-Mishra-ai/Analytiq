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
    comparison_message,
    heatmap_message,
    histogram_message,
    human_number,
    line_message,
    pie_message,
)
from app.services.dtypes import text_columns, dedupe_columns

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  Theme — the same tokens the interface uses
# ══════════════════════════════════════════════════════════
# Mirrors frontend/src/index.css. Kept as literals rather than fetched
# from the client, because a figure is serialised server-side and the two
# have to agree; if the interface palette changes, change it here in the
# same commit.

FONT = "IBM Plex Sans, system-ui, -apple-system, Segoe UI, sans-serif"
MONO = "IBM Plex Mono, ui-monospace, SFMono-Regular, monospace"

# Two palettes, because the app and the deliverable are read in
# different places. The interface is dark; a document that goes to a
# client is read on a laptop in a meeting, printed, and pasted into a
# deck, and a dark chart survives none of those. The light one is the
# default for anything exported.
PALETTES = {
    "dark": {
        "ink": "#f2f4f7", "ink2": "#c9d1dc", "mute": "#8b93a1",
        "grid": "#22262e", "axis": "#2d323c", "hover": "#161a1f",
        "series": ["#5b8def", "#2dd4a7", "#f0a44a", "#a78bfa",
                   "#f4676b", "#7aa5f5", "#38bdf8", "#fbbf24"],
        "diverging": [[0.0, "#f4676b"], [0.5, "#161a1f"], [1.0, "#5b8def"]],
        "on_fill": "#0b0d10",
    },
    "light": {
        "ink": "#1a1d23", "ink2": "#3d434e", "mute": "#6b7280",
        "grid": "#e8eaee", "axis": "#d4d8e0", "hover": "#ffffff",
        # Amber leads, as in a printed finance pack: it reads as the
        # subject colour on white where a mid-blue reads as a hyperlink.
        # The rest are chosen to stay distinguishable in greyscale, since
        # these get printed.
        "series": ["#f0a11e", "#1f6feb", "#0f9d6e", "#7c4dff",
                   "#e5484d", "#f7c948", "#0891b2", "#c2410c"],
        "diverging": [[0.0, "#e5484d"], [0.5, "#f6f7f9"], [1.0, "#1f6feb"]],
        "on_fill": "#ffffff",
    },
}

# Each domain reads in its own colour, the way a practice has house
# colours for the kind of work it does: a workforce review and a P&L
# should not be indistinguishable at a glance across a desk. Only the
# accent and the second series change — the neutrals, the grid and the
# text stay fixed, because that is what keeps five palettes looking like
# one system rather than five templates.
#
# Each lead is checked against white for text contrast and against its
# own neighbour for separation; the later entries repeat across domains
# on purpose, since a chart with six series is rare and the first two
# are what a reader actually distinguishes.
DOMAIN_ACCENTS = {
    "finance":    ("#f0a11e", "#1f6feb"),   # amber, as in a printed pack
    "hr":         ("#1f6feb", "#0f9d6e"),   # blue — the workforce convention
    "sales":      ("#0f9d6e", "#f0a11e"),   # green, against target
    "ecommerce":  ("#e8590c", "#1f6feb"),   # orange, retail
    "marketing":  ("#7c4dff", "#0891b2"),
    "operations": ("#0891b2", "#f0a11e"),
    "saas":       ("#7c4dff", "#0f9d6e"),
    "healthcare": ("#0e9f9f", "#1f6feb"),
    "general":    ("#1f6feb", "#f0a11e"),
}

# Filled in below, one light palette per domain.
_TAIL = ["#7c4dff", "#e5484d", "#f7c948", "#0891b2", "#c2410c", "#0f9d6e"]
for _domain, (_lead, _second) in DOMAIN_ACCENTS.items():
    _base = dict(PALETTES["light"])
    _series = [_lead, _second] + [c for c in _TAIL if c not in (_lead, _second)]
    _base["series"] = _series[:8]
    _base["diverging"] = [[0.0, "#e5484d"], [0.5, "#f6f7f9"], [1.0, _lead]]
    PALETTES["light-" + _domain] = _base


def theme_for(domain: str, mode: str = "light") -> str:
    """The palette name for a domain, falling back to the plain one."""
    key = "{}-{}".format(mode, str(domain or "").strip().lower())
    return key if key in PALETTES else mode


# The interface is dark, so that stays the module default; the export
# passes a domain palette.
_ACTIVE = "dark"

INK = PALETTES["dark"]["ink"]
INK2 = PALETTES["dark"]["ink2"]
MUTE = PALETTES["dark"]["mute"]
GRID = PALETTES["dark"]["grid"]
AXIS = PALETTES["dark"]["axis"]
PALETTE = PALETTES["dark"]["series"]
DIVERGING = PALETTES["dark"]["diverging"]

TEMPLATE = "plotly_dark"


# ══════════════════════════════════════════════════════════
#  IBCS semantic notation
# ══════════════════════════════════════════════════════════
# The International Business Communication Standards give each kind of
# series a fixed appearance, so a reader who knows the notation can read
# any chart in the pack without consulting a legend:
#
#   actual        solid fill
#   plan/budget   outline only, no fill
#   forecast      hatched
#   prior period  light grey solid
#
# The point is not decoration. On a chart of revenue against budget the
# two bars were the accent and the second series colour, which says
# "two categories" — the reader has to look at the legend to learn which
# is the commitment and which is what happened. Notation says it without
# a legend, and says it the same way on every chart.
SERIES_KINDS = ("actual", "plan", "forecast", "previous")

# Whole words, for the same reason the rest of the codebase uses them:
# "ly" as a substring matches `monthly`, and "py" matches `supply`, so a
# monthly revenue column would have been drawn in prior-period grey.
_PLAN_WORDS = frozenset({"budget", "plan", "planned", "target", "quota",
                         "goal", "committed"})
_FORECAST_WORDS = frozenset({"forecast", "forecasted", "projected",
                             "projection", "expected", "estimate",
                             "estimated"})
_PREVIOUS_WORDS = frozenset({"prior", "previous", "prev", "ly", "py",
                             "lastyear", "yoy"})


def series_kind(name: str) -> str:
    """What a series is, from its column name.

    `revenue` is an actual, `budget` is a plan, `forecast_revenue` is a
    projection, `revenue_last_year` is a prior period. Anything
    unrecognised is treated as an actual — drawing an unknown series as
    a hollow "plan" bar would assert something about it that the column
    name does not support.
    """
    from app.engines.domain_detect import tokenise

    tokens = set(tokenise(name))
    # "last year" arrives as two tokens; join adjacent pairs so the
    # phrase is matched as well as the single words.
    words = list(tokens)
    if "last" in tokens and "year" in tokens:
        words.append("lastyear")
    tokens = set(words)

    if tokens & _FORECAST_WORDS:
        return "forecast"
    if tokens & _PLAN_WORDS:
        return "plan"
    if tokens & _PREVIOUS_WORDS:
        return "previous"
    return "actual"


def notation(kind: str, colour: str) -> dict:
    """Plotly marker settings for one IBCS series kind."""
    if kind == "plan":
        # Outline only: a commitment is not an outcome, and a hollow bar
        # reads as "the shape we said" against the solid one that
        # happened.
        return {"color": "rgba(0,0,0,0)",
                "line": {"color": colour, "width": 2}}
    if kind == "forecast":
        return {"color": colour, "opacity": 0.45,
                "pattern": {"shape": "/", "size": 6, "solidity": 0.35},
                "line": {"width": 0}}
    if kind == "previous":
        return {"color": "#b6bcc6", "line": {"width": 0}}
    return {"color": colour, "line": {"width": 0}}


def palette(theme: str = None) -> dict:
    return PALETTES.get(theme or _ACTIVE, PALETTES["dark"])


class use_theme:
    """Render inside a palette. `with use_theme("light"): ...`

    A context manager rather than a parameter on every builder: the
    figure builders are called from six places and threading a theme
    argument through all of them is how one of them ends up forgotten
    and renders a dark chart into a light document.
    """

    def __init__(self, theme: str):
        self.theme = theme if theme in PALETTES else "dark"

    def __enter__(self):
        global _ACTIVE
        self._previous = _ACTIVE
        _ACTIVE = self.theme
        return self

    def __exit__(self, *_exc):
        global _ACTIVE
        _ACTIVE = self._previous
        return False


def _style(fig: go.Figure, message: str = "", subtitle: str = "") -> go.Figure:
    """App chrome, and the finding as the title.

    Transparent paper is the important part: the tile already has a
    background, a border and a radius. A figure that paints its own
    rectangle inside that shows a seam on every tile.
    """
    p = palette()
    title = None
    if message:
        title = dict(
            text=("<b>{}</b>".format(message)
                  + ("<br><span style='font-size:11px;color:{}'>{}</span>"
                     .format(p["mute"], subtitle) if subtitle else "")),
            x=0, xanchor="left", y=0.97, yanchor="top",
            font=dict(size=13, color=p["ink"]),
        )
    elif subtitle:
        title = dict(text=subtitle, x=0, xanchor="left",
                     font=dict(size=13, color=p["ink"]))

    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=p["ink2"], size=11),
        margin=dict(l=8, r=12, t=54 if title else 12, b=8),
        hoverlabel=dict(font=dict(family=MONO, size=11),
                        bgcolor=p["hover"], bordercolor=p["axis"]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22,
                    xanchor="left", x=0,
                    font=dict(size=10, color=p["mute"]),
                    bgcolor="rgba(0,0,0,0)"),
        colorway=p["series"],
        separators=".,",
    )
    # Horizontal rules only. Vertical gridlines between categories mark
    # nothing — a bar sitting on "region" has no x scale to read against.
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=p["axis"],
                     tickfont=dict(family=MONO, size=10, color=p["mute"]),
                     title=dict(font=dict(size=10, color=p["mute"])))
    fig.update_yaxes(gridcolor=p["grid"], zeroline=False, showline=False,
                     tickfont=dict(family=MONO, size=10, color=p["mute"]),
                     title=dict(font=dict(size=10, color=p["mute"])))
    return fig


def _zero_baseline(fig: go.Figure, values) -> None:
    """A bar chart is read by comparing lengths, so it starts at zero.

    Four bars between 980 and 1,020 on an axis that starts at 975: the
    shortest looks a fifth of the tallest and the difference is 4%. The
    axis labels that would correct it are the part nobody checks.

    Both plotting libraries happen to zero-base bars today — matplotlib
    because a bar patch extends to zero and enters the data limits that
    way, plotly.js because bar traces put their base into the axis
    extremes. Neither is a promise, and the headroom this function also
    sets is an explicit range, which is exactly the kind of change that
    silently takes the default away. The rule is stated here, and
    `test_chart_standards.py` measures the drawn bars in pixels rather
    than trusting either.

    Bars that go negative keep an autoscaled floor — clamping those to
    zero would hide them.
    """
    try:
        low = float(min(values))
        high = float(max(values))
    except (TypeError, ValueError):
        return
    if low < 0:
        return
    fig.update_yaxes(range=[0, high * 1.12 if high > 0 else 1],
                     rangemode="tozero")


def truncated_scale(values) -> Optional[float]:
    """The floor a line chart will be drawn from, when it is not zero.

    A line is read by its slope, and a slope is only meaningful against
    the scale it is drawn on. Revenue moving from 980 to 1,020 on an
    axis that starts at 975 climbs most of the height of the tile; the
    same series against zero is a flat line. Both are honest pictures of
    a 4% rise and a reader takes opposite conclusions from them.

    Bars must start at zero. A trend line may be truncated — sometimes
    it has to be, or a real movement disappears into the thickness of
    the line — but then the truncation is disclosed, which is what IBCS
    calls a scaling indicator. Returns the floor to disclose, or None
    when the axis will reach zero on its own and there is nothing to
    say.
    """
    try:
        low = float(min(values))
        high = float(max(values))
    except (TypeError, ValueError):
        return None
    if low <= 0 or high <= 0:
        return None
    # Plotly pads a line axis by a fraction of the data range, so the
    # axis reaches zero by itself once the movement is large relative to
    # the level. Below that it does not, and the slope is magnified.
    if (high - low) >= low:
        return None
    return low


def _scale_note(fig: go.Figure, values) -> None:
    floor = truncated_scale(values)
    if floor is None:
        return
    fig.add_annotation(
        text="scale starts at {}, not zero".format(human_number(floor)),
        xref="paper", yref="paper", x=0, y=-0.19, showarrow=False,
        font=dict(size=9, color=palette()["mute"], family=MONO),
        align="left", xanchor="left",
    )


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
    # Duplicate column names make `df[name]` return a DataFrame
    # instead of a Series, and every `.dtype` / `.nunique()` /
    # `to_numeric` call below then raises. Guarded here as well as
    # at the loader and the store, because this is a public entry
    # point and a caller can hand it any frame.
    df = dedupe_columns(df)
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
    # A headcount-style call ("how many rows per group") passes the same
    # column as both x and y. Summing a categorical column against
    # itself and then re-inserting it as an index column raises "cannot
    # insert <col>, already exists" — a count needs its own column name.
    # The API layer pre-aggregates around this before calling here, but
    # this is a public entry point too and should not depend on every
    # caller remembering to.
    is_count = x == y
    if is_count:
        agg = (df.groupby(x, dropna=True).size()
                 .reset_index(name="_n")
                 .sort_values("_n", ascending=False)
                 .head(top_n))
        value_col = "_n"
    else:
        agg = (df.groupby(x, dropna=True)[y].sum()
                 .reset_index()
                 .sort_values(y, ascending=False)
                 .head(top_n))
        value_col = y
    fig = px.bar(agg, x=x, y=value_col)
    # One colour. Colouring bars by their own height restates the y-axis
    # and adds a legend for a variable already on the chart.
    fig.update_traces(
        marker_color=palette()["series"][0],
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>" + ("Count" if is_count else str(y))
                       + ": %{y:,.0f}<extra></extra>",
    )
    _style(fig, bar_message(df, x, y, counts=is_count),
           title or "{} by {}".format(y, x))
    _human_ticks(fig)
    _zero_baseline(fig, agg[value_col])
    return fig


def make_line(df, x, y, title="", max_points: int = 120):
    """A trend, at a grain a reader can actually see.

    Plotting every row put 600 daily points on a tile 400 pixels wide:
    a solid band of zig-zag with a faint upward drift somewhere inside
    it. The finding — revenue rose 66% — was in the title and nowhere in
    the picture. Dense date axes are rolled up to weeks or months, which
    is what makes a trend line a trend line rather than a noise floor.
    """
    work = df[[x, y]].dropna().sort_values(x)
    if (pd.api.types.is_datetime64_any_dtype(work[x])
            and len(work) > max_points):
        span_days = (work[x].max() - work[x].min()).days or 1
        rule = ("D" if span_days <= max_points
                else "W" if span_days <= max_points * 7
                else "ME" if span_days <= max_points * 31 else "QE")
        work = (work.set_index(x)[y].resample(rule).sum()
                    .reset_index().dropna())
        if len(work) > max_points:
            work = work.tail(max_points)
    fig = px.line(work, x=x, y=y)
    fig.update_traces(
        line=dict(color=palette()["series"][0], width=2),
        hovertemplate="%{x}<br>" + str(y) + ": %{y:,.0f}<extra></extra>",
    )
    _style(fig, line_message(df, x, y), title or "{} over {}".format(y, x))
    _human_ticks(fig)
    _scale_note(fig, work[y])
    return fig


def make_scatter(df, x, y, color=None, title=""):
    fig = px.scatter(df.head(3000), x=x, y=y, color=color,
                     color_discrete_sequence=palette()["series"], opacity=0.65)
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    _style(fig, "", title or "{} against {}".format(y, x))
    _human_ticks(fig)
    _human_ticks(fig, "x")
    return fig


def make_histogram(df, col, nbins=40, title=""):
    fig = px.histogram(df, x=col, nbins=nbins)
    fig.update_traces(marker_color=palette()["series"][0], marker_line_width=0)
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s):
        # The median, not the mean: on a skewed column the mean sits
        # where few records are, and the reader takes the marked line as
        # "typical".
        median = float(s.median())
        fig.add_vline(
            x=median, line_width=1.5, line_dash="dash",
            line_color=palette()["series"][2],
            annotation_text="median {}".format(human_number(median)),
            annotation_position="top",
            annotation_font=dict(size=10, color=palette()["mute"], family=MONO),
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
        marker=dict(colors=palette()["series"],
                    line=dict(color=palette()["on_fill"], width=2)),
        textposition="inside", textinfo="percent",
        insidetextfont=dict(color=palette()["on_fill"], size=11),
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
                    color_continuous_scale=palette()["diverging"], zmin=-1, zmax=1)
    fig.update_traces(
        xgap=2, ygap=2,
        hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
    )
    fig.update_layout(coloraxis_colorbar=dict(
        thickness=8, len=0.6, outlinewidth=0,
        tickfont=dict(size=9, color=palette()["mute"], family=MONO)))
    _style(fig, heatmap_message(df, cols), "Correlation matrix")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def make_comparison(df, x, y, y2, title="", top_n: int = 25):
    """Two measures side by side — actual against plan, this year against
    last.

    The finance dashboard had a "revenue against budget" tile that
    plotted revenue alone: the builder took a single `y`, so the tile
    promised a comparison and delivered one series. A chart whose title
    names two things and shows one is worse than no chart, because the
    reader takes the shape of the bars as the answer to the question in
    the title.
    """
    p = palette()
    agg = (df.groupby(x, dropna=True)[[y, y2]].sum()
             .reset_index()
             .sort_values(y, ascending=False)
             .head(top_n))
    if pd.api.types.is_datetime64_any_dtype(agg[x]):
        agg = agg.sort_values(x)

    fig = go.Figure()
    # IBCS notation: the actual is solid, the plan is an outline, a
    # forecast is hatched. Two solid bars in different colours say "two
    # categories" and send the reader to the legend to find out which is
    # the commitment.
    for name in (y, y2):
        kind = series_kind(name)
        fig.add_bar(x=agg[x], y=agg[name],
                    name="{} ({})".format(name, kind) if kind != "actual"
                         else str(name),
                    marker=notation(kind, p["series"][0]),
                    hovertemplate="<b>%{x}</b><br>" + str(name)
                                  + ": %{y:,.0f}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)

    # Where it landed matters less than where it did not: naming the
    # worst group is the part a reader can act on.
    gaps = agg[y] - agg[y2]
    worst = ""
    if len(gaps) > 1 and gaps.min() < 0:
        worst = str(agg.loc[gaps.idxmin(), x])
    message = comparison_message(str(y), str(y2), float(agg[y].sum()),
                                 float(agg[y2].sum()), worst)
    _style(fig, message, title or "{} against {}".format(y, y2))
    _human_ticks(fig)
    _zero_baseline(fig, list(agg[y]) + list(agg[y2]))
    # Two series need a legend; the shared style hides it below the plot.
    fig.update_layout(showlegend=True)
    return fig
