import functools
import logging
import threading

import pandas as pd
import plotly.express as px

logger = logging.getLogger(__name__)

PALETTE  = ["#5b8def", "#2dd4a7", "#f0a44a", "#a78bfa", "#f4676b", "#38bdf8", "#fb7185", "#facc15"]
TEMPLATE = "plotly_dark"

# Plotly Express reads the shared template object in a way that is not
# thread-safe: the dashboard requests its tiles at once, they are built on
# parallel threads, and they failed with plotly's "ValueError: Invalid
# value". Figures are built one at a time; each takes milliseconds.
PLOTLY_LOCK = threading.RLock()


def _serial(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with PLOTLY_LOCK:
            return fn(*args, **kwargs)
    return wrapper


def _lab(col) -> str:
    """'tenure_years' → 'Tenure years' for axis titles."""
    from app.engines.chart_exporter import _label
    return _label(col)


def _style(fig, title: str = ""):
    """The app's look: transparent background so a chart sits in its tile,
    the app's typeface, quiet gridlines. A chart shown under a tile header
    gets no title of its own — it printed the same words twice."""
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=13)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, system-ui, sans-serif", color="#c9d1dc", size=12),
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        coloraxis_showscale=False,
        hoverlabel=dict(font_family="IBM Plex Mono, monospace"),
    )
    fig.update_xaxes(gridcolor="#22262e", zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor="#22262e", zeroline=False, automargin=True)
    return fig


@_serial
def make_bar(df, x, y, title=""):
    agg = (df.groupby(x)[y].sum()
             .reset_index()
             .sort_values(y, ascending=False)
             .head(25))
    fig = px.bar(agg, x=x, y=y, template=TEMPLATE,
                 labels={x: _lab(x), y: _lab(y)},
                 color_discrete_sequence=PALETTE[:1])
    fig.update_traces(hovertemplate="%{x}<br>%{y:,.4~g}<extra></extra>")
    return _style(fig, title)


@_serial
def make_line(df, x, y, title=""):
    fig = px.line(df.sort_values(x), x=x, y=y, template=TEMPLATE,
                  labels={x: _lab(x), y: _lab(y)},
                  color_discrete_sequence=PALETTE, markers=len(df) <= 40)
    return _style(fig, title)


@_serial
def make_scatter(df, x, y, color=None, title=""):
    fig = px.scatter(df.head(3000), x=x, y=y, color=color, template=TEMPLATE,
                     labels={x: _lab(x), y: _lab(y)},
                     color_discrete_sequence=PALETTE, opacity=0.7)
    return _style(fig, title)


@_serial
def make_histogram(df, col, nbins=40, title=""):
    fig = px.histogram(df, x=col, nbins=nbins, marginal="box", template=TEMPLATE,
                       labels={col: _lab(col)},
                       color_discrete_sequence=PALETTE[:1])
    fig.update_yaxes(title_text="Records", row=1, col=1)
    return _style(fig, title)


@_serial
def make_pie(df, names_col, values_col, title="", max_slices=10):
    """Shares of the whole: the largest slices, then one "Other".

    This took `.head(10)` of the groups in alphabetical order, so with more
    than ten categories the biggest could be missing — a category holding
    98.6% of the total was left out and ten others shown at 10% each.
    """
    totals = df.groupby(names_col)[values_col].sum().sort_values(ascending=False, kind="stable")
    if len(totals) > max_slices:
        top = totals.iloc[:max_slices - 1]
        other = totals.iloc[max_slices - 1:].sum()
        totals = pd.concat([top, pd.Series({"Other ({} more)".format(
            len(totals) - (max_slices - 1)): other})])
    agg = totals.rename_axis(names_col).reset_index(name=values_col)
    return _style(px.pie(
        agg, names=names_col, values=values_col,
        template=TEMPLATE, hole=0.45,
        color_discrete_sequence=PALETTE), title)


@_serial
def make_heatmap(df):
    """Spearman correlation between the measures it is given; callers pass
    measures only, so identifiers no longer appear as correlated."""
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if not pd.api.types.is_bool_dtype(df[c])]
    corr     = df[num_cols].corr(method="spearman").round(2)
    corr.index = [_lab(c) for c in corr.index]
    corr.columns = [_lab(c) for c in corr.columns]
    return _style(px.imshow(
        corr, text_auto=True,
        template=TEMPLATE,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1))
