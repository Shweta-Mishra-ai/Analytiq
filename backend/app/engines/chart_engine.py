import logging
import pandas as pd
import plotly.express as px

logger = logging.getLogger(__name__)

PALETTE  = ["#4f8ef7","#22d3a5","#f7934f","#a78bfa","#f77070","#ffd43b","#38bdf8","#fb7185"]
TEMPLATE = "plotly_dark"




def _style(fig):
    fig.update_layout(
        paper_bgcolor="#07080f",
        plot_bgcolor="#0e0f1a",
        font=dict(family="JetBrains Mono, monospace", color="#dde1f5"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(gridcolor="#1e2035", zeroline=False)
    fig.update_yaxes(gridcolor="#1e2035", zeroline=False)
    return fig


def make_bar(df, x, y, title=""):
    agg = (df.groupby(x)[y].sum()
             .reset_index()
             .sort_values(y, ascending=False)
             .head(25))
    return _style(px.bar(agg, x=x, y=y,
        title=title or f"{y} by {x}",
        template=TEMPLATE, color=y,
        color_continuous_scale="Blues"))


def make_line(df, x, y, title=""):
    return _style(px.line(
        df.sort_values(x), x=x, y=y,
        title=title or f"{y} over {x}",
        template=TEMPLATE,
        color_discrete_sequence=PALETTE))


def make_scatter(df, x, y, color=None, title=""):
    return _style(px.scatter(
        df.head(3000), x=x, y=y, color=color,
        title=title or f"{x} vs {y}",
        template=TEMPLATE,
        color_discrete_sequence=PALETTE,
        opacity=0.7))


def make_histogram(df, col, nbins=40, title=""):
    return _style(px.histogram(
        df, x=col, nbins=nbins, marginal="box",
        title=title or f"Distribution: {col}",
        template=TEMPLATE,
        color_discrete_sequence=PALETTE))


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
        title=title or f"{values_col} by {names_col}",
        template=TEMPLATE,
        color_discrete_sequence=PALETTE))


def make_heatmap(df):
    """Spearman correlation between the measures it is given; callers pass
    measures only, so identifiers no longer appear as correlated."""
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if not pd.api.types.is_bool_dtype(df[c])]
    corr     = df[num_cols].corr(method="spearman").round(2)
    return _style(px.imshow(
        corr, text_auto=True,
        title="",
        template=TEMPLATE,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1))
