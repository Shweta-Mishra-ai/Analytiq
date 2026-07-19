import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple, Optional

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


def recommend_charts(df: pd.DataFrame) -> List[Tuple[str, go.Figure]]:
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include="object").columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()
    charts    = []

    if date_cols and num_cols:
        fig = px.line(
            df.sort_values(date_cols[0]),
            x=date_cols[0], y=num_cols[0],
            title=f"📈 {num_cols[0]} Over Time",
            template=TEMPLATE, color_discrete_sequence=PALETTE
        )
        charts.append(("Time Series", _style(fig)))

    if len(num_cols) >= 3:
        corr = df[num_cols].corr().round(2)
        fig  = px.imshow(
            corr, text_auto=True,
            title="🔗 Correlation Matrix",
            template=TEMPLATE,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1
        )
        charts.append(("Correlations", _style(fig)))

    if cat_cols and num_cols:
        best = next((c for c in cat_cols if 2 <= df[c].nunique() <= 30), cat_cols[0])
        agg  = (df.groupby(best)[num_cols[0]].sum()
                  .reset_index()
                  .sort_values(num_cols[0], ascending=False)
                  .head(20))
        fig  = px.bar(
            agg, x=best, y=num_cols[0],
            title=f"📊 {num_cols[0]} by {best}",
            template=TEMPLATE, color=num_cols[0],
            color_continuous_scale="Blues"
        )
        charts.append(("Top Categories", _style(fig)))

    if num_cols:
        fig = px.histogram(
            df, x=num_cols[0], nbins=40, marginal="box",
            title=f"📉 Distribution: {num_cols[0]}",
            template=TEMPLATE, color_discrete_sequence=PALETTE
        )
        charts.append(("Distribution", _style(fig)))

    if len(num_cols) >= 2:
        fig = px.scatter(
            df.head(2000),
            x=num_cols[0], y=num_cols[1],
            color=cat_cols[0] if cat_cols else None,
            title=f"🔵 {num_cols[0]} vs {num_cols[1]}",
            template=TEMPLATE,
            color_discrete_sequence=PALETTE,
            opacity=0.7
        )
        charts.append(("Scatter", _style(fig)))

    return charts


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


def make_pie(df, names_col, values_col, title=""):
    agg = df.groupby(names_col)[values_col].sum().reset_index().head(10)
    return _style(px.pie(
        agg, names=names_col, values=values_col,
        title=title or f"{values_col} by {names_col}",
        template=TEMPLATE,
        color_discrete_sequence=PALETTE))


def make_heatmap(df):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    corr     = df[num_cols].corr().round(2)
    return _style(px.imshow(
        corr, text_auto=True,
        title="Correlation Matrix",
        template=TEMPLATE,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1))
