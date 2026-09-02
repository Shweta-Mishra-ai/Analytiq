"""
engines/chart_engine.py — interactive (Plotly) charts for the API.

Chart *selection* was ported from dataforge-ai; the dark palette and
template are Analytiq's own and stay, because these figures render inside
Analytiq's dark UI rather than a Streamlit page.

What selection used to do, and no longer does:
  * chart `num_cols[0]` — whichever numeric column came first, including
    an order_id or a row index
  * sum every metric, so a "total satisfaction rating" was a valid chart
  * draw a pie of anything, including averages and rates, where the
    slices do not represent parts of a whole
  * label charts with emoji

Domain metric priorities live on DomainSpec.chart_metrics, not in a table
here — the dataforge original hardcoded its own per-domain dict, which is
the drift the registry exists to prevent.
"""
import logging
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PALETTE  = ["#4f8ef7","#22d3a5","#f7934f","#a78bfa","#f77070","#ffd43b","#38bdf8","#fb7185"]
TEMPLATE = "plotly_dark"


from app.services.dtypes import text_columns


# ══════════════════════════════════════════════════════════
#  COLUMN ROLE DETECTION
# ══════════════════════════════════════════════════════════

_IDENTIFIER_NAMES = {
    "index", "idx", "id", "row", "rowid", "row_id", "row_num",
    "rownum", "serial", "sr", "sr_no", "sno", "s_no",
    "order_id", "orderid", "customer_id", "customerid",
    "user_id", "userid", "emp_id", "empid", "employee_id",
    "product_id", "productid", "item_id", "itemid", "sku_id",
    "transaction_id", "txn_id", "record_id", "entry_id",
    "asin", "uuid", "guid", "account_id", "accountid", "patient_id",
    "campaign_id", "work_order_id",
}

_METRIC_NAMES = {
    "amount", "revenue", "sales", "price", "cost", "profit",
    "margin", "salary", "income", "spend", "budget", "expense",
    "qty", "quantity", "units", "volume", "count",
    "score", "rating", "satisfaction", "evaluation", "performance",
    "rate", "percentage", "pct", "percent",
    "hours", "days", "tenure", "age",
}

# Metrics where slices genuinely sum to a meaningful whole. A pie of an
# average or a rate is not a part-of-whole relationship, and reading one
# as a share is simply wrong.
_PIE_VALID_METRICS = {
    "revenue", "sales", "amount", "profit", "spend",
    "qty", "quantity", "units", "volume", "count", "headcount",
    "impressions", "clicks", "conversions", "throughput", "mrr", "arr",
    "income", "salary", "charges", "cost", "expense", "budget", "value",
}

# Scores are averaged, never summed — a total of a 1-5 rating is
# meaningless.
_SCORE_METRICS = {
    "satisfaction", "rating", "score", "evaluation",
    "performance", "nps", "csat", "index", "level",
}


def _is_identifier(col_name: str, series: pd.Series, n_rows: int) -> bool:
    """True when a column identifies a row rather than measuring it."""
    col_lower = str(col_name).lower().strip()

    if col_lower in _IDENTIFIER_NAMES:
        return True
    if re.search(r'\bid\b|\bindex\b|\bidx\b', col_lower):
        return True
    if not pd.api.types.is_numeric_dtype(series):
        return False
    # A recognised measure keeps its role even if it happens to be unique.
    if any(kw in col_lower for kw in _METRIC_NAMES):
        return False

    try:
        n_unique = series.nunique()
        if n_unique / max(n_rows, 1) > 0.95 and n_rows > 100:
            return True
        clean = series.dropna().sort_values().reset_index(drop=True)
        if len(clean) > 10 and (clean.diff().dropna() == 1).mean() > 0.95:
            return True
    except Exception:
        logger.debug("identifier check failed for %r", col_name, exc_info=True)
    return False


def _get_analysis_columns(df: pd.DataFrame) -> Dict:
    """Split columns into the roles a chart actually needs."""
    n_rows = len(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = text_columns(df)
    date_cols = df.select_dtypes(include="datetime").columns.tolist()

    id_cols, metrics, score_metrics = [], [], []
    for col in numeric_cols:
        # Generated missingness companions belong to modelling, not to a
        # client-facing chart.
        if str(col).endswith("__was_missing"):
            continue
        if _is_identifier(col, df[col], n_rows):
            id_cols.append(col)
            continue
        if any(kw in str(col).lower() for kw in _SCORE_METRICS):
            score_metrics.append(col)
        else:
            metrics.append(col)

    dimensions = [
        c for c in cat_cols
        if 2 <= df[c].nunique(dropna=True) <= 30
        and not _is_identifier(c, df[c], n_rows)
    ]
    return {
        "metrics": metrics, "score_metrics": score_metrics,
        "all_metrics": metrics + score_metrics, "dimensions": dimensions,
        "date_cols": date_cols, "id_cols": id_cols, "cat_cols": cat_cols,
    }


def _pick_primary_metric(cols: Dict, domain: str = "general",
                         df_ref: Optional[pd.DataFrame] = None
                         ) -> Optional[str]:
    """The metric worth leading with, preferring what the domain cares
    about, then the one that varies most."""
    all_metrics = cols["all_metrics"]
    if not all_metrics:
        return None

    try:
        from app.engines.domains.registry import spec_for
        priority = spec_for(domain).chart_metrics
    except Exception:
        logger.debug("registry chart_metrics unavailable", exc_info=True)
        priority = ()

    for preferred in priority:
        for col in all_metrics:
            if preferred in str(col).lower().replace("_", ""):
                return col
    # No domain preference matched. Take the measure that actually varies
    # rather than whichever came first in the frame — a 0/1 flag sitting in
    # column 3 is not the headline of a subscription dataset.
    candidates = cols["metrics"] or cols["score_metrics"]
    best, best_cv = None, -1.0
    for c in candidates:
        try:
            ser = pd.to_numeric(df_ref[c], errors="coerce").dropna() \
                if df_ref is not None else None
            if ser is None or len(ser) < 10 or ser.nunique() < 3:
                continue
            m = float(ser.mean())
            cv = abs(float(ser.std()) / m) if m else 0.0
            if cv > best_cv:
                best, best_cv = c, cv
        except Exception:
            logger.debug("cv check failed for %r", c, exc_info=True)
    return best or candidates[0]


def _pick_best_dimension(df: pd.DataFrame, cols: Dict,
                         metric_col: Optional[str]) -> Optional[str]:
    """The dimension on which the metric actually varies — a chart of a
    metric that is flat across every group shows nothing."""
    dims = cols["dimensions"]
    if not dims:
        return None
    if not metric_col:
        return dims[0]
    best, best_spread = None, -1.0
    for d in dims:
        try:
            agg = "mean" if _is_score_metric(metric_col) else "sum"
            g = df.groupby(d)[metric_col].agg(agg).dropna()
            if len(g) < 2:
                continue
            hi = float(g.max())
            spread = (hi - float(g.min())) / abs(hi) if hi else 0.0
            if spread > best_spread:
                best, best_spread = d, spread
        except Exception:
            logger.debug("dimension spread failed for %r", d, exc_info=True)
    return best or dims[0]


def _is_score_metric(col_name: str) -> bool:
    return any(kw in str(col_name).lower() for kw in _SCORE_METRICS)


def _is_pie_valid(metric_col: str) -> bool:
    """Whether a pie of this metric represents parts of a whole."""
    return any(kw in str(metric_col).lower().replace("_", "")
               for kw in _PIE_VALID_METRICS)


def safe_pct_gap(val_a: float, val_b: float) -> str:
    """Percentage gap that refuses to print nonsense.

    A zero baseline makes the ratio undefined and a near-zero one makes it
    absurd ("14,200% above target"); both used to reach the page.
    """
    try:
        a, b = float(val_a), float(val_b)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return "n/a"
    pct = (a - b) / abs(b) * 100
    if not np.isfinite(pct) or abs(pct) > 9999:
        return "n/a"
    return "{:+.1f}%".format(pct)



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


def recommend_charts(df: pd.DataFrame, domain: str = "general"
                     ) -> List[Tuple[str, go.Figure]]:
    """Up to five charts chosen from column roles rather than column order.

    Identifiers are never charted as metrics, scores are averaged and
    volumes summed, and a pie is only drawn when the slices genuinely sum
    to a whole.
    """
    cols = _get_analysis_columns(df)
    metric = _pick_primary_metric(cols, domain, df_ref=df)
    dim = _pick_best_dimension(df, cols, metric)
    charts: List[Tuple[str, go.Figure]] = []

    if metric is None:
        # Everything numeric is an identifier. Five meaningless charts is
        # worse than none.
        logger.info("no chartable metric among %d columns", len(df.columns))
        return charts

    # 1. Primary metric by its most-varying dimension
    if dim:
        is_score = _is_score_metric(metric)
        agg = "mean" if is_score else "sum"
        label = "Average" if is_score else "Total"
        try:
            grouped = (df.groupby(dim)[metric].agg(agg)
                         .sort_values(ascending=False).head(20).reset_index())
            fig = px.bar(grouped, x=dim, y=metric,
                         title=f"{label} {metric} by {dim}",
                         template=TEMPLATE, color_discrete_sequence=PALETTE)
            charts.append((f"{metric} by {dim}", _style(fig)))
        except Exception:
            logger.warning("bar chart failed", exc_info=True)

    # 2. Trend — only on a real datetime column. A "trend" over a numeric
    #    column binned as an axis is not a time series.
    if cols["date_cols"]:
        try:
            d = cols["date_cols"][0]
            data = df[[d, metric]].dropna().sort_values(d)
            fig = px.line(data, x=d, y=metric,
                          title=f"{metric} Over Time",
                          template=TEMPLATE, color_discrete_sequence=PALETTE)
            charts.append((f"{metric} Over Time", _style(fig)))
        except Exception:
            logger.warning("line chart failed", exc_info=True)

    # 3. Distribution
    try:
        fig = px.histogram(df, x=metric, nbins=40,
                           title=f"Distribution of {metric}",
                           template=TEMPLATE, color_discrete_sequence=PALETTE)
        charts.append((f"Distribution: {metric}", _style(fig)))
    except Exception:
        logger.warning("histogram failed", exc_info=True)

    # 4. Correlation across measures only — identifiers excluded, so the
    #    matrix no longer reports that order_id correlates with row index.
    measures = cols["all_metrics"]
    if len(measures) >= 3:
        try:
            corr = df[measures].corr().round(2)
            fig = px.imshow(corr, text_auto=True, title="Correlation Matrix",
                            template=TEMPLATE, color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1)
            charts.append(("Correlation Matrix", _style(fig)))
        except Exception:
            logger.warning("heatmap failed", exc_info=True)

    # 5. Composition — only where the metric is genuinely additive
    if dim and _is_pie_valid(metric) and 2 <= df[dim].nunique() <= 10:
        try:
            grouped = df.groupby(dim)[metric].sum().reset_index()
            fig = px.pie(grouped, names=dim, values=metric,
                         title=f"{metric} Share by {dim}",
                         template=TEMPLATE, color_discrete_sequence=PALETTE)
            charts.append((f"{metric} Share by {dim}", _style(fig)))
        except Exception:
            logger.warning("pie chart failed", exc_info=True)

    return charts[:5]


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
    from app.engines.domains.base import is_id_column
    # A row number correlates with nothing and takes a row and a column
    # in the matrix saying so.
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if not is_id_column(c, df[c])]
    if len(num_cols) < 2:
        num_cols = df.select_dtypes(include="number").columns.tolist()
    corr     = df[num_cols].corr().round(2)
    return _style(px.imshow(
        corr, text_auto=True,
        title="Correlation Matrix",
        template=TEMPLATE,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1))
