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
import threading

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


# How much wider than the row count an integer key's range may run and
# still read as a key. Ids with gaps (deleted rows, sharded ranges) stay
# inside a small multiple; a measured quantity does not.
ID_SPAN_TOLERANCE = 3.0


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
        clean = series.dropna()
        if clean.empty:
            return False
        # Near-consecutive integers: a row counter however it is named.
        ordered = clean.sort_values().reset_index(drop=True)
        if len(ordered) > 10 and (ordered.diff().dropna() == 1).mean() > 0.95:
            return True

        # Uniqueness alone used to be the test, and it is wrong: a
        # continuous measure is nearly all-distinct by nature. It
        # classified `deal_size` — the single most important column in a
        # sales file — as an identifier, which removed it from every
        # chart and every correlation on the dashboard.
        if clean.nunique() / max(n_rows, 1) <= 0.95 or n_rows <= 100:
            return False
        # Decimals mean it was measured, not assigned.
        if not bool((clean % 1 == 0).all()):
            return False
        # An assigned integer key spans about as many values as there are
        # rows. A measured quantity spans far more (or far less) — deal
        # sizes from 1,065 to 113,382 across 3,000 rows span thirty-odd
        # times the row count.
        span = float(ordered.iloc[-1]) - float(ordered.iloc[0])
        return abs(span) <= n_rows * ID_SPAN_TOLERANCE
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
            if preferred not in str(col).lower().replace("_", ""):
                continue
            # The domain's preferred metric still has to be chartable.
            # Without this the priority list returned early and won: a
            # logistics dashboard led with `on_time` (a 0/1 flag, so
            # "Distribution of On Time" is two bars) and a sales one with
            # `quota`, which was the same 250,000 on every row — a flat
            # bar chart and a single-spike histogram.
            if _is_chartable(col, df_ref):
                return col
    # No domain preference matched. Take the measure that actually varies
    # rather than whichever came first in the frame — a 0/1 flag sitting in
    # column 3 is not the headline of a subscription dataset.
    candidates = cols["metrics"] or cols["score_metrics"]
    best, best_cv = None, -1.0
    for c in candidates:
        if not _is_chartable(c, df_ref):
            continue
        try:
            ser = pd.to_numeric(df_ref[c], errors="coerce").dropna()
            m = float(ser.mean())
            cv = abs(float(ser.std()) / m) if m else 0.0
            if cv > best_cv:
                best, best_cv = c, cv
        except Exception:
            logger.debug("cv check failed for %r", c, exc_info=True)
    if best is not None:
        return best
    # Nothing varies enough to lead with. Falling back to the first
    # column would put a constant on the front page, so say there is no
    # metric and let the caller render nothing instead.
    return next((c for c in candidates if _is_chartable(c, df_ref)), None)


# A measure needs at least this many distinct values to be worth a bar
# chart or a histogram. Two is a flag; one is a constant.
MIN_CHARTABLE_VALUES = 3


def _is_chartable(col, df_ref: Optional[pd.DataFrame]) -> bool:
    """True when this column carries enough variation to draw."""
    if df_ref is None or col not in getattr(df_ref, "columns", []):
        return False
    try:
        ser = pd.to_numeric(df_ref[col], errors="coerce").dropna()
        if len(ser) < 10:
            return False
        return int(ser.nunique()) >= MIN_CHARTABLE_VALUES
    except Exception:
        logger.debug("chartable check failed for %r", col, exc_info=True)
        return False


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


# `template="plotly_dark"` is a *name*: plotly.express resolves it to the
# one Template object registered on the module, then walks that object's
# prototype traces to decide the default colours. Those traces are shared
# mutable state, and detaching a child's props on one thread while
# another reads them raises "Invalid value" from deep inside plotly.
#
# It surfaced as a tile that failed to draw on the dashboard, roughly one
# request in seventeen, because the dashboard builds five charts at once
# and React's development double-mount makes that ten. A user saw one
# broken tile among four good ones and no way to tell why.
#
# Building the figure is milliseconds — the aggregation before it is the
# expensive part — so serialising construction costs nothing that anyone
# can perceive, and it is the only fix that does not depend on plotly's
# internals staying as they are.
_FIGURE_LOCK = threading.Lock()


def _build(make, *args, **kwargs):
    """Construct a plotly figure and style it, one thread at a time."""
    with _FIGURE_LOCK:
        return _style(make(*args, **kwargs))


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
            fig = _build(px.bar, grouped, x=dim, y=metric,
                         title=f"{label} {metric} by {dim}",
                         template=TEMPLATE, color_discrete_sequence=PALETTE)
            charts.append((f"{metric} by {dim}", fig))
        except Exception:
            logger.warning("bar chart failed", exc_info=True)

    # 2. Trend — only on a real datetime column. A "trend" over a numeric
    #    column binned as an axis is not a time series.
    if cols["date_cols"]:
        try:
            d = cols["date_cols"][0]
            data = df[[d, metric]].dropna().sort_values(d)
            fig = _build(px.line, data, x=d, y=metric,
                          title=f"{metric} Over Time",
                          template=TEMPLATE, color_discrete_sequence=PALETTE)
            charts.append((f"{metric} Over Time", fig))
        except Exception:
            logger.warning("line chart failed", exc_info=True)

    # 3. Distribution
    try:
        fig = _build(px.histogram, df, x=metric, nbins=40,
                           title=f"Distribution of {metric}",
                           template=TEMPLATE, color_discrete_sequence=PALETTE)
        charts.append((f"Distribution: {metric}", fig))
    except Exception:
        logger.warning("histogram failed", exc_info=True)

    # 4. Correlation across measures only — identifiers excluded, so the
    #    matrix no longer reports that order_id correlates with row index.
    measures = cols["all_metrics"]
    if len(measures) >= 3:
        try:
            corr = df[measures].corr().round(2)
            fig = _build(px.imshow, corr, text_auto=True, title="Correlation Matrix",
                            template=TEMPLATE, color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1)
            charts.append(("Correlation Matrix", fig))
        except Exception:
            logger.warning("heatmap failed", exc_info=True)

    # 5. Composition — only where the metric is genuinely additive
    if dim and _is_pie_valid(metric) and 2 <= df[dim].nunique() <= 10:
        try:
            grouped = df.groupby(dim)[metric].sum().reset_index()
            fig = _build(px.pie, grouped, names=dim, values=metric,
                         title=f"{metric} Share by {dim}",
                         template=TEMPLATE, color_discrete_sequence=PALETTE)
            charts.append((f"{metric} Share by {dim}", fig))
        except Exception:
            logger.warning("pie chart failed", exc_info=True)

    return charts[:5]


def make_bar(df, x, y, title=""):
    agg = (df.groupby(x)[y].sum()
             .reset_index()
             .sort_values(y, ascending=False)
             .head(25))
    return _build(px.bar, agg, x=x, y=y,
        title=title or f"{y} by {x}",
        template=TEMPLATE, color=y,
        color_continuous_scale="Blues")


def make_line(df, x, y, title=""):
    return _build(px.line,
        df.sort_values(x), x=x, y=y,
        title=title or f"{y} over {x}",
        template=TEMPLATE,
        color_discrete_sequence=PALETTE)


def make_scatter(df, x, y, color=None, title=""):
    return _build(px.scatter,
        df.head(3000), x=x, y=y, color=color,
        title=title or f"{x} vs {y}",
        template=TEMPLATE,
        color_discrete_sequence=PALETTE,
        opacity=0.7)


def make_histogram(df, col, nbins=40, title=""):
    return _build(px.histogram,
        df, x=col, nbins=nbins, marginal="box",
        title=title or f"Distribution: {col}",
        template=TEMPLATE,
        color_discrete_sequence=PALETTE)


def make_pie(df, names_col, values_col, title=""):
    agg = df.groupby(names_col)[values_col].sum().reset_index().head(10)
    return _build(px.pie,
        agg, names=names_col, values=values_col,
        title=title or f"{values_col} by {names_col}",
        template=TEMPLATE,
        color_discrete_sequence=PALETTE)


def make_heatmap(df):
    from app.engines.domains.base import is_id_column
    # A row number correlates with nothing and takes a row and a column
    # in the matrix saying so.
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if not is_id_column(c, df[c])]
    if len(num_cols) < 2:
        num_cols = df.select_dtypes(include="number").columns.tolist()
    corr     = df[num_cols].corr().round(2)
    return _build(px.imshow, 
        corr, text_auto=True,
        title="Correlation Matrix",
        template=TEMPLATE,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1)


# ══════════════════════════════════════════════════════════
#  TILE SPECS FOR THE DASHBOARD
# ══════════════════════════════════════════════════════════
# recommend_charts returns rendered figures, which the dashboard cannot
# use: its tiles must survive a filter change, a resize and a delete, so
# it needs the *recipe* rather than the picture. Lacking one, the page
# picked its own charts in the browser from column order — and that is
# how a finance dashboard opened on "budget by account" (a budget is
# assigned per account, so the bars restate how the file was built), an
# education one on "cohort by module" (an average of a year number), and
# a healthcare one on "age by department" instead of length of stay.
#
# Same column-role logic as recommend_charts, emitted as specs.

# A dimension with more levels than this makes an unreadable bar chart.
MAX_TILE_DIMENSION_LEVELS = 30


def recommend_tiles(df: pd.DataFrame, domain: str = "general") -> List[Dict]:
    """Up to six dashboard tiles, chosen by column role.

    Each is {id, title, type, x, y, agg} — exactly what /charts/build
    takes, so the dashboard can re-render every tile whenever a filter
    changes without asking what it was showing.
    """
    if df is None or df.empty:
        return []

    cols = _get_analysis_columns(df)
    metric = _pick_primary_metric(cols, domain, df_ref=df)
    if metric is None:
        logger.info("no chartable metric among %d columns", len(df.columns))
        return []

    dim = _pick_best_dimension(df, cols, metric)
    # "auto" defers to _agg_for_metric, which this codebase calls its
    # single source of truth for how a measure is aggregated. Hardcoding
    # "sum" here bypassed it and charted the TOTAL monthly income of
    # each attrition group — 15M against 3M, which is a headcount chart
    # wearing a salary label. That rule already knows income is averaged.
    agg = "auto"
    tiles: List[Dict] = []

    def add(title, kind, x=None, y=None, how=None):
        tiles.append({"id": "t{}".format(len(tiles) + 1), "title": title,
                      "type": kind, "x": x, "y": y, "agg": how or "auto"})

    if dim:
        add("{} by {}".format(_label(metric), _label(dim)), "bar", dim, metric,
            agg)

    # The finding the rest of the app leads with, as a chart. A dashboard
    # that never shows the outcome is a different product from the
    # Insights page looking at the same file.
    outcome = _outcome_for_tiles(df)
    if outcome:
        # When the outcome is itself the leading dimension — an HR file
        # grouped by Attrition — charting its rate against itself says
        # 0% and 100%. Cut it by the next dimension instead.
        by = dim if (dim and dim != outcome) else _second_dimension(
            df, cols, metric, dim)
        if by and by != outcome:
            add("{} rate by {}".format(_label(outcome), _label(by)),
                "bar", by, outcome, "mean")

    if cols["date_cols"]:
        add("{} over time".format(_label(metric)), "line",
            cols["date_cols"][0], metric, agg)

    add("Distribution of {}".format(_label(metric)), "histogram", metric)

    # A second dimension, so the reader can see the measure cut two ways.
    second = _second_dimension(df, cols, metric, dim)
    if second:
        add("{} by {}".format(_label(metric), _label(second)), "bar",
            second, metric, agg)

    if len(cols["all_metrics"]) >= 3:
        add("Correlation", "heatmap")

    return tiles[:6]


def _label(name) -> str:
    from app.engines.present import label as _present_label
    try:
        return _present_label(name)
    except Exception:
        logger.debug("label failed for %r", name, exc_info=True)
        return str(name)


def _outcome_for_tiles(df: pd.DataFrame) -> Optional[str]:
    """The binary outcome worth charting, if the file has one."""
    try:
        from app.engines.domains.outcome_discovery import discover_outcomes
        found = discover_outcomes(df, limit=1)
        return found[0].column if found else None
    except Exception:
        logger.debug("outcome discovery failed for tiles", exc_info=True)
        return None


def _second_dimension(df: pd.DataFrame, cols: Dict, metric: str,
                      first: Optional[str]) -> Optional[str]:
    """Another grouping column, so the dashboard is not one cut repeated."""
    for col in cols.get("cat_cols", []):
        if col == first or col == metric:
            continue
        try:
            levels = int(df[col].nunique(dropna=True))
        except Exception:
            logger.debug("level count failed for %s", col, exc_info=True)
            continue
        if 2 <= levels <= MAX_TILE_DIMENSION_LEVELS:
            return col
    return None
