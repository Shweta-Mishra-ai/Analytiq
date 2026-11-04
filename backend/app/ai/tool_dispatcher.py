import logging
import pandas as pd

from app.engines.chart_exporter import fmt
from dataclasses import dataclass
from typing import Optional, Any
from app.engines import chart_engine

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    success: bool
    text_output: Optional[str]       = None
    figure: Optional[Any]            = None
    dataframe: Optional[pd.DataFrame] = None
    error: Optional[str]             = None
    explanation: str                 = ""


def dispatch(df: pd.DataFrame, tool: str, params: dict, explanation: str = "") -> ToolResult:
    REGISTRY = {
        "aggregate":       _aggregate,
        "filter":          _filter,
        "top_n":           _top_n,
        "describe_column": _describe_col,
        "correlation":     _correlation,
        "count_values":    _count_values,
        "plot_bar":        _plot_bar,
        "plot_line":       _plot_line,
        "plot_scatter":    _plot_scatter,
        "plot_histogram":  _plot_histogram,
        "plot_pie":        _plot_pie,
        "plot_heatmap":    _plot_heatmap,
        "none":            _no_tool,
    }

    if tool not in REGISTRY:
        return ToolResult(
            success=False,
            error=f"Unknown tool '{tool}'.",
            explanation=explanation
        )

    try:
        result             = REGISTRY[tool](df, params)
        result.explanation = explanation
        return result
    except KeyError as e:
        missing = e.args[0] if e.args else ""
        # A parameter the model left out raised the same KeyError as a wrong
        # column, and was reported as "Column not found: 'value'".
        if isinstance(missing, str) and missing not in df.columns and missing not in (
                str(v) for v in (params or {}).values()):
            return ToolResult(
                success=False,
                error=f"The request was missing '{missing}' for '{tool}'.",
                explanation=explanation)
        return ToolResult(
            success=False,
            error=f"Column not found: {e}. Available: {list(df.columns)}",
            explanation=explanation
        )
    except Exception as e:
        return ToolResult(
            success=False,
            error=f"Error: {str(e)}",
            explanation=explanation
        )


# ── DATA TOOLS ────────────────────────────────────────────

_AGG_FUNCS = {"sum", "mean", "count", "max", "min", "median"}


def _aggregate(df, p):
    grp = p["group_col"]
    val = p["value_col"]
    agg = p.get("agg_func", "sum")
    # The model chooses this string and pandas looks it up by name, so it
    # is held to the functions the prompt offers.
    if agg not in _AGG_FUNCS:
        return ToolResult(success=False, error=f"Unsupported aggregation '{agg}'")

    res  = df.groupby(grp)[val].agg(agg).reset_index()
    res  = res.sort_values(val, ascending=False)
    # Lead with what the numbers say, as an analyst would, then the list.
    what = {"sum": "total", "mean": "average", "median": "median", "count": "count of",
            "max": "highest", "min": "lowest"}[agg]
    if len(res) >= 2:
        top, low = res.iloc[0], res.iloc[-1]
        text = (f"**{top[grp]}** has the highest {what} {val} ({fmt(top[val])}) and "
                f"**{low[grp]}** the lowest ({fmt(low[val])}), across {len(res):,} "
                f"{grp} groups.")
        p_val = _groups_differ_p(df, grp, val, agg)
        if p_val is not None and p_val >= 0.05:
            # The ranking was stated as a finding however small the gaps.
            text += (f" The gaps are within what chance produces (p={p_val:.2f}), so "
                     f"this order should not be relied on.")
        text += "\n\n"
    else:
        text = ""
    for _, row in res.head(15).iterrows():
        text += f"- {row[grp]}: {fmt(row[val])}\n"
    if len(res) > 15:
        text += f"- … and {len(res) - 15:,} more\n"
    return ToolResult(success=True, text_output=text.strip(), dataframe=res)


def _groups_differ_p(df, grp, val, agg):
    """p-value for "the groups differ" on the quantity the ranking used:
    values per row for mean/median/max/min, row counts for count, either for
    a sum (a total is rows × value per row). None when it cannot be tested."""
    from scipy import stats as sc
    sub = df[[grp, val]].dropna() if agg != "count" else df[[grp]].dropna()
    tests = []
    try:
        if agg in ("count", "sum"):
            counts = sub[grp].value_counts()
            if len(counts) >= 2:
                tests.append(float(sc.chisquare(counts.values).pvalue))
        if agg != "count" and pd.api.types.is_numeric_dtype(sub[val]):
            groups = [g[val].values for _, g in sub.groupby(grp) if len(g) >= 3]
            if len(groups) >= 2:
                tests.append(float(sc.kruskal(*groups).pvalue))
    except ValueError:
        return None
    if not tests:
        return None
    return min(1.0, min(tests) * len(tests))          # Bonferroni over the two


def _filter(df, p):
    col = p["column"]
    op  = p["operator"]
    val = p["value"]

    if op == ">":        mask = df[col] > float(val)
    elif op == "<":      mask = df[col] < float(val)
    elif op == "==":     mask = df[col].astype(str) == str(val)
    elif op == "!=":     mask = df[col].astype(str) != str(val)
    elif op == "contains":
        mask = df[col].astype(str).str.contains(str(val), case=False, na=False,
                                                regex=False)
    else:
        return ToolResult(success=False, error=f"Unknown operator: {op}")

    filtered = df[mask]
    share = len(filtered) / max(len(df), 1) * 100
    return ToolResult(
        success=True,
        text_output=(f"**{len(filtered):,}** of {len(df):,} rows ({share:.0f}%) have "
                     f"{col} {op} {val}."),
        dataframe=filtered
    )


def _top_n(df, p):
    col = p["sort_col"]
    n   = int(p.get("n", 10))
    asc = bool(p.get("ascending", False))
    res = df.sort_values(col, ascending=asc).head(n)
    text = f"The {len(res):,} rows with the {'lowest' if asc else 'highest'} {col}"
    if len(res) and pd.api.types.is_numeric_dtype(res[col]):
        text += f", from {fmt(res[col].iloc[0])} to {fmt(res[col].iloc[-1])}"
    return ToolResult(success=True, text_output=text + ".", dataframe=res)


def _describe_col(df, p):
    name  = p["column"]
    s     = df[name]
    c     = s.dropna()
    blank = len(s) - len(c)
    tail  = f" {blank:,} rows are blank." if blank else ""
    if pd.api.types.is_datetime64_any_dtype(s) and len(c):
        text = (f"{name} runs from **{c.min().date()}** to **{c.max().date()}** "
                f"({c.nunique():,} different dates).")
    elif pd.api.types.is_numeric_dtype(s) and len(c):
        q1, med, q3 = (float(v) for v in c.quantile([0.25, 0.5, 0.75]))
        text = (f"A typical {name} is **{fmt(med)}** (the median); half the rows lie "
                f"between {fmt(q1)} and {fmt(q3)}, and the full range is {fmt(c.min())} to "
                f"{fmt(c.max())}.")
        if med and abs(c.mean() - med) / abs(med) > 0.1:
            text += (f" The average, {fmt(c.mean())}, is pulled "
                     f"{'up' if c.mean() > med else 'down'} by a few extreme values.")
    elif len(c):
        vc = c.value_counts()
        text = (f"{name} has {len(vc):,} different values; the most common is "
                f"**{vc.index[0]}** ({vc.iloc[0] / len(c) * 100:.0f}% of rows).")
        if len(vc) > 1:
            text += " Next: " + ", ".join(
                f"{k} ({v / len(c) * 100:.0f}%)" for k, v in vc.iloc[1:5].items()) + "."
    else:
        text = f"{name} is empty."
    return ToolResult(success=True, text_output=text + tail)


def _correlation(df, p):
    a, b      = p["col_a"], p["col_b"]
    both      = df[[a, b]].dropna()
    n         = len(both)
    if n < 3 or both[a].nunique() < 2 or both[b].nunique() < 2:
        # A constant or near-empty column gave r = nan, which fell through
        # every comparison to "move in opposite directions slightly".
        return ToolResult(success=True, text_output=(
            f"A correlation between {a} and {b} cannot be measured: one of them has "
            f"fewer than two different values on the {n:,} rows where both are filled."))
    from scipy.stats import pearsonr
    r, p_val  = (float(x) for x in pearsonr(both[a], both[b]))
    if p_val >= 0.05:
        # r = 0.45 on five rows is chance; the size alone was reported.
        text = (f"There is no reliable link between {a} and {b} in this data "
                f"(r = {r:.2f}, p = {p_val:.2f}, {n:,} rows).")
    elif abs(r) < 0.2:
        text = (f"{a} and {b} are barely related in this data (r = {r:.2f}, "
                f"{n:,} rows).")
    else:
        how = "closely" if abs(r) > 0.7 else "somewhat" if abs(r) > 0.4 else "slightly"
        way = "rise together" if r > 0 else "move in opposite directions"
        text = (f"{a} and {b} {way} {how} (r = {r:.2f}, {n:,} rows). That is a link, "
                f"not proof that one causes the other.")
    return ToolResult(success=True, text_output=text)


def _count_values(df, p):
    vc             = df[p["column"]].value_counts()
    counts         = vc.head(20).reset_index()
    counts.columns = [p["column"], "count"]
    text = (f"{p['column']} has {len(vc):,} different values; **{vc.index[0]}** is the most "
            f"common ({vc.iloc[0]:,} rows)." if len(vc) else f"{p['column']} is empty.")
    return ToolResult(success=True, text_output=text, dataframe=counts)


# ── CHART TOOLS ───────────────────────────────────────────

def _plot_bar(df, p):
    return ToolResult(success=True,
        figure=chart_engine.make_bar(df, p["x"], p["y"], p.get("title", "")))

def _plot_line(df, p):
    return ToolResult(success=True,
        figure=chart_engine.make_line(df, p["x"], p["y"], p.get("title", "")))

def _plot_scatter(df, p):
    return ToolResult(success=True,
        figure=chart_engine.make_scatter(df, p["x"], p["y"], p.get("color"), p.get("title", "")))

def _plot_histogram(df, p):
    return ToolResult(success=True,
        figure=chart_engine.make_histogram(df, p["column"], int(p.get("nbins", 30)), p.get("title", "")))

def _plot_pie(df, p):
    return ToolResult(success=True,
        figure=chart_engine.make_pie(df, p["names_col"], p["values_col"], p.get("title", "")))

def _plot_heatmap(df, p):
    return ToolResult(success=True,
        figure=chart_engine.make_heatmap(df))

def _no_tool(df, p):
    return ToolResult(
        success=True,
        text_output=("I couldn't match that to this data. Try naming the columns, for "
                     "example \"average salary by department\".")
    )
