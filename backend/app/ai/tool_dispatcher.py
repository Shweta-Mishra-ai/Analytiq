import logging
import pandas as pd
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

def _aggregate(df, p):
    grp = p["group_col"]
    val = p["value_col"]
    agg = p.get("agg_func", "sum")

    res  = df.groupby(grp)[val].agg(agg).reset_index()
    res  = res.sort_values(val, ascending=False)
    text = f"**{agg.capitalize()} of `{val}` by `{grp}`:**\n"
    for _, row in res.head(15).iterrows():
        text += f"- **{row[grp]}**: {row[val]:,.2f}\n"
    return ToolResult(success=True, text_output=text, dataframe=res)


def _filter(df, p):
    col = p["column"]
    op  = p["operator"]
    val = p["value"]

    if op == ">":        mask = df[col] > float(val)
    elif op == "<":      mask = df[col] < float(val)
    elif op == "==":     mask = df[col].astype(str) == str(val)
    elif op == "!=":     mask = df[col].astype(str) != str(val)
    elif op == "contains":
        mask = df[col].astype(str).str.contains(str(val), case=False, na=False)
    else:
        return ToolResult(success=False, error=f"Unknown operator: {op}")

    filtered = df[mask]
    return ToolResult(
        success=True,
        text_output=f"**{len(filtered):,} rows** match `{col} {op} {val}`.",
        dataframe=filtered
    )


def _top_n(df, p):
    col = p["sort_col"]
    n   = int(p.get("n", 10))
    asc = bool(p.get("ascending", False))
    res = df.sort_values(col, ascending=asc).head(n)
    return ToolResult(
        success=True,
        text_output=f"**Top {n} rows by `{col}`:**",
        dataframe=res
    )


def _describe_col(df, p):
    s     = df[p["column"]]
    lines = [
        f"**Column: `{p['column']}`**",
        f"- Type: `{s.dtype}`",
        f"- Non-null: {s.notna().sum():,} / {len(s):,}",
        f"- Unique values: {s.nunique():,}",
    ]
    if pd.api.types.is_numeric_dtype(s):
        c = s.dropna()
        lines += [
            f"- Mean: {c.mean():,.2f}",
            f"- Median: {c.median():,.2f}",
            f"- Std Dev: {c.std():,.2f}",
            f"- Min: {c.min():,} | Max: {c.max():,}",
        ]
    else:
        top5 = s.value_counts().head(5)
        lines.append(f"- Top values: {', '.join(str(v) for v in top5.index.tolist())}")
    return ToolResult(success=True, text_output="\n".join(lines))


def _correlation(df, p):
    r         = df[[p["col_a"], p["col_b"]]].dropna().corr().iloc[0, 1]
    strength  = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
    direction = "positive" if r > 0 else "negative"
    return ToolResult(
        success=True,
        text_output=(
            f"**Correlation: `{p['col_a']}` & `{p['col_b']}`**\n\n"
            f"r = **{r:.3f}** — {strength} {direction} correlation."
        )
    )


def _count_values(df, p):
    counts         = df[p["column"]].value_counts().head(20).reset_index()
    counts.columns = [p["column"], "count"]
    return ToolResult(
        success=True,
        text_output=f"**Value counts for `{p['column']}`:**",
        dataframe=counts
    )


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
        text_output="I couldn't find a matching analysis. Try rephrasing your question."
    )
