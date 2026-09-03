"""
engines/charts/style.py — how a chart looks, and what its axes mean.

Split out of chart_exporter.py, which had grown to 1,217 lines covering
three unrelated jobs: how a chart looks, how it is drawn, and which
charts are worth drawing at all. Every change to any of them touched the
same file.

This half is the one that decides presentation: the palettes, the axis
labels, the reference lines, and — the part that is not really styling —
which aggregate a metric deserves. Summing a satisfaction score is
meaningless and averaging revenue hides the total, so `_agg_for_metric`
lives here with the naming rules it depends on.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")                      # no display; PNG bytes only
import matplotlib.pyplot as plt            # noqa: E402

from app.engines import present as _present  # noqa: E402

logger = logging.getLogger(__name__)

import io
import re



def _is_grouping_column(df, col, max_unique=25) -> bool:
    """True if `col` is a genuine categorical grouping dimension — not an
    ID column that happens to have a 'reasonable' cardinality by coincidence.

    A '2 <= nunique <= max_unique' cardinality check alone isn't enough:
    for a small dataset, a pure ID column (one row per group, e.g.
    EmployeeID with 18 unique values across 18 rows) can land inside that
    window just as easily as a real category with 18 real groups. The
    'pick the column with the biggest between-group spread' selection
    logic then actively PREFERS ID columns, since 1-row groups produce
    artificially extreme min/max spread — this isn't a rare edge case,
    it reproduces on any dataset small enough that len(df) is itself
    under ~30-40 rows.

    Requires actual grouping: on average, each category must cover more
    than one row (nunique meaningfully less than row count), on top of
    the existing absolute-cardinality bound.
    """
    try:
        nun = df[col].nunique(dropna=True)
    except Exception:
        return False
    if not (2 <= nun <= max_unique):
        return False
    # Average group size must be > 1 for this to be a real grouping
    # dimension rather than a near-1:1 identifier.
    if nun > max(len(df) * 0.8, 1):
        return False
    return True



def _pretty(name: str) -> str:
    """Human label for a column: split camelCase and snake_case, title-case
    each word but keep acronym runs. 'EnvironmentSatisfaction' -> 'Environment
    Satisfaction' (not the mangled 'Environmentsatisfaction' from .title())."""
    s = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', str(name))
    s = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', s).replace("_", " ")
    return " ".join(w if w.isupper() else w.capitalize() for w in s.split())


LIGHT_COLORS = ["#1565C0", "#0D47A1", "#B71C1C", "#1B5E20", "#4527A0", "#E65100"]
DARK_COLORS  = ["#64B5F6", "#4DB6AC", "#FFB74D", "#CE93D8", "#EF9A9A", "#FFF176"]
GREEN_COLORS = ["#1B5E20", "#2E7D32", "#388E3C", "#43A047", "#1A237E", "#0D47A1"]

# Must be defined at module level before any function references it
_SCORE_KEYWORDS = {"satisfaction", "rating", "score", "evaluation", "performance",
                   "sentiment", "nps", "csat", "quality", "health", "level", "index"}

# Metrics where TOTALING across a group answers a real question — genuine
# volume/monetary flow, not a per-entity attribute. Everything else defaults
# to mean now (see _agg_for_metric below) since most per-row numeric columns
# in a dataset are attributes of an entity (age, tenure, years-since-X,
# distance, num-of-Y) where a group SUM is meaningless — e.g. 'Total Years
# Since Last Promotion: 1,320' or 'Total Num Companies Worked: 887' say
# nothing interpretable, whereas 'Total Revenue' or 'Total Units Sold' do.
_SUM_KEYWORDS = {"revenue", "sales", "amount", "spend", "cost", "price", "quantity",
                  "units", "transactions", "orders", "volume", "headcount", "count",
                  "profit", "expense", "expenditure", "budget", "qty"}


def _tick_budget(n_ticks: int, base: int = 14) -> int:
    """How many characters a tick label can take, given how many there are.

    A fixed budget cut "Research & Development" to "Research & De..." on a
    chart with three bars and an inch of room under each. Fewer bars, more
    room.
    """
    if n_ticks <= 3:
        return max(base, 26)
    if n_ticks <= 5:
        return max(base, 20)
    if n_ticks <= 8:
        return max(base, 16)
    return base


def _axis_label(y_col: str, agg_func: str) -> str:
    """Name the axis after what was actually computed."""
    prefix = "Average " if agg_func == "mean" else "Total "
    return prefix + _pretty(y_col)


def _reference_line(df, y_col: str, agg, agg_func: str):
    """The comparison line on a grouped bar chart, and what to call it.

    Taking the mean of the group means gives an average of averages: on
    an income chart with 627, 701 and 142 people in the three groups it
    read 8,311 where the actual average income was 8,583 — and the table
    of the same figures on the same page said 8,583. Averaging a mean
    across groups weights a group of 142 the same as one of 701.
    """
    try:
        if agg_func == "mean":
            return float(df[y_col].mean()), "Overall average"
        # For a summed metric the group totals have no overall mean to
        # compare against; the average group total is the honest line,
        # named as such.
        return float(agg[y_col].mean()), "Average per group"
    except Exception:
        logger.debug("reference line failed for %r", y_col, exc_info=True)
        return float(agg[y_col].mean()), "Average"


def _agg_for_metric(y_col: str):
    """Single source of truth for how a metric is aggregated across a category.

    Score/rating metrics (0-1 or 1-5 scales) are averaged — a *total* of a
    rating is meaningless. Genuine volume/monetary metrics (revenue, spend,
    units, headcount) are summed — a total answers a real business question.
    Everything else (the common case: per-entity attributes like age, tenure,
    distance, years-since-X) is averaged too, since summing an attribute
    across a group produces a number with no interpretable meaning. BOTH the
    chart renderer AND the pair-selection logic call this, so the category a
    chart highlights can never disagree with the aggregation the narrative
    describes. Returns (agg_func, is_score).
    """
    y_lower = y_col.lower()
    is_score = any(kw in y_lower for kw in _SCORE_KEYWORDS)
    if is_score:
        return "mean", True
    is_volume = any(kw in y_lower for kw in _SUM_KEYWORDS)
    return ("sum" if is_volume else "mean"), False


def _get_style(theme_name: str) -> dict:
    if theme_name == "Dark Tech":
        return {
            "figure.facecolor": "#07080f",
            "axes.facecolor":   "#0e0f1a",
            "axes.edgecolor":   "#1e2035",
            "axes.labelcolor":  "#dde1f5",
            "xtick.color":      "#1e3a5f",
            "ytick.color":      "#1e3a5f",
            "text.color":       "#dde1f5",
            "grid.color":       "#1e2035",
            "grid.alpha":       0.8,
        }
    else:
        return {
            "figure.facecolor": "#ffffff",
            "axes.facecolor":   "#F8FAFF",
            "axes.edgecolor":   "#CBD5E1",
            "axes.labelcolor":  "#0F172A",   # near-black for labels
            "xtick.color":      "#0F172A",   # dark tick labels
            "ytick.color":      "#0F172A",   # dark tick labels
            "text.color":       "#0A1628",   # very dark for titles
            "grid.color":       "#CBD5E1",
            "grid.alpha":       0.5,
        }


def _get_colors(theme_name: str) -> list:
    if theme_name == "Dark Tech":
        return DARK_COLORS
    elif theme_name == "Executive Green":
        return GREEN_COLORS
    return LIGHT_COLORS


def _apply_style(ax, style: dict):
    ax.set_facecolor(style["axes.facecolor"])
    ax.tick_params(colors=style["xtick.color"])
    # Value-axis gridlines only — a full grid competes with the data.
    ax.grid(True, axis="y", color=style["grid.color"],
            alpha=style["grid.alpha"] * 0.7, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(style["axes.edgecolor"])
    ax.spines["bottom"].set_color(style["axes.edgecolor"])


def _footnote(fig, n: int, style: dict, note: str = ""):
    """Sample-size + source annotation — every client-grade chart states
    what it was computed from."""
    txt = "n = {:,} records | Source: client dataset".format(n)
    if note:
        txt += " | " + note
    fig.text(0.99, 0.005, txt, ha="right", va="bottom",
             fontsize=7, color="#94A3B8")


def _gap_headline(agg_sorted, x_col, y_col, fmt) -> str:
    """Chart headline that states the finding, not the axes.
    'Sales leads at 0.63 — HR trails at 0.55' beats 'Score by Department'."""
    try:
        # 22 rather than 16, and cut on a word boundary: at 16 the
        # headline read "'Research & Deve...' highest at 8,872".
        top_name = _present.truncate(_present.value(agg_sorted.iloc[0, 0]), 22)
        top_v = float(agg_sorted.iloc[0][y_col])
        bot_name = _present.truncate(_present.value(agg_sorted.iloc[-1, 0]), 22)
        bot_v = float(agg_sorted.iloc[-1][y_col])
        # Relative spread vs the top (bounded) — avoids 'inf× gap' when the
        # lowest group is ~0.
        rel_spread = abs(top_v - bot_v) / max(abs(top_v), 1e-9)
        if rel_spread < 0.03:
            return "Consistent across {}: {} to {}".format(
                _present.label(x_col), fmt.format(bot_v), fmt.format(top_v))
        gap = ""
        if bot_v != 0 and abs(top_v / bot_v) >= 1.15:
            gap = " ({:.1f}× gap)".format(abs(top_v / bot_v))
        elif bot_v == 0:
            gap = " (lowest ≈ 0)"
        # Unquoted: these are the client's own segment names, not string
        # literals from a program.
        return "{} highest at {} — {} lowest at {}{}".format(
            top_name, fmt.format(top_v), bot_name, fmt.format(bot_v), gap)
    except Exception:
        logger.warning("chart headline computation failed", exc_info=True)
        return ""


def fig_to_bytes(fig: plt.Figure, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    data = buf.read()
    plt.close(fig)
    return data

