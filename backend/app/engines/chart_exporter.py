"""
engines/chart_exporter.py — static (matplotlib) chart rendering for PDF
reports.

Ported from dataforge-ai, which carried roughly 1,000 lines of chart
selection and rendering this app had lost in an earlier port. What came
back with it:

  * `_agg_for_metric` — the aggregation a metric actually deserves.
    Analytiq's bar chart summed unconditionally, so a report could show
    "Total Employee Satisfaction: 4,410" or a sum of ages, numbers with
    no interpretable meaning.
  * `_pretty` — human labels. Charts were titled with raw column names
    ("YearsWithCurrManager by Attrition") in client-facing PDFs.
  * `_best_metric_by_category` — chart the pair that actually varies,
    rather than the first numeric column against the first categorical.
  * `_SENSITIVE_DIMS` — never auto-chart a metric by marital status,
    gender or race. An automatically generated demographic comparison is
    spurious at best.
  * Ranked bar, bullet, risk-heatmap and driver-importance charts.

Deviations from the dataforge original, all deliberate:
  * The correlation-heatmap column ranking no longer crashes on a
    read-only array (np.fill_diagonal on .values), which silently
    degraded the heatmap to the first ten columns by position.
  * Metric selection also uses Analytiq's value-aware is_id_column, not
    only a name-substring match.
  * Month-end resampling goes through app.services.dtypes.MONTH_END.
"""
import io
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional
import logging
logger = logging.getLogger(__name__)
from app.engines.pdf_primitives import truncate_label
from app.engines.domains.base import is_id_column
from app.services.dtypes import MONTH_END


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
        top_name, top_v = truncate_label(str(agg_sorted.iloc[0, 0]), 16), float(agg_sorted.iloc[0][y_col])
        bot_name, bot_v = truncate_label(str(agg_sorted.iloc[-1, 0]), 16), float(agg_sorted.iloc[-1][y_col])
        # Relative spread vs the top (bounded) — avoids 'inf× gap' when the
        # lowest group is ~0.
        rel_spread = abs(top_v - bot_v) / max(abs(top_v), 1e-9)
        if rel_spread < 0.03:
            return "Consistent across {}: {} – {}".format(
                x_col.replace("_", " "), fmt.format(bot_v), fmt.format(top_v))
        gap = ""
        if bot_v != 0 and abs(top_v / bot_v) >= 1.15:
            gap = " ({:.1f}× gap)".format(abs(top_v / bot_v))
        elif bot_v == 0:
            gap = " (lowest ≈ 0)"
        return "'{}' highest at {} — '{}' lowest at {}{}".format(
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


def make_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    theme_name: str = "Corporate Light",
    top_n: int = 15,
) -> bytes:
    """Smart bar: mean for score/rating metrics, sum for revenue/count metrics.
    Fixes the bug where satisfaction_level (0-1 range) displayed as 2,544 (count)."""
    style  = _get_style(theme_name)
    colors = _get_colors(theme_name)

    agg_func, is_score = _agg_for_metric(y_col)
    fmt      = "{:.3f}" if is_score else "{:,.0f}"
    y_label  = ("Avg " if is_score else "Total ") + _pretty(y_col)

    agg = (df.groupby(x_col)[y_col]
             .agg(agg_func)
             .reset_index()
             .sort_values(y_col, ascending=False)
             .head(top_n))

    org_avg = float(agg[y_col].mean())
    bar_colors = [colors[0] if v >= org_avg else "#64748B"  # slate-500 — below avg
                  for v in agg[y_col]]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(style["figure.facecolor"])
    _apply_style(ax, style)

    bars = ax.bar(
        range(len(agg)), agg[y_col],
        color=bar_colors, alpha=0.88,
        edgecolor=style["axes.edgecolor"], linewidth=0.5
    )

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.012,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=8, color=style["text.color"], fontweight="bold"
        )

    ax.axhline(org_avg, color=(colors[1] if len(colors) > 1 else "#888888"),
               linestyle="--", linewidth=1.2, alpha=0.7,
               label="Avg: {}".format(fmt.format(org_avg)))
    ax.legend(fontsize=8, framealpha=0)

    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(
        [truncate_label(str(v), 14) for v in agg[x_col]],
        rotation=35, ha="right", fontsize=8.5
    )
    ax.set_ylabel(y_label, fontsize=9, color=style["axes.labelcolor"])
    # Headline states the finding; the descriptive title becomes the subtitle
    headline = _gap_headline(agg, x_col, y_col, fmt)
    desc = title or "{} by {}".format(y_label, _pretty(x_col))
    if headline:
        ax.set_title(headline, fontsize=11.5, fontweight="bold",
                     color=style["text.color"], pad=18, loc="left")
        ax.text(0, 1.02, desc, transform=ax.transAxes,
                fontsize=8.5, color="#64748B")
    else:
        ax.set_title(desc, fontsize=11, fontweight="bold",
                     color=style["text.color"], pad=12)
    _footnote(fig, int(df[[x_col, y_col]].dropna().shape[0]), style,
              note="aggregation: " + agg_func)
    fig.tight_layout()
    return fig_to_bytes(fig, dpi=170)


def make_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    theme_name: str = "Corporate Light",
) -> bytes:
    """
    Smart line chart:
    - If x is datetime: aggregate by month
    - If x is numeric with >50 unique values: bin into 20 buckets
    - If x is categorical: bar chart style instead
    - Otherwise: plot directly
    """
    style  = _get_style(theme_name)
    colors = _get_colors(theme_name)

    data = df[[x_col, y_col]].dropna().copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(style["figure.facecolor"])
    _apply_style(ax, style)

    x_is_datetime = pd.api.types.is_datetime64_any_dtype(data[x_col])
    x_is_numeric  = pd.api.types.is_numeric_dtype(data[x_col])
    n_unique      = data[x_col].nunique()

    if x_is_datetime:
        # Aggregate by month
        data[x_col] = pd.to_datetime(data[x_col])
        data = data.set_index(x_col).resample(MONTH_END)[y_col].mean().reset_index()
        x_vals = range(len(data))
        y_vals = data[y_col].values
        labels = [str(d)[:7] for d in data[x_col]]

    elif x_is_numeric and n_unique > 50:
        # Bin into 20 buckets for clean trend
        data["_bin"] = pd.cut(data[x_col], bins=20)
        agg = data.groupby("_bin")[y_col].mean().reset_index()
        agg = agg.dropna()
        x_vals = range(len(agg))
        y_vals = agg[y_col].values
        labels = [truncate_label(str(b), 10) for b in agg["_bin"]]

    elif x_is_numeric and n_unique <= 50:
        # Sort and plot directly
        data = data.sort_values(x_col)
        x_vals = range(len(data))
        y_vals = data[y_col].values
        labels = [str(v) for v in data[x_col]]

    else:
        # Categorical x — aggregate by mean
        agg = data.groupby(x_col)[y_col].mean().reset_index().sort_values(y_col, ascending=False).head(15)
        x_vals = range(len(agg))
        y_vals = agg[y_col].values
        labels = [truncate_label(str(v), 12) for v in agg[x_col]]

    ax.plot(
        x_vals, y_vals,
        color=colors[0], linewidth=2.5,
        marker="o", markersize=4,
        markerfacecolor=colors[1],
        markeredgecolor=colors[0]
    )
    ax.fill_between(x_vals, y_vals, alpha=0.12, color=colors[0])

    # X axis labels — max 10
    step = max(1, len(labels) // 10)
    ax.set_xticks(list(x_vals)[::step])
    ax.set_xticklabels(labels[::step], rotation=35, ha="right", fontsize=8)

    ax.set_ylabel(y_col, fontsize=9, color=style["axes.labelcolor"])
    # Headline: direction and size of the change across the period
    try:
        first_v, last_v = float(y_vals[0]), float(y_vals[-1])
        if first_v != 0 and len(y_vals) >= 3:
            chg = (last_v - first_v) / abs(first_v) * 100
            word = "up" if chg > 1 else "down" if chg < -1 else "flat"
            headline = "{} {} {:.0f}% across the period ({:.3g} → {:.3g})".format(
                y_col.replace("_", " "), word, abs(chg), first_v, last_v) \
                if word != "flat" else \
                "{} is stable across the period (~{:.3g})".format(
                    y_col.replace("_", " "), last_v)
            ax.set_title(headline, fontsize=11.5, fontweight="bold",
                         color=style["text.color"], pad=18, loc="left")
            ax.text(0, 1.02, title or "{} Trend".format(y_col),
                    transform=ax.transAxes, fontsize=8.5, color="#64748B")
        else:
            raise ValueError("headline not computable")
    except Exception:
        ax.set_title(title or "{} Trend".format(y_col),
                     fontsize=11, fontweight="bold",
                     color=style["text.color"], pad=10)
    _footnote(fig, int(len(df[[x_col, y_col]].dropna())), style)
    fig.tight_layout()
    return fig_to_bytes(fig)


def make_histogram(
    df: pd.DataFrame,
    col: str,
    title: str = "",
    theme_name: str = "Corporate Light",
    bins: int = 25,
) -> bytes:
    style  = _get_style(theme_name)
    colors = _get_colors(theme_name)
    data   = df[col].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(style["figure.facecolor"])
    _apply_style(ax, style)

    ax.hist(data, bins=bins, color=colors[0],
            alpha=0.8, edgecolor=style["axes.edgecolor"],
            linewidth=0.4)

    mean_val   = data.mean()
    median_val = data.median()

    ax.axvline(mean_val, color=colors[2],
               linestyle="--", linewidth=1.8,
               label="Mean: {:.2f}".format(mean_val))
    ax.axvline(median_val, color=colors[3] if len(colors) > 3 else colors[1],
               linestyle=":", linewidth=1.8,
               label="Median: {:.2f}".format(median_val))
    ax.legend(fontsize=8)

    ax.set_xlabel(col, fontsize=9, color=style["axes.labelcolor"])
    ax.set_ylabel("Frequency", fontsize=9, color=style["axes.labelcolor"])
    # Headline: typical value + spread — what a reader actually needs
    try:
        p10, p90 = float(data.quantile(0.10)), float(data.quantile(0.90))
        headline = "Typical {} is {:.2f} (P10 {:.2f} – P90 {:.2f})".format(
            col.replace("_", " "), float(median_val), p10, p90)
        ax.set_title(headline, fontsize=11.5, fontweight="bold",
                     color=style["text.color"], pad=18, loc="left")
        ax.text(0, 1.02, title or "Distribution: {}".format(col),
                transform=ax.transAxes, fontsize=8.5, color="#64748B")
    except Exception:
        logger.warning("histogram headline failed", exc_info=True)
        ax.set_title(title or "Distribution: {}".format(col),
                     fontsize=11, fontweight="bold",
                     color=style["text.color"], pad=10)
    _footnote(fig, int(len(data)), style)
    fig.tight_layout()
    return fig_to_bytes(fig)


def make_ranked_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    theme_name: str = "Corporate Light",
) -> bytes:
    """Horizontal ranked bar — clearer than a pie chart for comparing
    many roughly-equal categories on a score metric."""
    style  = _get_style(theme_name)
    colors = _get_colors(theme_name)

    agg_func, is_score = _agg_for_metric(y_col)
    fmt      = "{:.3f}" if is_score else "{:,.0f}"

    agg = (df.groupby(x_col)[y_col]
             .agg(agg_func)
             .reset_index()
             .sort_values(y_col, ascending=True)
             .head(15))

    org_avg = float(agg[y_col].mean())
    bar_colors = [colors[0] if v >= org_avg else "#64748B"
                  for v in agg[y_col]]

    fig, ax = plt.subplots(figsize=(9, max(4, len(agg) * 0.45)))
    fig.patch.set_facecolor(style["figure.facecolor"])
    _apply_style(ax, style)

    bars = ax.barh(
        range(len(agg)), agg[y_col],
        color=bar_colors, alpha=0.88,
        edgecolor=style["axes.edgecolor"], linewidth=0.5
    )

    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.005, bar.get_y() + bar.get_height() / 2,
                fmt.format(w), va="center", ha="left", fontsize=8,
                color=style["text.color"], fontweight="bold")

    ax.axvline(org_avg, color=(colors[3] if len(colors) > 3 else colors[1]),
               linestyle="--", linewidth=1.2, alpha=0.7,
               label="Avg: {}".format(fmt.format(org_avg)))
    ax.legend(fontsize=8, framealpha=0)

    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels([truncate_label(str(v), 16) for v in agg[x_col]], fontsize=9)
    ax.set_xlabel(("Avg " if is_score else "Total ") + _pretty(y_col),
                  fontsize=9, color=style["axes.labelcolor"])
    headline = _gap_headline(agg.sort_values(y_col, ascending=False), x_col, y_col, fmt)
    desc = title or "{} Ranking by {}".format(
        _pretty(y_col), _pretty(x_col))
    if headline:
        ax.set_title(headline, fontsize=11.5, fontweight="bold",
                     color=style["text.color"], pad=18, loc="left")
        ax.text(0, 1.02, desc, transform=ax.transAxes,
                fontsize=8.5, color="#64748B")
    else:
        ax.set_title(desc, fontsize=11, fontweight="bold",
                     color=style["text.color"], pad=12)
    _footnote(fig, int(df[[x_col, y_col]].dropna().shape[0]), style,
              note="aggregation: " + agg_func)
    fig.tight_layout()
    return fig_to_bytes(fig, dpi=170)


def make_pie_chart(
    df: pd.DataFrame,
    names_col: str,
    values_col: str,
    title: str = "",
    theme_name: str = "Corporate Light",
) -> bytes:
    style  = _get_style(theme_name)
    colors = _get_colors(theme_name)

    agg = (df.groupby(names_col)[values_col]
             .sum()
             .reset_index()
             .sort_values(values_col, ascending=False)
             .head(8))

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(style["figure.facecolor"])

    wedges, texts, autotexts = ax.pie(
        agg[values_col],
        labels=None,
        autopct="%1.1f%%",
        colors=colors[:len(agg)],
        startangle=90,
        pctdistance=0.75,
        wedgeprops={"edgecolor": style["figure.facecolor"], "linewidth": 2}
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color("white")
        at.set_fontweight("bold")

    ax.legend(
        wedges,
        [truncate_label(str(v), 15) for v in agg[names_col]],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=9,
        framealpha=0,
        labelcolor=style["text.color"]
    )
    ax.set_title(title or "{} by {}".format(values_col, names_col),
                 fontsize=11, fontweight="bold",
                 color=style["text.color"], pad=10)
    fig.tight_layout()
    return fig_to_bytes(fig)


def make_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Matrix",
    theme_name: str = "Corporate Light",
) -> bytes:
    style    = _get_style(theme_name)
    from app.engines.pdf_primitives import is_id_col, truncate_label
    num_cols = [c for c in df.select_dtypes(include="number").columns.tolist()
                if not is_id_col(c, df[c])]

    if len(num_cols) < 2:
        return None

    if len(num_cols) > 10:
        # Pick the 10 columns most involved in a strong relationship, not
        # just the first 10 by raw column order — otherwise a genuinely
        # strong pair (e.g. JobLevel-MonthlyIncome) can be invisible on the
        # chart just because both columns happen to sit late in the
        # dataframe, while the chart shows only weak/incidental pairs among
        # whatever came first.
        try:
            full_corr = df[num_cols].corr().abs()
            # Blank the diagonal without writing through .values: on a
            # read-only backing array np.fill_diagonal raises
            # "underlying array is read-only" and the ranking silently
            # falls back to the first 10 columns by order.
            import numpy as _np
            full_corr = full_corr.mask(
                _np.eye(len(full_corr), dtype=bool), 0.0)
            max_partner_corr = full_corr.max(axis=1)
            num_cols = max_partner_corr.sort_values(ascending=False).index[:10].tolist()
        except Exception:
            logger.warning("heatmap column ranking failed — using first 10 by order", exc_info=True)
            num_cols = num_cols[:10]

    corr = df[num_cols[:10]].corr().round(2)
    n    = len(corr)

    fig, ax = plt.subplots(figsize=(max(7, n), max(5, n - 1)))
    fig.patch.set_facecolor(style["figure.facecolor"])
    ax.set_facecolor(style["axes.facecolor"])

    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            ax.text(j, i, "{:.2f}".format(val),
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(val) > 0.5 else style["text.color"])

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([truncate_label(c, 10) for c in corr.columns],
                       rotation=45, ha="right", fontsize=8,
                       color=style["xtick.color"])
    ax.set_yticklabels([truncate_label(c, 10) for c in corr.index],
                       fontsize=8, color=style["ytick.color"])
    ax.set_title(title, fontsize=11, fontweight="bold",
                 color=style["text.color"], pad=10)

    plt.colorbar(im, ax=ax, shrink=0.8)
    _footnote(fig, int(len(df)), style)
    fig.tight_layout()
    return fig_to_bytes(fig)


def make_bullet_chart(
    value: float, target: float, label: str,
    theme_name: str = "Corporate Light",
    good_low: bool = True, vmax: float = None, unit: str = "%",
) -> bytes:
    """
    Executive bullet chart: one measure (value) against a target marker, over
    qualitative bands (good / watch / poor). Far more 'boardroom' than a plain
    bar for a single KPI like 'attrition 16% vs 10% target'.
    good_low=True means lower is better (attrition, churn, cost).
    """
    style  = _get_style(theme_name)
    vmax = vmax or max(value, target) * 1.6
    fig, ax = plt.subplots(figsize=(9, 1.7))
    fig.patch.set_facecolor(style["figure.facecolor"])
    ax.set_facecolor(style["figure.facecolor"])

    # Qualitative bands (green→amber→red for good_low)
    b1, b2 = target, target * 1.5
    bands = [(0, b1, "#D1FAE5"), (b1, b2, "#FEF3C7"), (b2, vmax, "#FEE2E2")]
    if not good_low:
        bands = [(0, b1, "#FEE2E2"), (b1, b2, "#FEF3C7"), (b2, vmax, "#D1FAE5")]
    for lo, hi, c in bands:
        ax.barh([0], hi - lo, left=lo, height=0.5, color=c, edgecolor="none")

    # The measure bar
    mcolor = "#DC2626" if (good_low and value > b2) else \
             "#D97706" if (good_low and value > b1) else "#059669"
    ax.barh([0], value, height=0.22, color=mcolor, edgecolor="none", zorder=3)
    # Target marker
    ax.plot([target, target], [-0.3, 0.3], color="#0A1628", linewidth=2.5, zorder=4)
    ax.text(target, 0.42, "Target {:.0f}{}".format(target, unit),
            ha="center", va="bottom", fontsize=8, color="#0A1628", fontweight="bold")
    ax.text(value, -0.45, "{:.1f}{}".format(value, unit),
            ha="center", va="top", fontsize=11, color=mcolor, fontweight="bold")

    ax.set_xlim(0, vmax)
    ax.set_ylim(-0.7, 0.7)
    ax.set_yticks([])
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(label, fontsize=11, fontweight="bold",
                 color=style["text.color"], loc="left", pad=8)
    fig.tight_layout()
    return fig_to_bytes(fig, dpi=170)


def make_risk_heatmap(
    df: pd.DataFrame, target_col: str, dim_a: str, dim_b: str,
    theme_name: str = "Corporate Light", event_label: str = "event",
) -> bytes:
    """
    Signature 2-way risk grid: event rate (%) for every combination of two
    segments, colour-scaled red = high. This is the Deloitte-style visual that
    makes a risk concentration jump off the page (e.g. attrition by Job Role ×
    Overtime). Returns None if not computable.
    """
    try:
        from app.engines.predictive import _to_binary
    except Exception:
        return None
    y = _to_binary(df[target_col])
    if y is None:
        return None
    style = _get_style(theme_name)
    work = df[[dim_a, dim_b]].astype(str).copy()
    work["_e"] = y.values
    piv = work.pivot_table(index=dim_a, columns=dim_b, values="_e", aggfunc="mean") * 100
    cnt = work.pivot_table(index=dim_a, columns=dim_b, values="_e", aggfunc="count")
    if piv.size == 0 or piv.shape[0] > 12 or piv.shape[1] > 12:
        return None

    fig, ax = plt.subplots(figsize=(max(6, piv.shape[1] * 1.3), max(3.5, piv.shape[0] * 0.7)))
    fig.patch.set_facecolor(style["figure.facecolor"])
    im = ax.imshow(piv.values, cmap="OrRd", aspect="auto",
                   vmin=0, vmax=float(min(100, max(1.0, piv.max().max()))))
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if v == v:  # not NaN
                n = cnt.values[i, j]
                ax.text(j, i, "{:.0f}%\n(n={:.0f})".format(v, n),
                        ha="center", va="center", fontsize=8,
                        color="white" if v > piv.max().max() * 0.55 else "#1A1A1A",
                        fontweight="bold")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_yticks(range(piv.shape[0]))
    ax.set_xticklabels([truncate_label(str(c), 14) for c in piv.columns], rotation=25, ha="right",
                       fontsize=9, color=style["xtick.color"])
    ax.set_yticklabels([truncate_label(str(r), 16) for r in piv.index], fontsize=9,
                       color=style["ytick.color"])
    ax.set_xlabel(_pretty(dim_b), fontsize=9, color=style["axes.labelcolor"])
    ax.set_ylabel(_pretty(dim_a), fontsize=9, color=style["axes.labelcolor"])
    ax.set_title("{} rate by {} × {}".format(
        event_label.title(), _pretty(dim_a), _pretty(dim_b)),
        fontsize=11.5, fontweight="bold", color=style["text.color"], pad=10, loc="left")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("{} rate %".format(event_label), fontsize=8)
    _footnote(fig, int(len(df)), style)
    fig.tight_layout()
    return fig_to_bytes(fig, dpi=170)


def make_driver_importance_chart(
    drivers,                       # list of (name, importance_pct)
    theme_name: str = "Corporate Light",
    title: str = "What Most Predicts the Outcome",
    target_label: str = "",
) -> bytes:
    """
    Signature visual for the predictive section: a clean horizontal bar chart
    of model-derived driver importance. The #1 driver is highlighted so the
    single most important factor reads at a glance — more memorable than a
    table of numbers.
    """
    style  = _get_style(theme_name)
    colors = _get_colors(theme_name)
    drivers = list(drivers)[:8]
    if not drivers:
        return None
    names = [_pretty(str(n)) for n, _ in drivers][::-1]      # top at the top
    vals  = [float(v) for _, v in drivers][::-1]
    top_v = max(vals) if vals else 1.0

    fig, ax = plt.subplots(figsize=(9, max(3.2, len(drivers) * 0.52)))
    fig.patch.set_facecolor(style["figure.facecolor"])
    _apply_style(ax, style)
    ax.grid(False)
    # Highlight the strongest driver (the last one after the reverse = top row)
    bar_colors = [colors[0] if v == top_v else "#94A3B8" for v in vals]
    bars = ax.barh(range(len(vals)), vals, color=bar_colors, alpha=0.9,
                   edgecolor=style["axes.edgecolor"], linewidth=0.4, height=0.66)
    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(v + top_v * 0.015, bar.get_y() + bar.get_height()/2,
                "{:.1f}%".format(v), va="center", ha="left",
                fontsize=9, fontweight="bold", color=style["text.color"])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9.5, color=style["ytick.color"])
    ax.set_xlim(0, top_v * 1.18)
    ax.set_xticks([])
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    sub = (title + (f" — {target_label}" if target_label else ""))
    ax.set_title(sub, fontsize=11.5, fontweight="bold",
                 color=style["text.color"], pad=12, loc="left")
    _footnote(fig, 0, style, note="model-derived importance")
    # Footnote helper prints n=0; overwrite with a cleaner caption
    fig.texts[-1].set_text("Random-forest feature importance | Source: client dataset")
    fig.tight_layout()
    return fig_to_bytes(fig, dpi=170)


def make_box_plot(
    df: pd.DataFrame,
    col: str,
    title: str = "",
    theme_name: str = "Corporate Light",
) -> bytes:
    style  = _get_style(theme_name)
    colors = _get_colors(theme_name)
    data   = df[col].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(style["figure.facecolor"])
    _apply_style(ax, style)

    bp = ax.boxplot(
        data, vert=True, patch_artist=True,
        notch=False, widths=0.4,
        flierprops=dict(marker="o", markerfacecolor=colors[4] if len(colors) > 4 else colors[0],
                        markersize=4, alpha=0.5)
    )
    bp["boxes"][0].set_facecolor(colors[0])
    bp["boxes"][0].set_alpha(0.7)
    bp["medians"][0].set_color(colors[1])
    bp["medians"][0].set_linewidth(2)

    ax.set_xticklabels([col], fontsize=9, color=style["xtick.color"])
    ax.set_title(title or "Outlier Analysis: {}".format(col),
                 fontsize=11, fontweight="bold",
                 color=style["text.color"], pad=10)
    fig.tight_layout()
    return fig_to_bytes(fig)



_ID_METRIC_KW = ("id", "index", "number", "code", "zip", "phone", "guid", "uuid")



def _rank_measures(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """Order numeric columns by how much they look like a business measure.

    Charting the first numeric column in the frame is how a finance report
    ended up with "invoice_id by category", a distribution of invoice_id,
    and a pie chart of summed invoice IDs — three of five charts plotting
    an identifier. Identifiers are dropped outright; the rest are ranked so
    recognised measures (revenue, profit, cost, amount…) lead, then any
    column that actually varies.
    """
    MEASURE_WORDS = (
        "revenue", "sales", "profit", "margin", "cost", "expense", "amount",
        "total", "price", "value", "spend", "budget", "income", "salary",
        "score", "rating", "quantity", "qty", "units", "count", "rate",
        "satisfaction", "tenure", "age", "duration", "balance", "opex",
    )
    scored = []
    for c in cols:
        if is_id_column(c, df[c]):
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty or s.nunique() <= 1:
            continue          # constant column charts as a flat line — useless
        name = str(c).lower()
        score = 0
        if any(w in name for w in MEASURE_WORDS):
            score += 10
        # A column with real spread carries more information than a near-
        # constant one; use CV so it is scale-independent.
        mean = float(s.mean())
        if mean != 0:
            cv = abs(float(s.std()) / mean)
            score += min(cv, 2.0)
        scored.append((score, c))
    scored.sort(key=lambda t: -t[0])
    return [c for _, c in scored]


def _pick_best_metric(num_cols, df=None, cat_cols=None):
    """
    Headline metric = a numeric column with REAL variation, not an ID and not
    a flat Likert score. The old version always returned the first
    'satisfaction/level' column, which on many datasets is a near-constant
    1-5 scale — producing charts where every bar looks identical.
    """
    if not num_cols:
        return None
    if df is None:
        for c in num_cols:
            if any(kw in c.lower() for kw in _SCORE_KEYWORDS):
                return c
        return num_cols[0]

    best, best_score = None, -1.0
    for c in num_cols:
        cl = c.lower()
        if any(k in cl for k in _ID_METRIC_KW):
            continue
        # Value-aware check as well as the name check: a monotonic 1-step
        # integer sequence is an identifier whatever it is called.
        if df is not None and c in df.columns and is_id_column(c, df[c]):
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 10 or s.nunique() < 3:
            continue
        m = float(s.mean())
        cv = float(s.std()) / abs(m) if m else 0.0
        score = min(cv, 2.0)                       # variation is the main signal
        if any(k in cl for k in _SCORE_KEYWORDS):  # mild tiebreak toward KPIs
            score += 0.15
        if score > best_score:
            best, best_score = c, score
    # No `or num_cols[0]` fallback: when every numeric column is an
    # identifier, the honest answer is "nothing to chart". The dataforge
    # original fell back to the first column, which put "Record Id by
    # Label" into a client report.
    return best


# Sensitive / demographic dimensions. Charting an arbitrary metric BY these
# ('Stock Option Level by Marital Status') manufactures a spurious — and
# sometimes discriminatory — comparison. Only chart by them when the metric
# is clearly related (e.g. a pay-equity review), never as the auto-headline.
_SENSITIVE_DIMS = ("marital", "gender", "sex", "race", "ethnic", "religion",
                   "nationality", "disab", "age", "orientation")


def _best_metric_by_category(df, num_cols, cat_cols, exclude_pairs=None,
                             allow_sensitive=False):
    """
    Pick the (categorical, numeric) pair where the metric's group means vary
    the MOST (relative spread) — the chart that actually shows a finding.
    Skips sensitive/demographic dimensions so we never auto-generate a
    spurious 'metric by marital status' chart. Returns (cat, metric, spread).
    """
    exclude_pairs = exclude_pairs or set()
    best = None
    for cat in cat_cols:
        if not allow_sensitive and any(k in cat.lower() for k in _SENSITIVE_DIMS):
            continue
        nun = df[cat].nunique(dropna=True)
        if not (2 <= nun <= 25) or not _is_grouping_column(df, cat):
            continue
        for metric in num_cols:
            if (cat, metric) in exclude_pairs or \
                    any(k in metric.lower() for k in _ID_METRIC_KW) or \
                    is_id_column(metric, df[metric]):
                continue
            try:
                # Rank pairs by the SAME aggregation the chart will render
                # (mean for scores, sum for additive metrics) so the category
                # a chart highlights matches what the narrative reports.
                agg_func, _ = _agg_for_metric(metric)
                g = df.groupby(cat)[metric].agg(agg_func).dropna()
                if len(g) < 2:
                    continue
                lo, hi = float(g.min()), float(g.max())
                spread = (hi - lo) / abs(hi) if hi else 0.0
                if best is None or spread > best[2]:
                    best = (cat, metric, spread)
            except Exception:
                logger.debug("pair spread failed %s x %s", cat, metric, exc_info=True)
    return best


def generate_all_charts(
    df: pd.DataFrame,
    theme_name: str = "Corporate Light",
    max_charts: int = 5,
) -> List[Tuple[str, bytes]]:
    """Auto-generate best charts for this dataset.

    FIX: previously always used num_cols[0] for every chart and a numeric-bin
    line chart with unreadable x-axis labels when no datetime column existed.
    Now picks the best score metric and falls back to a second categorical
    breakdown or ranked bar chart instead of meaningless numeric bins.
    """
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include=["object", "string"]).columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()
    charts    = []

    best_metric = _pick_best_metric(num_cols, df=df, cat_cols=cat_cols)
    if best_metric is None:
        # Every numeric column is an identifier. Returning nothing beats
        # filling a client report with charts of row numbers.
        logger.info("no non-identifier numeric column to chart among %d",
                    len(num_cols))
        return []

    # Chart 1: the (category, metric) pair that VARIES most — the chart that
    # actually shows a finding, instead of flat bars on a near-constant score.
    pair1 = _best_metric_by_category(df, num_cols, cat_cols) if (num_cols and cat_cols) else None
    if pair1:
        best_cat, chart1_metric, _ = pair1
        title = "{} by {}".format(_pretty(chart1_metric), _pretty(best_cat))
        try:
            charts.append((title, make_bar_chart(df, best_cat, chart1_metric, title, theme_name)))
        except Exception:
            logger.warning("chart 1 failed", exc_info=True)
    else:
        best_cat = next((c for c in cat_cols
                         if _is_grouping_column(df, c)
                         and not any(k in c.lower() for k in _SENSITIVE_DIMS)),
                        None) if cat_cols else None
        if best_cat is None and cat_cols:
            # Only fall back to a qualifying column — never force cat_cols[0]
            # regardless of cardinality. A column like a review_id/order_id
            # with tens of thousands of unique values (or, on a small
            # dataset, an ID column whose cardinality coincidentally looks
            # 'reasonable') is not a usable chart dimension just because it
            # happened to be first; forcing it produces charts with as many
            # bars as there are rows.
            best_cat = next((c for c in cat_cols if _is_grouping_column(df, c)), None)
        chart1_metric = best_metric
        if best_cat and best_metric:
            title = "{} by {}".format(_pretty(best_metric), _pretty(best_cat))
            try:
                charts.append((title, make_bar_chart(df, best_cat, best_metric, title, theme_name)))
            except Exception:
                logger.warning("chart 1 fallback failed", exc_info=True)
        # If no categorical dimension qualifies, Chart 1 is simply skipped —
        # Chart 3 below already shows Distribution: {best_metric}, so a
        # histogram fallback here would just be a duplicate of that chart.

    # 2. Second view — datetime trend OR the next most-varying (cat, metric)
    #    pair, never repeating chart 1's pair.
    if date_cols and best_metric:
        title = "{} Over Time".format(_pretty(best_metric))
        try:
            charts.append((title, make_line_chart(
                df, date_cols[0], best_metric, title, theme_name
            )))
        except Exception:
            logger.warning("chart 2 (trend) failed", exc_info=True)
    else:
        pair2 = (_best_metric_by_category(
            df, num_cols, cat_cols,
            exclude_pairs={(pair1[0], pair1[1])} if pair1 else set())
            if (num_cols and cat_cols) else None)
        if pair2 and pair2[2] > 0.03:
            c2, m2, _ = pair2
            title = "{} by {}".format(_pretty(m2), _pretty(c2))
            try:
                charts.append((title, make_bar_chart(df, c2, m2, title, theme_name)))
            except Exception:
                logger.warning("chart 2 (pair) failed", exc_info=True)
        elif len(num_cols) >= 2:
            second_metric = next((c for c in num_cols if c != best_metric), num_cols[0])
            title = "Distribution: {}".format(_pretty(second_metric))
            try:
                charts.append((title, make_histogram(df, second_metric, title, theme_name)))
            except Exception:
                logger.warning("chart 2 (hist) failed", exc_info=True)

    # 3. Histogram — distribution of primary metric
    if best_metric:
        title = "Distribution: {}".format(_pretty(best_metric))
        try:
            charts.append((title, make_histogram(
                df, best_metric, title, theme_name
            )))
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    # 4. Correlation heatmap
    if len(num_cols) >= 3:
        try:
            charts.append(("Correlation Matrix", make_correlation_heatmap(
                df, "Correlation Matrix", theme_name
            )))
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    # 5. Ranked bar of a DIFFERENT metric than charts 1-3 used. Repeating the
    #    same variable a 4th time is what makes a report feel padded/AI-made.
    #    Pick the second metric with the most spread (highest CV), ranked by
    #    the category where it varies most.
    alt_metric = _pick_second_metric(df, num_cols, exclude={best_metric})
    if alt_metric and best_cat:
        rank_cat = _pick_high_spread_dimension(df, alt_metric, exclude=set()) or best_cat
        title = "{} Ranking by {}".format(_pretty(alt_metric), _pretty(rank_cat))
        try:
            charts.append((title, make_ranked_bar_chart(
                df, rank_cat, alt_metric, title, theme_name
            )))
        except Exception:
            logger.warning("ranked chart 5 failed", exc_info=True)
    elif best_cat and best_metric:
        # Only one numeric metric exists — a ranked view of it is still useful
        title = "{} Ranking by {}".format(_pretty(best_metric), _pretty(best_cat))
        try:
            charts.append((title, make_ranked_bar_chart(
                df, best_cat, best_metric, title, theme_name
            )))
        except Exception:
            logger.warning("ranked chart 5 fallback failed", exc_info=True)

    return charts[:max_charts]


def _pick_second_metric(df, num_cols, exclude=None):
    """Numeric column (not in exclude, not an ID) with the highest coefficient
    of variation — the metric that varies most, hence most worth a second chart."""
    exclude = exclude or set()
    best, best_cv = None, 0.0
    for c in num_cols:
        if c in exclude or any(k in c.lower() for k in ("id", "number", "code")):
            continue
        try:
            s = df[c].dropna()
            if len(s) < 20 or s.nunique() < 5:
                continue
            m = float(s.mean())
            cv = float(s.std()) / abs(m) if m else 0.0
            if cv > best_cv:
                best, best_cv = c, cv
        except Exception:
            logger.debug("cv check failed for %s", c, exc_info=True)
    return best


def _pick_high_spread_dimension(df, metric, exclude=None):
    """Categorical column (2-15 groups) whose group-means of `metric` show
    the LARGEST relative spread — where the metric actually varies."""
    exclude = exclude or set()
    best, best_spread = None, 0.0
    for c in df.select_dtypes(include=["object", "string"]).columns:
        if c in exclude or not _is_grouping_column(df, c, max_unique=15):
            continue
        if any(k in c.lower() for k in _SENSITIVE_DIMS):
            continue
        try:
            g = df.groupby(c)[metric].mean().dropna()
            if len(g) < 2:
                continue
            lo, hi = float(g.min()), float(g.max())
            spread = (hi - lo) / abs(hi) if hi else 0.0
            if spread > best_spread:
                best, best_spread = c, spread
        except Exception:
            logger.debug("spread check failed for %s", c, exc_info=True)
    return best if best_spread > 0.05 else None
