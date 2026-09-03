"""
engines/charts/plots.py — drawing one chart.

Each function here turns a frame and a pair of columns into PNG bytes.
They share the palette and axis rules from style.py and know nothing
about which chart a report should contain — that is selection.py's job.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")                      # no display; PNG bytes only
import matplotlib.pyplot as plt            # noqa: E402
import pandas as pd                        # noqa: E402


logger = logging.getLogger(__name__)



from app.engines.pdf_primitives import truncate_label
from app.services.dtypes import MONTH_END
from app.engines.charts.style import (
    _agg_for_metric, _apply_style, _axis_label, _footnote, _gap_headline,
    _get_colors, _get_style, _pretty, _reference_line, _tick_budget,
    fig_to_bytes,
)


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
    # The axis is named after the aggregation actually performed. It was
    # named from is_score, which is a different question, so a chart of
    # mean monthly income carried an axis reading "Total Monthly Income".
    y_label  = _axis_label(y_col, agg_func)

    agg = (df.groupby(x_col)[y_col]
             .agg(agg_func)
             .reset_index()
             .sort_values(y_col, ascending=False)
             .head(top_n))

    org_avg, avg_label = _reference_line(df, y_col, agg, agg_func)
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
               label="{}: {}".format(avg_label, fmt.format(org_avg)))
    ax.legend(fontsize=8, framealpha=0)

    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(
        [truncate_label(str(v), _tick_budget(len(agg), 14))
         for v in agg[x_col]],
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
        labels = [truncate_label(str(v), _tick_budget(len(agg), 12))
                  for v in agg[x_col]]

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

    org_avg, avg_label = _reference_line(df, y_col, agg, agg_func)
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
               label="{}: {}".format(avg_label, fmt.format(org_avg)))
    ax.legend(fontsize=8, framealpha=0)

    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels(
        [truncate_label(str(v), _tick_budget(len(agg), 16))
         for v in agg[x_col]], fontsize=9)
    ax.set_xlabel(_axis_label(y_col, agg_func),
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


