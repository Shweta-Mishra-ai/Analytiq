import logging
import io
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

from app.engines.chart_message import human_number as _human_num
from app.engines.domains.base import is_id_column
from app.services.dtypes import MONTH_END

logger = logging.getLogger(__name__)


def _theme(theme_name: str) -> dict:
    """The report theme these charts are being printed into.

    The charts used to carry their own three palettes, keyed on theme
    names that mostly did not exist. `"Executive Green"` is not a theme —
    the green one is called `"Sales Green"` — so the green palette was
    dead code and every light report got the same `#1a4a8a` blue. An
    e-commerce report therefore had an orange cover, orange headings,
    orange section badges and blue charts; the HR report drew `#1976D2`
    headings above `#1a4a8a` bars, two blues close enough to look like a
    mistake rather than a choice.

    Taking the palette from the theme itself is what makes a document
    look designed instead of assembled.
    """
    from app.engines.pdf_builder import THEMES

    return THEMES.get(theme_name) or THEMES["Corporate Light"]


def _get_style(theme_name: str) -> dict:
    t = _theme(theme_name)
    return {
        # The figure is drawn onto the page, so its background has to be
        # the page's — otherwise every chart shows as a pale rectangle on
        # a dark theme, or a grey one on white.
        "figure.facecolor": t["page_bg"],
        "axes.facecolor":   t["bg_card"],
        "axes.edgecolor":   t["border"],
        "axes.labelcolor":  t["text"],
        "xtick.color":      t["text_muted"],
        "ytick.color":      t["text_muted"],
        "text.color":       t["text"],
        "grid.color":       t["border"],
        "grid.alpha":       0.8,
    }


def _get_colors(theme_name: str) -> list:
    """Series colours, led by the theme's own accent.

    Ordered so a single-series chart — which is most of them — draws in
    exactly the colour the headings and section rules use.
    """
    t = _theme(theme_name)
    return [t["accent"], t["accent2"], t["info"], t["positive"],
            t["warning"], t["negative"]]


def _apply_style(ax, style: dict, axis: str = "both"):
    """Grid behind the data, and only on the axis that carries a scale.

    Two defaults made every chart look homemade. `ax.grid(True)` draws on
    top of whatever is plotted, so gridlines ran across the face of each
    bar; and gridding "both" axes puts vertical lines between categories,
    which measure nothing — a bar sitting on `region` has no x scale to
    read against.
    """
    ax.set_facecolor(style["axes.facecolor"])
    ax.tick_params(colors=style["xtick.color"])
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color=style["grid.color"],
            alpha=style["grid.alpha"], linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(style["axes.edgecolor"])
    ax.spines["bottom"].set_color(style["axes.edgecolor"])


def fig_to_bytes(fig: plt.Figure, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    data = buf.read()
    plt.close(fig)
    return data


def _human_axis(ax, axis: str = "y") -> None:
    """Ticks a reader can read, and no "1e6" in the corner.

    matplotlib's offset notation puts a small "1e6" at the top-left of
    the axes, where it collides with anything else drawn there — the
    chart subtitle landed on top of it. Formatting the ticks in human
    units removes the offset entirely and is what a reader expects to
    see on a money axis anyway.
    """
    from matplotlib.ticker import FuncFormatter

    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(FuncFormatter(lambda v, _p: _human_num(v)))
    try:
        target.get_offset_text().set_visible(False)
    except Exception:
        logger.debug("could not hide the axis offset text", exc_info=True)


def _headline(ax, style, message, label: str) -> None:
    """Message as the headline, variable names as the subtitle.

    The consulting convention, and the reason a deck reads faster than a
    dashboard: the reader takes the finding from the title and uses the
    axes only to check it. "revenue by region" says what is plotted and
    nothing about what to take from it.
    """
    if message:
        ax.set_title(message, fontsize=10.5, fontweight="bold",
                     color=style["text.color"], pad=18, loc="left",
                     wrap=True)
        ax.text(0, 1.015, label, transform=ax.transAxes, fontsize=8,
                color=style["axes.labelcolor"], va="bottom", zorder=5)
    else:
        ax.set_title(label, fontsize=11, fontweight="bold",
                     color=style["text.color"], pad=10, loc="left")


def make_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    theme_name: str = "Corporate Light",
    top_n: int = 15,
) -> bytes:
    style  = _get_style(theme_name)
    colors = _get_colors(theme_name)

    agg = (df.groupby(x_col)[y_col]
             .sum()
             .reset_index()
             .sort_values(y_col, ascending=False)
             .head(top_n))

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(style["figure.facecolor"])
    _apply_style(ax, style, axis="y")

    bars = ax.bar(
        range(len(agg)), agg[y_col],
        color=colors[0], alpha=0.85,
        edgecolor=style["axes.edgecolor"], linewidth=0.5
    )

    # Labels in the same units as the axis. "3,242,612" printed above a
    # bar on an axis reading "3.0m" makes the reader do the conversion
    # themselves to check the two agree.
    tallest = max((b.get_height() for b in bars), default=0.0)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + tallest * 0.015,
            _human_num(h),
            ha="center", va="bottom",
            fontsize=7.5, color=style["text.color"]
        )
    if tallest > 0:
        # Otherwise the label on the tallest bar is clipped by the frame.
        ax.set_ylim(top=tallest * 1.12)

    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(
        [str(v)[:12] for v in agg[x_col]],
        rotation=35, ha="right", fontsize=8
    )
    ax.set_ylabel(y_col, fontsize=9, color=style["axes.labelcolor"])
    from app.engines.chart_message import bar_message
    _human_axis(ax)
    _headline(ax, style, bar_message(df, x_col, y_col),
              title or "{} by {}".format(y_col, x_col))
    fig.tight_layout()
    return fig_to_bytes(fig)


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
        labels = [str(b)[:10] for b in agg["_bin"]]

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
        labels = [str(v)[:12] for v in agg[x_col]]

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
    from app.engines.chart_message import line_message
    _human_axis(ax)
    _headline(ax, style, line_message(df, x_col, y_col),
              title or "{} Trend".format(y_col))
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
               label="Mean: {}".format(_human_num(mean_val)))
    ax.axvline(median_val, color=colors[3] if len(colors) > 3 else colors[1],
               linestyle=":", linewidth=1.8,
               label="Median: {}".format(_human_num(median_val)))
    # The default legend box is transparent, so it sat over the bars with
    # the bar colour showing through the text.
    ax.legend(fontsize=8, framealpha=0.92,
              facecolor=style["figure.facecolor"],
              edgecolor=style["axes.edgecolor"],
              labelcolor=style["text.color"])

    ax.set_xlabel(col, fontsize=9, color=style["axes.labelcolor"])
    ax.set_ylabel("Frequency", fontsize=9, color=style["axes.labelcolor"])
    from app.engines.chart_message import histogram_message
    _human_axis(ax)
    _headline(ax, style, histogram_message(df, col),
              title or "Distribution: {}".format(col))
    fig.tight_layout()
    return fig_to_bytes(fig)


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
        [str(v)[:15] for v in agg[names_col]],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=9,
        framealpha=0,
        labelcolor=style["text.color"]
    )
    from app.engines.chart_message import pie_message
    _headline(ax, style, pie_message(df, names_col, values_col),
              title or "{} by {}".format(values_col, names_col))
    fig.tight_layout()
    return fig_to_bytes(fig)


def make_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Matrix",
    theme_name: str = "Corporate Light",
) -> bytes:
    style = _get_style(theme_name)
    # Identifiers belong nowhere near a correlation matrix. On a sales
    # export ordered by date, `order_id` correlated 0.22 with revenue —
    # an artefact of the row order, printed in the same grid and the same
    # colour as the real relationships, with nothing marking it as
    # meaningless.
    num_cols = _rank_measures(df, df.select_dtypes(include="number")
                              .columns.tolist())

    if len(num_cols) < 2:
        return None

    cols = sorted(num_cols[:10])
    corr = df[cols].corr().round(2)
    n    = len(corr)

    fig, ax = plt.subplots(figsize=(max(7, n), max(5, n - 1)))
    fig.patch.set_facecolor(style["figure.facecolor"])
    ax.set_facecolor(style["axes.facecolor"])

    # The diagonal is 1.00 by definition. Left in, it takes the strongest
    # colour on the scale, so the eye lands on the one thing on the chart
    # that carries no information.
    matrix = corr.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(matrix, np.nan)
    cmap = matplotlib.colormaps["RdBu_r"].with_extremes(
        bad=style["axes.facecolor"])
    im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = matrix[i, j]
            # The cell colour comes from the colormap, not the theme, so
            # the label has to be read against the cell. Using the theme's
            # text colour drew near-white figures on the pale middle of
            # the scale — every weak correlation was invisible on the dark
            # theme, which is where most of the cells sit.
            r, g, b, _a = cmap((val + 1) / 2)
            dark_cell = (0.299 * r + 0.587 * g + 0.114 * b) < 0.55
            ax.text(j, i, "{:.2f}".format(val),
                    ha="center", va="center", fontsize=8,
                    color="#ffffff" if dark_cell else "#101018")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([c[:10] for c in corr.columns],
                       rotation=45, ha="right", fontsize=8,
                       color=style["xtick.color"])
    ax.set_yticklabels([c[:10] for c in corr.index],
                       fontsize=8, color=style["ytick.color"])
    from app.engines.chart_message import heatmap_message
    _headline(ax, style, heatmap_message(df, cols),
              title or "Correlation Matrix")

    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig_to_bytes(fig)


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


def _rank_measures(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """Order numeric columns by how much they look like a business measure.

    Charting the first numeric column in the frame is how a finance report
    ended up with "invoice_id by category", a distribution of invoice_id,
    and a pie chart of summed invoice IDs — three of five charts plotting
    an identifier. Identifiers are dropped outright; the rest are ranked so
    recognised measures (revenue, profit, cost, amount…) lead, then any
    column that actually varies.
    """
    # Ranked in tiers, because "is it a measure?" is not the question a
    # reader asks — they ask which measure the business is about. Money
    # first, then volume, then rates and scores.
    #
    # Scoring every measure word equally and breaking ties on coefficient
    # of variation ranked `units` above `revenue` on a sales file, purely
    # because unit counts are noisier than money. All five charts were
    # then about units and revenue never appeared.
    MONEY_WORDS = (
        "revenue", "sales", "profit", "margin", "gmv", "turnover", "income",
        "cost", "expense", "spend", "amount", "price", "value", "budget",
        "salary", "opex", "capex", "balance", "fee", "charge",
    )
    VOLUME_WORDS = (
        "quantity", "qty", "units", "count", "orders", "visits", "sessions",
        "headcount", "volume", "transactions", "clicks", "impressions",
    )
    RATE_WORDS = (
        "rate", "score", "rating", "satisfaction", "percent", "pct", "ratio",
        "share", "index", "tenure", "age", "duration", "days", "hours",
    )
    scored = []
    for c in cols:
        if is_id_column(c, df[c]):
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty or s.nunique() <= 1:
            continue          # constant column charts as a flat line — useless
        name = str(c).lower()
        # A rate marker demotes whatever else the name says: "margin_pct"
        # is a percentage, not money, and charting it as the headline
        # measure puts a ratio where the reader expects a total.
        is_rate = any(m in name for m in
                      ("_pct", "pct_", "percent", "_rate", "rate_", "ratio",
                       "_share", "share_", "_index", "per_"))
        if is_rate:
            score = 100
        elif any(w in name for w in MONEY_WORDS):
            score = 300
        elif any(w in name for w in VOLUME_WORDS):
            score = 200
        elif any(w in name for w in RATE_WORDS):
            score = 100
        else:
            score = 0
        # Spread separates columns within a tier only. A near-constant
        # column carries less information than a varying one, but no
        # amount of variance makes a unit count more important than
        # revenue.
        mean = float(s.mean())
        if mean != 0:
            score += min(abs(float(s.std()) / mean), 2.0)
        scored.append((score, c))
    scored.sort(key=lambda t: -t[0])
    return [c for _, c in scored]


def generate_all_charts(
    df: pd.DataFrame,
    theme_name: str = "Corporate Light",
    max_charts: int = 5,
) -> List[Tuple[str, bytes]]:
    """Auto-generate best charts for this dataset."""
    # include="object" alone misses pandas 3's `str` dtype in some paths;
    # ask for both so categorical columns are found on either version.
    raw_num = df.select_dtypes(include="number").columns.tolist()
    raw_cat = df.select_dtypes(include=["object", "string"]).columns.tolist()

    num_cols  = _rank_measures(df, raw_num)
    cat_cols  = [c for c in raw_cat if not is_id_column(c, df[c])
                 and 2 <= df[c].nunique() <= 25]
    date_cols = df.select_dtypes(include="datetime").columns.tolist()

    if not num_cols:
        # Nothing chartable that isn't an identifier — better to return no
        # charts than to fill the report with meaningless ones.
        logger.info("no non-identifier numeric columns to chart")
        return []

    charts    = []

    # 1. Bar chart — categorical x numeric
    if cat_cols and num_cols:
        best_cat = next(
            (c for c in cat_cols if 2 <= df[c].nunique() <= 25),
            cat_cols[0]
        )
        title = "{} by {}".format(num_cols[0], best_cat)
        try:
            charts.append((title, make_bar_chart(
                df, best_cat, num_cols[0], title, theme_name
            )))
        except Exception:
            logger.debug("generate_all_charts: suppressed exception", exc_info=True)

    # 2. Line chart — trend over time or numeric
    if date_cols and num_cols:
        title = "{} Over Time".format(num_cols[0])
        try:
            charts.append((title, make_line_chart(
                df, date_cols[0], num_cols[0], title, theme_name
            )))
        except Exception:
            logger.debug("generate_all_charts: suppressed exception", exc_info=True)
    elif len(num_cols) >= 2:
        title = "{} Trend".format(num_cols[1])
        try:
            charts.append((title, make_line_chart(
                df, num_cols[0], num_cols[1], title, theme_name
            )))
        except Exception:
            logger.debug("generate_all_charts: suppressed exception", exc_info=True)

    # 3. Histogram — distribution of the SECOND measure where there is
    # one. Four views of a single column is not a chart pack: on a sales
    # file every chart was about `units` and revenue never appeared.
    dist_col = num_cols[1] if len(num_cols) > 1 else num_cols[0]
    if num_cols:
        title = "Distribution: {}".format(dist_col)
        try:
            charts.append((title, make_histogram(
                df, dist_col, title, theme_name
            )))
        except Exception:
            logger.debug("generate_all_charts: suppressed exception", exc_info=True)

    # 4. Correlation heatmap
    if len(num_cols) >= 3:
        try:
            charts.append(("Correlation Matrix", make_correlation_heatmap(
                df, "Correlation Matrix", theme_name
            )))
        except Exception:
            logger.debug("generate_all_charts: suppressed exception", exc_info=True)

    # 5. Pie chart — category share
    if cat_cols and num_cols:
        best_cat = next(
            (c for c in cat_cols if 2 <= df[c].nunique() <= 10),
            None
        )
        if best_cat:
            share_col = (num_cols[1] if len(num_cols) > 1
                         and best_cat == cat_cols[0] else num_cols[0])
            title = "{} Share by {}".format(share_col, best_cat)
            try:
                charts.append((title, make_pie_chart(
                    df, best_cat, share_col, title, theme_name
                )))
            except Exception:
                logger.debug("generate_all_charts: suppressed exception", exc_info=True)

    return charts[:max_charts]
