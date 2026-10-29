"""
engines/chart_exporter.py — the charts printed in the PDF reports, each
with a caption computed from the very numbers it plots.

What this replaced, and why:
- Every chart summed its measure by group, so a report showed "tenure by
  department" as 1,300 years and the share of total salary as a pie. A
  measure is totalled only when a total means something (revenue, cost,
  units); otherwise it is averaged, and the axis says which.
- The text beside each chart was written separately, by guessing the
  columns back from the title, and fell back to the first numeric column —
  usually an identifier. "tenure_years over time" was described as
  "Employee Id across 4 Department groups". The caption is now built from
  the aggregated data the chart draws.
- Charts drew in their own blues whatever the report's colours, carried a
  second title inside the image, and included identifiers in the
  correlation matrix.
- A chart now appears only when it shows something: a trend needs an event
  date and four full months; a correlation matrix needs a correlation.
"""
from __future__ import annotations

import io
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.engines.domains.base import is_id_column  # noqa: E402
from app.services.dtypes import MONTH_END, text_columns  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class ReportChart:
    title: str
    png: bytes
    caption: str
    kind: str
    columns: Tuple[str, ...] = field(default_factory=tuple)


# ── vocabulary ────────────────────────────────────────────

# A total of these means something; of anything else it does not.
_ADDITIVE = {"revenue", "sales", "amount", "cost", "costs", "expense", "expenses",
             "spend", "profit", "income", "units", "quantity", "qty", "orders",
             "volume", "budget", "opex", "capex", "payment", "payments", "fee",
             "fees", "tax", "sold", "gmv", "turnover", "bookings", "hours"}
_MONEY = {"revenue", "sales", "amount", "cost", "expense", "spend", "profit",
          "income", "budget", "gmv", "turnover", "bookings", "salary", "price"}
# Dates that mark when something began, not when an event happened: a
# trend over them plots a cohort artefact (tenure falls with hire date).
_NOT_EVENT_DATE = {"hire", "hired", "birth", "dob", "born", "join", "joined",
                   "joining", "start", "started", "founded", "opened", "since"}
_GROUP_WORDS = {"region", "department", "dept", "category", "segment", "channel",
                "team", "product", "type", "plan", "tier", "store", "branch",
                "country", "city", "division", "unit", "group", "source"}


def _words(col) -> set:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(col))
    return {w for w in re.split(r"[^a-z0-9]+", spaced.lower()) if w}


def _label(col) -> str:
    """'tenure_years' → 'Tenure years' for axis labels and captions."""
    s = re.sub(r"[_\s]+", " ", str(col)).strip()
    return s[:1].upper() + s[1:]


def is_additive(col) -> bool:
    return bool(_words(col) & _ADDITIVE)


def fmt(x: float, money: bool = False) -> str:
    """A figure as a reader writes it: 1,250 · 3.4M · 0.42."""
    if x is None or not np.isfinite(x):
        return "—"
    a = abs(x)
    if a >= 1e9:
        s = "{:.1f}B".format(x / 1e9)
    elif a >= 1e6:
        s = "{:.1f}M".format(x / 1e6)
    elif a >= 1e4:
        s = "{:,.0f}".format(x)
    elif a >= 100:
        s = "{:,.0f}".format(x)
    elif a >= 1:
        s = "{:,.2f}".format(x).rstrip("0").rstrip(".")
    else:
        s = "{:.2g}".format(x)
    return s


def _pct(x: float) -> str:
    return "{:.0f}%".format(x) if abs(x) >= 10 else "{:.1f}%".format(x)


# ── drawing ───────────────────────────────────────────────

def _font_family() -> Optional[str]:
    from matplotlib import font_manager as fm
    from app.engines.pdf_fonts import FONT_DIR
    try:
        for f in ("Carlito-Regular.ttf", "Carlito-Bold.ttf"):
            p = os.path.join(FONT_DIR, f)
            if os.path.exists(p):
                fm.fontManager.addfont(p)
        return "Carlito"
    except Exception:
        logger.debug("chart font unavailable", exc_info=True)
        return None


_FAMILY = _font_family()


def _rc(T: dict) -> dict:
    rc = {
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": T["border"], "axes.labelcolor": T["text_muted"],
        "xtick.color": T["text_muted"], "ytick.color": T["text_muted"],
        "text.color": T["text"], "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
    }
    if _FAMILY:
        rc["font.family"] = _FAMILY
    return rc


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return buf.getvalue()


def _value_axis(ax, T, horizontal: bool):
    from matplotlib.ticker import FuncFormatter
    form = FuncFormatter(lambda v, _: fmt(v))
    (ax.xaxis if horizontal else ax.yaxis).set_major_formatter(form)
    ax.grid(axis="x" if horizontal else "y", color=T["border"], linewidth=0.6)
    ax.set_axisbelow(True)


def _bars(T, labels, values, value_label, ref=None, ref_label="",
          percent=False) -> bytes:
    """Horizontal bars, largest at the top: readable for any label length."""
    n = len(values)
    with plt.rc_context(_rc(T)):
        fig, ax = plt.subplots(figsize=(7.6, max(1.6, 0.36 * n + 0.7)))
        y = np.arange(n)[::-1]
        ax.barh(y, values, color=T["accent"], height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels([str(l)[:28] for l in labels])
        top = max(values) if len(values) else 0
        for yi, v in zip(y, values):
            ax.text(v + top * 0.01, yi, (_pct(v) if percent else fmt(v)),
                    va="center", fontsize=8.5, color=T["text"])
        if ref is not None:
            ax.axvline(ref, color=T["text_muted"], linestyle="--", linewidth=0.9)
            ax.text(ref, n - 0.35, " " + ref_label, fontsize=8, color=T["text_muted"],
                    va="bottom")
        ax.set_xlim(0, top * 1.15 if top > 0 else 1)
        ax.set_xlabel(value_label)
        _value_axis(ax, T, horizontal=True)
        if percent:
            from matplotlib.ticker import FuncFormatter
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "{:.0f}%".format(v)))
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        return _png(fig)


# ── choosing what to chart ────────────────────────────────

def _measures(df: pd.DataFrame) -> List[str]:
    """Measure columns, most useful first: money and recognised business
    measures, then anything with a known good direction, then the rest."""
    from app.engines.benchmarking import metric_direction
    scored = []
    for c in df.select_dtypes(include="number").columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s) or is_id_column(c, s):
            continue
        s = s.dropna()
        if len(s) < 5 or s.nunique() <= 2:
            continue      # constant, or a 0/1 flag — a rate, not a measure
        w = _words(c)
        score = 0.0
        if w & _MONEY:
            score += 20
        if w & _ADDITIVE:
            score += 10
        if metric_direction(c, df[c]):
            score += 5
        mean = float(s.mean())
        if mean:
            score += min(abs(float(s.std()) / mean), 2.0)
        scored.append((score, c))
    scored.sort(key=lambda t: -t[0])
    return [c for _, c in scored]


# Kept for callers that rank measures directly.
def _rank_measures(df: pd.DataFrame, cols: List[str]) -> List[str]:
    keep = set(cols)
    return [c for c in _measures(df) if c in keep]


def _group_column(df: pd.DataFrame, exclude=()) -> Optional[str]:
    best = None
    for c in text_columns(df):
        if c in exclude or is_id_column(c, df[c]):
            continue
        k = df[c].nunique(dropna=True)
        if not 2 <= k <= 30:
            continue
        vc = df[c].value_counts()
        if vc.iloc[-1] < 5 and k > 12:
            continue
        score = (10 if _words(c) & _GROUP_WORDS else 0) + (5 if 3 <= k <= 12 else 0) \
            - (3 if k == 2 else 0)
        if best is None or score > best[0]:
            best = (score, c)
    return best[1] if best else None


def _event_date(df: pd.DataFrame) -> Optional[str]:
    for c in df.select_dtypes(include="datetime").columns:
        if _words(c) & _NOT_EVENT_DATE:
            continue
        if df[c].notna().sum() >= 20:
            return c
    return None


def _outcome(df: pd.DataFrame) -> Optional[Tuple[str, pd.Series, str]]:
    """A named two-value outcome (attrition, churn, returned…), as 0/1."""
    from app.engines.predictive import _BINARY_TARGET_NAMES, _EVENT_WORDS, _to_binary
    for c in df.columns:
        s = df[c].dropna()
        if s.nunique() != 2 or len(s) < 30:
            continue
        values = {str(v).lower().strip() for v in s.unique()}
        named = any(k in str(c).lower() for k in _BINARY_TARGET_NAMES + ("returned", "converted"))
        if not named and not (values & (_EVENT_WORDS - {"1", "1.0", "y", "t"})):
            continue
        coded = _to_binary(df[c])
        if coded is None:
            continue
        y, event = coded
        y = y.where(df[c].notna())
        if 0.02 <= y.mean() <= 0.98:
            return c, y, event
    return None


# ── the charts ────────────────────────────────────────────

def _by_group(df, T, metric, group) -> Optional[ReportChart]:
    from scipy import stats as st
    additive = is_additive(metric)
    data = df[[group, metric]].dropna()
    counts = data[group].value_counts()
    top = counts.index[:12]
    data = data[data[group].isin(top)]
    agg = data.groupby(group)[metric].agg("sum" if additive else "mean")
    agg = agg.sort_values(ascending=False)
    if len(agg) < 2:
        return None
    what = ("Total" if additive else "Average") + " " + _label(metric).lower()
    png = _bars(T, list(agg.index), agg.values, what)

    hi, lo = agg.index[0], agg.index[-1]
    gap = (agg.iloc[0] - agg.iloc[-1]) / abs(agg.iloc[-1]) * 100 if agg.iloc[-1] else float("nan")
    parts = []
    if additive:
        share = agg.iloc[0] / agg.sum() * 100 if agg.sum() else 0
        parts.append("{} has the largest total {} ({}, {} of the total shown); {} the "
                     "smallest ({}).".format(hi, _label(metric).lower(), fmt(agg.iloc[0]),
                                             _pct(share), lo, fmt(agg.iloc[-1])))
    else:
        parts.append("Average {} is highest in {} ({}) and lowest in {} ({}){}.".format(
            _label(metric).lower(), hi, fmt(agg.iloc[0]), lo, fmt(agg.iloc[-1]),
            ", a {} difference".format(_pct(gap)) if np.isfinite(gap) else ""))
    groups = [g[metric].values for _, g in data.groupby(group) if len(g) >= 3]
    if len(groups) >= 2:
        try:
            p = float(st.kruskal(*groups).pvalue)
            if additive:
                # A total also grows with the number of records, so say
                # which of the two is behind the ranking.
                parts.append(
                    "Individual {} values differ between groups by more than chance "
                    "(Kruskal-Wallis p={:.3f}).".format(_label(metric).lower(), p) if p < 0.05
                    else "Individual {} values do not differ between groups by more than "
                         "chance (Kruskal-Wallis p={:.2f}); the totals differ mainly because "
                         "the groups have different numbers of records.".format(
                             _label(metric).lower(), p))
            else:
                parts.append(
                    "The groups differ by more than chance would explain (Kruskal-Wallis "
                    "p={:.3f}).".format(p) if p < 0.05 else
                    "Differences this size are within what chance alone produces "
                    "(Kruskal-Wallis p={:.2f}), so the ranking should not be relied on."
                    .format(p))
        except Exception:
            logger.debug("kruskal failed", exc_info=True)
    n_lo, n_hi = int(counts[top].min()), int(counts[top].max())
    parts.append("Each group has {}–{} records{}.".format(
        fmt(n_lo), fmt(n_hi),
        "; the 12 largest of {} groups are shown".format(len(counts)) if len(counts) > 12 else ""))
    return ReportChart(title="{} by {}".format(what, _label(group).lower()), png=png,
                       caption=" ".join(parts), kind="by_group", columns=(metric, group))


def _rate_by_group(df, T, outcome, group) -> Optional[ReportChart]:
    from app.services.stat_guards import chi2_association
    col, y, event = outcome
    data = pd.DataFrame({"g": df[group], "y": y}).dropna()
    counts = data["g"].value_counts()
    data = data[data["g"].isin(counts.index[:12])]
    rate = (data.groupby("g")["y"].mean() * 100).sort_values(ascending=False)
    if len(rate) < 2:
        return None
    overall = float(data["y"].mean() * 100)
    what = "{} = {}".format(_label(col), event)
    png = _bars(T, list(rate.index), rate.values, "Share of records where " + what,
                ref=overall, ref_label="overall {}".format(_pct(overall)), percent=True)
    parts = ["Overall, {} of records have {}. It is highest in {} ({}) and lowest in {} "
             "({}).".format(_pct(overall), what, rate.index[0], _pct(rate.iloc[0]),
                            rate.index[-1], _pct(rate.iloc[-1]))]
    test = chi2_association(pd.crosstab(data["g"], data["y"]))
    if test:
        parts.append("The difference between groups is {} (chi-square p={:.3f})."
                     .format("statistically significant" if test["p"] < 0.05
                             else "not statistically significant", test["p"]))
    return ReportChart(title="{} rate by {}".format(_label(col), _label(group).lower()),
                       png=png, caption=" ".join(parts), kind="rate", columns=(col, group))


def _trend(df, T, metric, date_col) -> Optional[ReportChart]:
    additive = is_additive(metric)
    data = df[[date_col, metric]].dropna().set_index(date_col).sort_index()
    if data.empty:
        return None
    monthly = data[metric].resample(MONTH_END).agg("sum" if additive else "mean")
    days = data[metric].resample(MONTH_END).apply(lambda s: s.index.normalize().nunique())
    # A month the data only partly covers is not comparable to the others.
    first, last = data.index.min(), data.index.max()
    keep = monthly.index
    if len(keep) and first.day > 3:
        keep = keep[1:]
    if len(keep) and last.day < 25:
        keep = keep[:-1]
    monthly = monthly.loc[keep]
    monthly = monthly[days.loc[keep] > 0]
    if len(monthly) < 4:
        return None
    what = ("Monthly total" if additive else "Monthly average") + " " + _label(metric).lower()
    # Each point sits at the start of its month: resampling labels a month
    # by its last day, which put January's value beside the February tick.
    x = monthly.index.to_period("M").to_timestamp()
    with plt.rc_context(_rc(T)):
        fig, ax = plt.subplots(figsize=(7.6, 3.0))
        ax.plot(x, monthly.values, color=T["accent"], linewidth=1.8,
                marker="o" if len(monthly) <= 24 else None, markersize=3.2)
        ax.set_ylabel(what)
        _value_axis(ax, T, horizontal=False)
        lo = float(monthly.min())
        # A total starts from zero, or a 20% movement looks like a collapse.
        if additive and lo >= 0:
            ax.set_ylim(0, float(monthly.max()) * 1.1)
        import matplotlib.dates as mdates
        loc = mdates.AutoDateLocator(maxticks=8)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y" if len(monthly) <= 36 else "%Y"))
        png = _png(fig)
    change = (monthly.iloc[-1] - monthly.iloc[0]) / abs(monthly.iloc[0]) * 100 \
        if monthly.iloc[0] else float("nan")
    peak = monthly.idxmax()
    caption = ("{} went from {} in {} to {} in {}{}. The highest month was {} ({}). "
               "Months the data only partly covers are left out.").format(
        what, fmt(monthly.iloc[0]), monthly.index[0].strftime("%b %Y"),
        fmt(monthly.iloc[-1]), monthly.index[-1].strftime("%b %Y"),
        " ({:+.0f}%)".format(change) if np.isfinite(change) else "",
        peak.strftime("%b %Y"), fmt(monthly.max()))
    return ReportChart(title=what, png=png, caption=caption, kind="trend",
                       columns=(metric, date_col))


def _distribution(df, T, metric) -> Optional[ReportChart]:
    s = df[metric].dropna()
    if len(s) < 20:
        return None
    q1, med, q3 = (float(v) for v in s.quantile([0.25, 0.5, 0.75]))
    with plt.rc_context(_rc(T)):
        fig, ax = plt.subplots(figsize=(7.6, 2.9))
        bins = min(30, max(8, int(np.sqrt(len(s)))))
        if s.nunique() <= 25 and pd.api.types.is_integer_dtype(s):
            bins = np.arange(s.min() - 0.5, s.max() + 1.5, 1)
        ax.hist(s, bins=bins, color=T["accent"], alpha=0.85, edgecolor="white", linewidth=0.6)
        ax.axvline(med, color=T["primary"], linewidth=1.2, linestyle="--")
        ax.text(med, ax.get_ylim()[1] * 0.97, " median " + fmt(med), fontsize=8,
                color=T["primary"], va="top")
        ax.set_xlabel(_label(metric))
        ax.set_ylabel("Records")
        _value_axis(ax, T, horizontal=False)
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))
        png = _png(fig)
    skew = float(s.skew())
    shape = ("closely bunched around the middle" if abs(skew) < 0.5 else
             "stretched towards high values, so the median is the better summary"
             if skew > 0 else "stretched towards low values, so the median is the better summary")
    caption = ("Half the records have {} between {} and {} (median {}). The spread is {}. "
               "Range: {} to {}.").format(_label(metric).lower(), fmt(q1), fmt(q3), fmt(med),
                                          shape, fmt(s.min()), fmt(s.max()))
    return ReportChart(title="Distribution of " + _label(metric).lower(), png=png,
                       caption=caption, kind="distribution", columns=(metric,))


def _correlations(df, T, measures) -> Optional[ReportChart]:
    cols = measures[:8]
    if len(cols) < 3:
        return None
    corr = df[cols].corr(method="spearman")
    off = corr.where(~np.eye(len(cols), dtype=bool))
    pairs = off.abs().stack().sort_values(ascending=False)
    if pairs.empty or pairs.iloc[0] < 0.3:
        return None            # a grid of zeros tells the reader nothing
    n = len(cols)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("t", [T["negative"], "#FFFFFF", T["accent"]])
    with plt.rc_context(_rc(T)):
        fig, ax = plt.subplots(figsize=(min(7.6, 1.0 * n + 2.4), min(6.0, 0.8 * n + 1.4)))
        ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
        for i in range(n):
            for j in range(n):
                v = corr.values[i, j]
                ax.text(j, i, "{:.2f}".format(v), ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.6 else T["text"])
        names = [_label(c) for c in cols]
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(names)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        png = _png(fig)
    seen, lines = set(), []
    for (a, b), _ in pairs.items():
        if (b, a) in seen or len(lines) >= 3:
            continue
        seen.add((a, b))
        r = corr.loc[a, b]
        if abs(r) < 0.3:
            break
        lines.append("{} and {} ({}, r={:.2f})".format(
            _label(a), _label(b).lower(), "move together" if r > 0 else "move in opposite directions",
            r))
    caption = ("Strongest relationships: {}. Spearman rank correlation over {} records; it "
               "shows that two measures move together, not that one causes the other."
               ).format("; ".join(lines), fmt(len(df)))
    return ReportChart(title="How the measures relate", png=png, caption=caption,
                       kind="correlation", columns=tuple(cols))


def build_report_charts(df: pd.DataFrame, theme_name: str = "", domain: str = "general",
                        max_charts: int = 5) -> List[ReportChart]:
    """The charts worth printing for this dataset, most useful first."""
    from app.engines import report_theme
    T = report_theme.get(theme_name, domain)
    measures = _measures(df)
    outcome = _outcome(df)
    group = _group_column(df, exclude=(outcome[0],) if outcome else ())
    date_col = _event_date(df)
    primary = measures[0] if measures else None

    plan = []
    if outcome and group:
        plan.append(lambda: _rate_by_group(df, T, outcome, group))
    if primary and group:
        plan.append(lambda: _by_group(df, T, primary, group))
    if primary and date_col:
        plan.append(lambda: _trend(df, T, primary, date_col))
    if primary:
        plan.append(lambda: _distribution(df, T, primary))
    if len(measures) >= 3:
        plan.append(lambda: _correlations(df, T, measures))
    if len(measures) >= 2 and group:
        plan.append(lambda: _by_group(df, T, measures[1], group))

    charts = []
    for make in plan:
        if len(charts) >= max_charts:
            break
        try:
            c = make()
        except Exception:
            logger.warning("report chart failed", exc_info=True)
            continue
        if c is not None:
            charts.append(c)
    return charts


def generate_all_charts(df: pd.DataFrame, theme_name: str = "Corporate Light",
                        max_charts: int = 5, domain: str = "general") -> List[Tuple[str, bytes]]:
    """(title, png) pairs, for callers that print no caption."""
    return [(c.title, c.png) for c in build_report_charts(df, theme_name, domain, max_charts)]
