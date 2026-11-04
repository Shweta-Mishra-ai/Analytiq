"""
engines/scorecard.py — the business figures a report leads with.

A consulting deliverable opens with a handful of headline measures (revenue,
profit, margin, volume), how each moved against the prior period, and the
three to five messages the reader should leave with — each backed by a
figure and tied to an action. The report used to open with whichever
statistical finding scored highest (often a customer-recency quirk) and
never stated the size of the business, its growth or its margin.

Everything here is computed from the uploaded data and only for the columns
that exist; a dataset without a revenue-like measure gets no scorecard, not
an invented one. Comparisons between groups are tested before they are
called differences, and the period comparison is a fact about totals, not an
inference, so it is reported as such.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REVENUE = ("revenue", "sales", "turnover", "gmv", "amount", "income", "billing")
_PROFIT = ("profit", "earnings", "ebit", "ebitda", "contribution")
_QTY = ("quantity", "qty", "units", "volume")
_DISCOUNT = ("discount",)
_CUSTOMER = ("customer", "client", "account", "buyer", "member")
_ORDER = ("order", "invoice", "transaction", "receipt", "booking")
_NOT_MONEY = ("pct", "percent", "rate", "ratio", "share", "count", "qty", "quantity",
              "units", "score", "rank", "id", "growth", "margin")


def _words(name: str) -> List[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    return [w for w in re.split(r"[^a-z0-9]+", spaced.lower()) if w]


def _has(name: str, keys) -> bool:
    ws = _words(name)
    return any(w.startswith(k) or w == k for w in ws for k in keys)


def _measure(df: pd.DataFrame, keys, exclude=()) -> Optional[str]:
    from app.engines.domains.base import is_id_column
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        if _has(c, keys) and not _has(c, exclude) and not is_id_column(c, df[c]):
            if df[c].notna().sum() >= 10:
                return c
    return None


def _key_column(df: pd.DataFrame, keys) -> Optional[str]:
    """An identifier column (customer, order) used for counting."""
    from app.services.dtypes import is_text_dtype
    for c in df.columns:
        if _has(c, keys) and (_has(c, ("id", "code", "no", "number", "key", "ref"))
                              or is_text_dtype(df[c])):
            return c
    return None


@dataclass
class Metric:
    label: str
    kind: str                      # money | pct | count | number
    value: float
    prior: Optional[float] = None

    @property
    def change(self) -> Optional[float]:
        """Relative change for amounts; percentage points for a percentage."""
        if self.prior is None or not np.isfinite(self.prior):
            return None
        if self.kind == "pct":
            return (self.value - self.prior) * 100
        if self.prior == 0:
            return None
        return (self.value - self.prior) / abs(self.prior) * 100


@dataclass
class Message:
    title: str                     # the finding, stated as a sentence
    detail: str                    # the figures behind it
    action: str                    # what to do about it
    at_stake: float = 0.0          # revenue the finding concerns, for ranking
    kind: str = "info"
    lead: bool = False             # states where the business stands; shown first


@dataclass
class Scorecard:
    revenue_col: Optional[str] = None
    profit_col: Optional[str] = None
    date_col: Optional[str] = None
    period: str = ""               # e.g. "12 months to Dec 2024"
    prior_period: str = ""
    metrics: List[Metric] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)
    total_revenue: float = 0.0

    @property
    def has_comparison(self) -> bool:
        return bool(self.prior_period)


def fmt_value(m: Metric) -> str:
    from app.engines.chart_exporter import fmt
    if m.kind == "pct":
        return "{:.1f}%".format(m.value * 100)
    return fmt(m.value)


def fmt_change(m: Metric) -> str:
    ch = m.change
    if ch is None:
        return "—"
    if m.kind == "pct":
        return "{:+.1f} pp".format(ch)
    return "{:+.1f}%".format(ch)


# ── periods ──────────────────────────────────────────────────────────────

def period_bounds(dates: pd.Series):
    """(start, cut, end, label, prior label) of the latest complete
    comparison the dates allow: 12 months against the 12 before when they
    cover two years, otherwise 3 months against the 3 before when they cover
    six. Periods are whole calendar months, ending with the last month the
    dates cover to within its final three days, so the figures match what a
    reader totals by month. `end` is exclusive. None when they cover too
    little."""
    # English month names: the label is matched back to its catalog entry
    # when printed, and the month is translated there with the rest.
    def month_year(ts):
        return ts.strftime("%b %Y")
    d = pd.to_datetime(dates, errors="coerce")
    lo, hi = d.min(), d.max()
    if pd.isna(lo) or pd.isna(hi):
        return None
    month_end = hi + pd.offsets.MonthEnd(0)
    end = month_end if (month_end - hi).days <= 3 else hi - pd.offsets.MonthEnd(1)
    end = end.normalize() + pd.Timedelta(days=1)            # exclusive bound
    for months, label in ((12, "12 months to {}"), (3, "3 months to {}")):
        cut = end - pd.DateOffset(months=months)
        start = cut - pd.DateOffset(months=months)
        if lo > start + pd.Timedelta(days=3):
            continue
        return (start, cut, end, label.format(month_year(end - pd.Timedelta(days=1))),
                label.format(month_year(cut - pd.Timedelta(days=1))))
    return None


def _periods(dates: pd.Series, bounds=None):
    """(current mask, prior mask, label, prior label) over `dates`, for the
    given bounds or the ones these dates allow. A filtered view passes the
    whole dataset's bounds: its own last row may fall a few days short of a
    month end, and its cards then compared different periods from the
    unfiltered ones."""
    bounds = bounds or period_bounds(dates)
    if not bounds:
        return None
    start, cut, end, label, prior_label = bounds
    d = pd.to_datetime(dates, errors="coerce")
    cur = (d >= cut) & (d < end)
    pri = (d >= start) & (d < cut)
    if cur.sum() < 10 or pri.sum() < 10:
        return None
    return cur, pri, label, prior_label


# ── scorecard ────────────────────────────────────────────────────────────

def build_scorecard(df: pd.DataFrame, bounds=None) -> Scorecard:
    """The scorecard of `df`. `bounds` (from period_bounds on the whole
    dataset) fixes the comparison periods for a filtered view."""
    sc = Scorecard()
    rev = _measure(df, _REVENUE, exclude=_NOT_MONEY + _PROFIT + ("cost", "price"))
    if rev is None:
        return sc
    sc.revenue_col = rev
    profit = _measure(df, _PROFIT, exclude=("pct", "percent", "rate", "ratio", "margin"))
    sc.profit_col = profit
    qty = _measure(df, _QTY, exclude=("price",))
    disc = _measure(df, _DISCOUNT, exclude=("amount", "value", "price", "cost"))
    customer = _key_column(df, _CUSTOMER)
    order = _key_column(df, _ORDER)
    dates = df.select_dtypes(include="datetime")
    if not dates.empty:
        sc.date_col = dates.notna().sum().idxmax()

    data = df.copy()
    data[rev] = pd.to_numeric(data[rev], errors="coerce")
    data = data[data[rev].notna()]
    sc.total_revenue = float(data[rev].sum())

    def figures(part: pd.DataFrame) -> dict:
        r = float(part[rev].sum())
        out = {"Revenue": ("money", r)}
        if profit:
            p = float(part[profit].sum())
            out["Profit"] = ("money", p)
            if r > 0:
                out["Profit margin"] = ("pct", p / r)
        n_orders = part[order].nunique() if order else len(part)
        out["Orders" if order else "Records"] = ("count", float(n_orders))
        if n_orders:
            out["Average order value" if order else "Revenue per record"] = (
                "money", r / n_orders)
        if customer:
            out["Customers"] = ("count", float(part[customer].nunique()))
        if qty:
            out["Units"] = ("count", float(part[qty].sum()))
        if disc:
            rate = _as_rate(part[disc])
            if rate is not None and r > 0:
                w = part[rev].clip(lower=0)
                out["Average discount"] = ("pct", float((rate * w).sum() / max(w.sum(), 1e-9)))
        return out

    periods = _periods(data[sc.date_col], bounds) if sc.date_col else None
    if periods:
        cur_mask, pri_mask, sc.period, sc.prior_period = periods
        cur, pri = figures(data[cur_mask]), figures(data[pri_mask])
        sc.metrics = [Metric(k, kind, v, pri.get(k, (kind, None))[1])
                      for k, (kind, v) in cur.items()]
    else:
        sc.metrics = [Metric(k, kind, v) for k, (kind, v) in figures(data).items()]

    try:
        _messages(sc, data, rev, profit, disc, customer, periods)
    except Exception:
        logger.warning("scorecard messages failed", exc_info=True)
    # The state of the business first (revenue and its change), then the
    # findings by the revenue they concern.
    sc.messages.sort(key=lambda m: (not m.lead, -m.at_stake))
    return sc


def _as_rate(s: pd.Series) -> Optional[pd.Series]:
    """A discount column as a fraction (0.2 for 20%), or None if it is not
    a rate (an amount in money would read as a 2,000% discount)."""
    v = pd.to_numeric(s, errors="coerce")
    if v.dropna().empty or v.min() < 0:
        return None
    if v.max() <= 1:
        return v
    if v.max() <= 100:
        return v / 100
    return None


def _messages(sc, data, rev, profit, disc, customer, periods):
    from app.engines.chart_exporter import fmt
    total = max(sc.total_revenue, 1e-9)
    by = {m.label: m for m in sc.metrics}

    # 1. Growth against the prior period, with the margin beside it.
    r = by.get("Revenue")
    if r and r.change is not None:
        up = r.change >= 0
        detail = "{} against {} in the {}.".format(fmt(r.value), fmt(r.prior), sc.prior_period)
        pm = by.get("Profit margin")
        if pm and pm.change is not None:
            detail += " Profit margin {} from {:.1f}% to {:.1f}%.".format(
                "rose" if pm.change > 0 else "fell" if pm.change < 0 else "held",
                pm.prior * 100, pm.value * 100)
        drivers = _growth_drivers(data, rev, periods)
        if drivers:
            detail += " " + drivers
        sc.messages.append(Message(
            title="Revenue {} {:.1f}% to {} in the {}".format(
                "grew" if up else "fell", abs(r.change), fmt(r.value), sc.period),
            detail=detail,
            action=("Confirm which segments carried the change before setting next year's "
                    "targets on it." if up else
                    "Find which segments account for the fall and whether it is volume or "
                    "price."),
            at_stake=abs(r.value - r.prior), kind="positive" if up else "negative",
            lead=True))

    # 2. Discounting and margin.
    if disc and profit:
        msg = _discount_message(data, rev, profit, disc)
        if msg:
            sc.messages.append(msg)

    # 3. Where margin is made and lost.
    if profit:
        msg = _segment_margin_message(data, rev, profit)
        if msg:
            sc.messages.append(msg)

    # 4. Customer concentration.
    if customer:
        per = data.groupby(customer)[rev].sum().sort_values(ascending=False)
        per = per[per > 0]
        if len(per) >= 20:
            k = max(1, int(round(len(per) * 0.2)))
            share = float(per.iloc[:k].sum() / per.sum() * 100)
            if share >= 50:
                sc.messages.append(Message(
                    title="The top 20% of customers bring {:.0f}% of revenue".format(share),
                    detail="{:,} of {:,} customers account for {} of {}.".format(
                        k, len(per), fmt(per.iloc[:k].sum()), fmt(per.sum())),
                    action="Put an account plan and a retention check on the largest "
                           "customers; losing a few would move the total.",
                    at_stake=float(per.iloc[:k].sum()) * 0.25, kind="warning"))
    del total


def _growth_drivers(data, rev, periods) -> str:
    """Which group of the main category carried most of the change."""
    if not periods:
        return ""
    cur_mask, pri_mask = periods[0], periods[1]
    cat = _main_category(data)
    if not cat:
        return ""
    cur = data[cur_mask].groupby(cat)[rev].sum()
    pri = data[pri_mask].groupby(cat)[rev].sum()
    delta = cur.sub(pri, fill_value=0).sort_values()
    if len(delta) < 2:
        return ""
    from app.engines.chart_exporter import fmt
    # The largest rise and the largest fall, in money. A share of the net
    # change ("'East' accounts for 95% of it") hid a region that fell while
    # the others rose.
    parts = []
    if delta.iloc[-1] > 0:
        parts.append("'{}' added the most (+{})".format(
            delta.index[-1], fmt(float(delta.iloc[-1]))))
    if delta.iloc[0] < 0:
        parts.append("'{}' fell the most ({})".format(
            delta.index[0], fmt(float(delta.iloc[0]))))
    if not parts:
        return ""
    return "By '{}': {}.".format(cat, "; ".join(parts))


def _main_category(data) -> Optional[str]:
    from app.engines.domains.base import is_id_column
    from app.services.dtypes import text_columns
    best = None
    for c in text_columns(data):
        n = data[c].nunique()
        if 2 <= n <= 12 and not is_id_column(c, data[c]):
            words = _words(c)
            score = (2 if any(w in ("region", "segment", "category", "channel", "market",
                                    "country", "division") for w in words) else 1)
            if best is None or score > best[0]:
                best = (score, c)
    return best[1] if best else None


def _discount_message(data, rev, profit, disc) -> Optional[Message]:
    from scipy import stats as sc_stats
    from app.engines.chart_exporter import fmt
    rate = _as_rate(data[disc])
    if rate is None:
        return None
    sub = pd.DataFrame({"rate": rate, "rev": data[rev], "profit": data[profit]}).dropna()
    sub = sub[sub["rev"] > 0]
    if len(sub) < 60 or (sub["rate"] == 0).sum() < 20:
        return None
    bins = [-0.001, 0.0, 0.10, 0.20, 1.0]
    names = ["no discount", "up to 10%", "10–20%", "over 20%"]
    sub["band"] = pd.cut(sub["rate"], bins=bins, labels=names)
    g = sub.groupby("band", observed=True).agg(rev=("rev", "sum"), profit=("profit", "sum"),
                                                n=("rev", "size"))
    g = g[g["n"] >= 20]
    if "no discount" not in g.index or len(g) < 2:
        return None
    g["margin"] = g["profit"] / g["rev"]
    deepest = g.index[-1]
    base_m, deep_m = float(g.loc["no discount", "margin"]), float(g.loc[deepest, "margin"])
    gap_pp = (base_m - deep_m) * 100
    share = float(g.loc[deepest, "rev"] / g["rev"].sum() * 100)
    row_margin = sub["profit"] / sub["rev"]
    groups = [row_margin[sub["band"] == b].values for b in g.index]
    p = float(sc_stats.kruskal(*groups).pvalue)
    if gap_pp < 5 or share < 5 or p >= 0.05:
        return None
    loses = deep_m < 0
    title = ("Orders discounted {} lose money: {:.1f}% margin against {:.1f}% at full price"
             if loses else
             "Orders discounted {} earn {:.1f}% margin against {:.1f}% at full price")
    return Message(
        title=title.format(deepest, deep_m * 100, base_m * 100),
        detail=("They carry {:.0f}% of revenue ({}). Margin falls band by band: {}. "
                "This is an association across orders; it does not show what volume the "
                "discounts won.").format(
                    share, fmt(g.loc[deepest, "rev"]),
                    " · ".join("{}: {:.1f}%".format(b, m * 100)
                               for b, m in g["margin"].items())),
        action=("Require approval for discounts {} and test a lower cap on a subset of "
                "orders, measuring the volume lost against the margin regained."
                ).format(deepest),
        at_stake=float(g.loc[deepest, "rev"]), kind="negative" if loses else "warning")


def _segment_margin_message(data, rev, profit) -> Optional[Message]:
    from scipy import stats as sc_stats
    from app.engines.chart_exporter import fmt
    from app.services.stat_guards import bh_adjust
    from app.engines.domains.base import is_id_column
    from app.services.dtypes import text_columns
    sub = data[data[rev] > 0]
    tested = []
    for c in text_columns(sub):
        if not (2 <= sub[c].nunique() <= 12) or is_id_column(c, sub[c]):
            continue
        g = sub.groupby(c).agg(rev=(rev, "sum"), profit=(profit, "sum"), n=(rev, "size"))
        g = g[g["n"] >= 30]
        if len(g) < 2:
            continue
        g["margin"] = g["profit"] / g["rev"]
        rm = sub[profit] / sub[rev]
        groups = [rm[sub[c] == k].values for k in g.index]
        try:
            p = float(sc_stats.kruskal(*groups).pvalue)
        except ValueError:
            continue
        tested.append((c, g, p))
    if not tested:
        return None
    best = None
    for (c, g, _p), q in zip(tested, bh_adjust([t[2] for t in tested])):
        spread = float(g["margin"].max() - g["margin"].min()) * 100
        if q < 0.05 and spread >= 5 and (best is None or spread > best[0]):
            best = (spread, c, g)
    if not best:
        return None
    spread, c, g = best
    hi, lo = g["margin"].idxmax(), g["margin"].idxmin()
    lo_share = float(g.loc[lo, "rev"] / g["rev"].sum() * 100)
    return Message(
        title="Margin ranges from {:.1f}% in '{}' to {:.1f}% in '{}'".format(
            g.loc[lo, "margin"] * 100, lo, g.loc[hi, "margin"] * 100, hi),
        detail=("Across '{}', a {:.1f}-point spread; '{}' carries {:.0f}% of revenue ({})."
                ).format(c, spread, lo, lo_share, fmt(g.loc[lo, "rev"])),
        action=("Break '{}' down by product, price and discount to find what holds its "
                "margin down.").format(lo),
        at_stake=float(g.loc[lo, "rev"]), kind="warning")


def priority(msg: Message, total_revenue: float) -> str:
    """High / Medium / Low by the share of revenue the finding concerns."""
    share = msg.at_stake / max(total_revenue, 1e-9)
    return "High" if share >= 0.20 else "Medium" if share >= 0.05 else "Low"
