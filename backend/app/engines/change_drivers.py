"""
engines/change_drivers.py — why the main measure moved.

"Revenue fell 12%" is where a reader's questions start: was it fewer orders
or smaller ones, and where — which region, which product, which channel?
This answers both from the data, for the same periods the scorecard
compares (whole calendar months, latest against the one before):

  - volume and value: the change split exactly into the part from the
    number of orders and the part from the value of each order;
  - where: every grouping column is tried, and the one in which the
    smallest number of segments carries the movement is reported, with
    each segment's change and its share of all the movement;
  - one level down: inside the segment that moved most, the grouping that
    explains its change.

When those periods barely moved, the latest whole month is compared with
the month before (with the same month a year earlier given as context), and
reported only when a segment moved beyond chance: a region that lost 40% of
its sales last month is the first thing an owner wants to know, and a flat
year hid it.

A segment is named only when its change is beyond chance: at least twice
(more when many segments are tested) the spread its total would show from
the orders alone (a period's total is
a sum of orders, so its variance is the sum of their squares). Without
this a 169 → 140 swing in one region's orders, a fluke, was reported as a
cause beside the real one. Segments with too few rows in either period are
not named either; their movement is still in the totals.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Below this a change is reported as stable, not explained: splitting a
# 0.4% movement into segments describes noise.
MIN_CHANGE = 0.02
# Rows a segment needs in each period to be named on its own.
MIN_ROWS = 10
# A change this many times its chance spread is named as a cause.
MIN_Z = 2.0
# explain_change looks twice (the headline periods, then the latest month),
# so each look gets half of the 5% chance of naming a cause that is not
# there; measured on 200 files of noise: 8 named one at 5% per look.
_LOOK_ALPHA = 0.025


@dataclass
class Contribution:
    segment: str
    prior: float
    current: float
    delta: float
    share: float           # of all movement in this grouping (sum of |delta|)
    z: float = 0.0         # change over its chance spread

    @property
    def pct(self) -> Optional[float]:
        return (self.delta / abs(self.prior) * 100) if self.prior else None


@dataclass
class Breakdown:
    column: str
    contributions: List[Contribution] = field(default_factory=list)
    concentration: float = 0.0     # share of movement in the top segment


@dataclass
class ChangeExplanation:
    measure: str
    period: str
    prior_period: str
    prior: float
    current: float
    volume_effect: Optional[float] = None    # from the number of orders
    value_effect: Optional[float] = None     # from the value per order
    orders_change: Optional[float] = None    # %
    value_change: Optional[float] = None     # %
    order_label: str = "orders"
    months: int = 0                          # length of each period compared
    where: Optional[Breakdown] = None
    within: Optional[Breakdown] = None
    within_segment: str = ""
    sentences: List[str] = field(default_factory=list)

    @property
    def delta(self) -> float:
        return self.current - self.prior

    @property
    def pct(self) -> Optional[float]:
        return (self.delta / abs(self.prior) * 100) if self.prior else None


def _groupings(data: pd.DataFrame, skip=()) -> List[str]:
    from app.engines.domains.base import is_id_column
    from app.engines.scorecard import _words
    from app.services.dtypes import text_columns
    out = []
    for c in text_columns(data):
        if c in skip:
            continue
        n = data[c].nunique(dropna=True)
        named = any(w in ("region", "segment", "category", "channel", "market", "country",
                          "division", "territory", "state", "city", "store", "product",
                          "brand", "department", "type", "class", "source")
                    for w in _words(c))
        if 2 <= n <= (60 if named else 25) and not is_id_column(c, data[c]):
            out.append(c)
    return out


def _breakdown(data, measure, cur, pri, col, direction, z_cut=MIN_Z,
               order=None) -> Optional[Breakdown]:
    sq = data[measure] ** 2
    work = data.assign(_sq=sq)
    c = work.loc[cur].groupby(col, observed=True).agg(sum=(measure, "sum"), size=(measure, "size"),
                                                       sq=("_sq", "sum"))
    p = work.loc[pri].groupby(col, observed=True).agg(sum=(measure, "sum"), size=(measure, "size"),
                                                       sq=("_sq", "sum"))
    both = c.join(p, how="outer", lsuffix="_c", rsuffix="_p").fillna(0)
    if len(both) < 2:
        return None
    both["delta"] = both["sum_c"] - both["sum_p"]
    gross = float(both["delta"].abs().sum())
    if gross <= 0:
        return None
    # Named on its own only with enough rows in both periods; a segment
    # that appeared or vanished is named whatever its size, since that is
    # itself the finding.
    big = ((both["size_c"] >= MIN_ROWS) & (both["size_p"] >= MIN_ROWS)) | \
          (both["size_c"] == 0) | (both["size_p"] == 0)
    spread = np.sqrt(both["sq_c"] + both["sq_p"]).replace(0, np.nan)
    both["z"] = (both["delta"] / spread).fillna(0.0)
    # The number of orders is tested too. Order values spread widely (35 to
    # 2,600 in one file), which buries a real fall in the value test: a
    # region whose orders went from 80 to 43 lost 40% of its sales at
    # z = -1.9 on value but -3.3 on the count. Counts of independent orders
    # are Poisson, so the change over the root of their sum is the z.
    if order is not None and order in work.columns:
        n_c = work.loc[cur].groupby(col, observed=True)[order].nunique()
        n_p = work.loc[pri].groupby(col, observed=True)[order].nunique()
        counts = pd.concat([n_c.rename("n_c"), n_p.rename("n_p")], axis=1).fillna(0)
    else:
        counts = both[["size_c", "size_p"]].set_axis(["n_c", "n_p"], axis=1)
    counts = counts.reindex(both.index).fillna(0)
    total_n = (counts["n_c"] + counts["n_p"]).replace(0, np.nan)
    zn = ((counts["n_c"] - counts["n_p"]) / np.sqrt(total_n)).fillna(0.0)
    # Named on the count only when the orders moved the way the money did.
    by_count = (zn.abs() >= z_cut) & (np.sign(zn) == np.sign(both["delta"]))
    both["z"] = np.where(by_count & (zn.abs() > both["z"].abs()), zn, both["z"])
    real = big & ((both["z"].abs() >= z_cut) | by_count)
    rows = both[real].sort_values("delta", key=lambda s: -s * direction)
    contribs = [Contribution(str(k), float(r["sum_p"]), float(r["sum_c"]),
                             float(r["delta"]), float(abs(r["delta"]) / gross), float(r["z"]))
                for k, r in rows.iterrows()]
    if not contribs:
        return None
    # How much of the movement the segment that moved most with the total
    # carries. A grouping whose every segment moved a little explains
    # nothing a reader can act on.
    lead = max((x for x in contribs if np.sign(x.delta) == direction),
               key=lambda x: abs(x.delta), default=None)
    conc = lead.share if lead else 0.0
    return Breakdown(col, contribs, conc)


def _best(data, measure, cur, pri, direction, skip=(), order=None,
          alpha: float = 0.05) -> Optional[Breakdown]:
    from scipy.stats import norm
    cols = _groupings(data, skip)
    # Every segment of every grouping is a test, twice (its value and its
    # number of orders); the bar rises with their number (Bonferroni, 5%
    # overall). At a flat 2, 6 of 40 datasets of pure noise had a "cause"
    # named.
    m = 2 * sum(int(data[c].nunique()) for c in cols)
    z_cut = max(MIN_Z, float(norm.isf(alpha / (2 * max(m, 1)))))
    best = None
    for col in cols:
        try:
            b = _breakdown(data, measure, cur, pri, col, direction, z_cut, order)
        except Exception:
            logger.debug("breakdown failed for %s", col, exc_info=True)
            continue
        if b and (best is None or b.concentration > best.concentration):
            best = b
    return best


def explain_change(df: pd.DataFrame) -> Optional[ChangeExplanation]:
    """Why the main measure moved, or None when nothing moved that can be
    explained (see explain_change_or_reason for why)."""
    return explain_change_or_reason(df)[0]


def explain_change_or_reason(df: pd.DataFrame):
    """(explanation, None) or (None, the reason there is none, in words)."""
    from app.engines.scorecard import _ORDER, _key_column, _periods, build_scorecard
    from app.engines.transactions import order_column
    sc = build_scorecard(df)
    if not sc.revenue_col:
        return None, ("No change to explain: there is no money measure, such as sales or "
                      "revenue, to follow.")
    if not sc.date_col:
        return None, "No change to explain: there is no date column to compare periods by."
    measure = sc.revenue_col
    data = df.copy()
    data[measure] = pd.to_numeric(data[measure], errors="coerce")
    data = data[data[measure].notna()]
    order = order_column(df) or _key_column(df, _ORDER)
    headline, described = None, None
    periods = _periods(data[sc.date_col]) if sc.period else None
    if periods:
        e = explain_periods(data, measure, periods[0], periods[1], sc.period,
                            sc.prior_period, order, alpha=_LOOK_ALPHA)
        if e is not None:
            e.months = 12 if sc.period.startswith("12") else 3
            if e.where is not None:
                return e, None
            # The total moved but no group beyond chance: a description,
            # kept only if the latest month has no cause to name.
            described = e
        else:
            cur = float(data.loc[periods[0], measure].sum())
            pri = float(data.loc[periods[1], measure].sum())
            if pri:
                headline = "'{}' changed {:+.1f}% between the {} and the {}".format(
                    measure, (cur - pri) / abs(pri) * 100, sc.period, sc.prior_period)
    recent = _latest_month(data, measure, sc.date_col, order)
    if recent is not None:
        return recent, None
    if described is not None:
        return described, None
    if headline:
        return None, (headline + ", too little to explain, and in the latest month no "
                      "group moved by more than chance produces.")
    return None, ("No change to explain: the dates cover too little to compare two whole "
                  "periods, or no group moved by more than chance produces.")


def _latest_month(data, measure, date_col, order) -> Optional[ChangeExplanation]:
    """The latest whole month against the same month a year earlier (the
    month before when there is no year of history), explained only when a
    segment moved beyond chance: one month's total moves by chance alone."""
    from app.engines.scorecard import last_complete_end
    d = pd.to_datetime(data[date_col], errors="coerce")
    end = last_complete_end(d)
    if end is None:
        return None
    month = (end - pd.Timedelta(days=1)).to_period("M")
    months = d.dt.to_period("M")
    covered = d.min()
    # Against the month before: the question an owner asks ("why did
    # October fall?"). The same month a year earlier is a single month too,
    # just as noisy, and is given beside it as context for the season.
    prior = month - 1
    if covered > prior.start_time + pd.Timedelta(days=3):
        return None
    cur, pri = months == month, months == prior
    if cur.sum() < MIN_ROWS or pri.sum() < MIN_ROWS:
        return None
    label = "month of {}".format(month.start_time.strftime("%b %Y"))
    prior_label = "month of {}".format(prior.start_time.strftime("%b %Y"))
    e = explain_periods(data, measure, cur, pri, label, prior_label, order,
                        alpha=_LOOK_ALPHA)
    if e is None or e.where is None or e.where.concentration <= 0:
        return None
    e.months = 1
    year_ago = month - 12
    if covered <= year_ago.start_time + pd.Timedelta(days=3):
        before = float(data.loc[months == year_ago, measure].sum())
        if before:
            from app.engines.chart_exporter import fmt
            e.sentences.insert(1, "Against {} a year earlier ({}), it is {:+.1f}%.".format(
                year_ago.start_time.strftime("%b %Y"), fmt(before),
                (e.current - before) / abs(before) * 100))
    return e


def explain_periods(data: pd.DataFrame, measure: str, cur, pri, period: str,
                    prior_period: str, order: Optional[str] = None,
                    where_only: bool = False,
                    alpha: float = 0.05) -> Optional[ChangeExplanation]:
    """The explanation for given period masks. The scorecard calls this
    with its own periods (`where_only` for just the segments: it states
    volume and value itself)."""
    from app.engines.transactions import cancellation_mask
    now, before = float(data.loc[cur, measure].sum()), float(data.loc[pri, measure].sum())
    out = ChangeExplanation(measure, period, prior_period, before, now)
    if not before or abs(out.delta) / abs(before) < MIN_CHANGE:
        return None
    # The total is not tested against chance on its own: a 5% fall can be
    # within the spread of the whole book while one region's sales of one
    # category fell 83%, far beyond it. Each segment named is tested.
    direction = 1 if out.delta > 0 else -1

    # Volume and value, exactly: Δ = (O1 − O0)·V0 + O1·(V1 − V0).
    if order is not None and not where_only:
        keep = ~cancellation_mask(data)
        n1 = data.loc[cur & keep, order].nunique()
        n0 = data.loc[pri & keep, order].nunique()
        if n0 and n1:
            v0, v1 = before / n0, now / n1
            out.volume_effect = (n1 - n0) * v0
            out.value_effect = n1 * (v1 - v0)
            out.orders_change = (n1 - n0) / n0 * 100
            out.value_change = (v1 - v0) / abs(v0) * 100 if v0 else None

    out.where = _best(data, measure, cur, pri, direction, order=order, alpha=alpha)
    if out.where and out.where.contributions:
        lead = out.where.contributions[0]
        if np.sign(lead.delta) == direction:
            inside = data[data[out.where.column].astype(str) == lead.segment]
            within = _best(inside, measure, cur.loc[inside.index], pri.loc[inside.index],
                           direction, skip=(out.where.column,), order=order, alpha=alpha)
            if within and within.concentration >= 0.4:
                out.within, out.within_segment = within, lead.segment
    out.sentences = _sentences(out)
    return out


def _money(v: float) -> str:
    from app.engines.chart_exporter import fmt
    return ("+" if v > 0 else "−" if v < 0 else "") + fmt(abs(v))


def _item(c: Contribution) -> str:
    pct = c.pct
    if c.prior == 0:
        return "{}: {} (new)".format(c.segment, _money(c.delta))
    if c.current == 0:
        return "{}: {} (none this period)".format(c.segment, _money(c.delta))
    return "{}: {} ({:+.0f}%)".format(c.segment, _money(c.delta), pct)


def _sentences(e: ChangeExplanation) -> List[str]:
    from app.engines.chart_exporter import fmt
    up = e.delta > 0
    s = ["'{}' {} {:.1f}% ({}) in the {}, against {} in the {}.".format(
        e.measure, "rose" if up else "fell", abs(e.pct), _money(e.delta), e.period,
        fmt(e.prior), e.prior_period)]
    if e.volume_effect is not None and e.value_effect is not None and e.value_change is not None:
        s.append("{} {} {:.1f}% and the value per order {} {:.1f}%: {} came from the number "
                 "of {}, {} from their value.".format(
                     e.order_label.capitalize(), "rose" if e.orders_change >= 0 else "fell",
                     abs(e.orders_change), "rose" if e.value_change >= 0 else "fell",
                     abs(e.value_change), _money(e.volume_effect), e.order_label,
                     _money(e.value_effect)))
    if e.where:
        same = [c for c in e.where.contributions if (c.delta > 0) == up][:3]
        other = [c for c in e.where.contributions if (c.delta > 0) != up and c.delta][:2]
        if same:
            line = "By '{}': {}".format(e.where.column, " · ".join(_item(c) for c in same))
            if other:
                line += "; {} the other way: {}".format(
                    "offset" if len(other) > 1 else "offset", " · ".join(_item(c) for c in other))
            s.append(line + ". '{}' carries {:.0f}% of all the movement across '{}'.".format(
                same[0].segment, same[0].share * 100, e.where.column))
    if e.within and e.within.contributions:
        same = [c for c in e.within.contributions if (c.delta > 0) == up][:2]
        if same:
            s.append("Within '{}', by '{}': {}.".format(
                e.within_segment, e.within.column, " · ".join(_item(c) for c in same)))
    return s
