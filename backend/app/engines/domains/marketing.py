"""
domains/marketing.py — campaign and channel performance.

A marketing export (one row per campaign, ad, day or channel) carries
spend, impressions, clicks, conversions and often revenue. The questions a
marketing lead asks of it are fixed: what did we spend and get, which
channels and campaigns earn their money and which lose it, where in the
funnel each one leaks, and where the next pound should go.

Method (the same rules a careful analyst applies):

  * Every rate is a ratio of totals. CTR is total clicks over total
    impressions, never the average of each row's CTR: averaging rates
    weights a campaign with 40 impressions like one with 4 million.
  * A ratio column the file already carries ("CTR", "ROAS") is not trusted
    over its components; it is recomputed from them.
  * A channel or campaign is ranked only with enough volume behind it, and
    differences in conversion rate are tested (chi-square across channels,
    each channel against the rest with Benjamini-Hochberg correction), so a
    small channel with three lucky conversions is not crowned.
  * The funnel is diagnosed stage by stage: a low click-through rate is a
    targeting or creative problem, a low conversion rate after the click
    is a landing-page or offer problem. They need different fixes.
  * Reallocation advice names its own limit: returns fall as spend rises,
    so a channel's past return is an upper bound for the next pound.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sps

from app.engines.domains.base import build_insight, is_id_column

logger = logging.getLogger(__name__)

MIN_CLICKS = 100          # below this a conversion rate is an anecdote
MIN_SPEND_SHARE = 0.02    # a channel under 2% of spend is not ranked


def _words(name) -> List[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    return [w for w in re.split(r"[^a-z0-9]+", spaced.lower()) if w]


def _num(df: pd.DataFrame, keys, exclude=()) -> Optional[str]:
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        ws = _words(c)
        if any(w.startswith(k) for w in ws for k in keys) and not any(
                w.startswith(x) for w in ws for x in exclude):
            if not is_id_column(c, df[c]):
                return c
    return None


def _cat(df: pd.DataFrame, keys, max_levels=60) -> Optional[str]:
    from app.services.dtypes import text_columns
    for c in text_columns(df):
        ws = _words(c)
        if any(w.startswith(k) for w in ws for k in keys):
            k = df[c].nunique(dropna=True)
            if 2 <= k <= max_levels:
                return c
    return None


@dataclass
class MarketingColumns:
    spend: Optional[str] = None
    impressions: Optional[str] = None
    clicks: Optional[str] = None
    conversions: Optional[str] = None
    revenue: Optional[str] = None
    channel: Optional[str] = None
    campaign: Optional[str] = None
    date: Optional[str] = None

    @property
    def usable(self) -> bool:
        # Something spent and something that came back.
        return bool(self.spend and (self.clicks or self.conversions or self.revenue))


def detect_columns(df: pd.DataFrame) -> MarketingColumns:
    rate = ("rate", "ctr", "cpc", "cpm", "cpa", "roas", "roi", "pct", "percent", "per", "avg",
            "average", "cost per")
    cols = MarketingColumns(
        spend=_num(df, ("spend", "cost", "budget", "amount spent"), exclude=rate + ("unit",)),
        impressions=_num(df, ("impression", "impr", "views", "reach"), exclude=rate),
        clicks=_num(df, ("click",), exclude=rate),
        conversions=_num(df, ("conversion", "conv", "lead", "signup", "sign", "purchase",
                              "acquisition", "install", "sale"),
                         exclude=rate + ("value", "revenue", "amount")),
        revenue=_num(df, ("revenue", "conversion value", "value", "sales amount", "income"),
                     exclude=rate + ("spend", "cost")),
        channel=_cat(df, ("channel", "source", "medium", "platform", "network")),
        campaign=_cat(df, ("campaign", "ad group", "adgroup", "adset", "ad"), max_levels=500),
    )
    if cols.conversions == cols.revenue:
        cols.revenue = None
    dates = df.select_dtypes(include="datetime")
    if not dates.empty:
        cols.date = dates.notna().sum().idxmax()
    return cols


@dataclass
class GroupRow:
    name: str
    spend: float
    impressions: float = float("nan")
    clicks: float = float("nan")
    conversions: float = float("nan")
    revenue: float = float("nan")

    @property
    def ctr(self):
        return _ratio(self.clicks, self.impressions)

    @property
    def cvr(self):
        return _ratio(self.conversions, self.clicks)

    @property
    def cpc(self):
        return _ratio(self.spend, self.clicks)

    @property
    def cpm(self):
        r = _ratio(self.spend, self.impressions)
        return r * 1000 if np.isfinite(r) else r

    @property
    def cpa(self):
        return _ratio(self.spend, self.conversions)

    @property
    def roas(self):
        return _ratio(self.revenue, self.spend)


def _ratio(a, b) -> float:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return float("nan")
    return a / b if np.isfinite(a) and np.isfinite(b) and b > 0 else float("nan")


@dataclass
class MarketingReport:
    columns: MarketingColumns
    total: GroupRow
    by_channel: List[GroupRow] = field(default_factory=list)
    by_campaign: List[GroupRow] = field(default_factory=list)
    period: str = ""


def _sum(part: pd.DataFrame, col: Optional[str]) -> float:
    if not col:
        return float("nan")
    return float(pd.to_numeric(part[col], errors="coerce").clip(lower=0).sum())


def _row(name: str, part: pd.DataFrame, c: MarketingColumns) -> GroupRow:
    return GroupRow(name=str(name), spend=_sum(part, c.spend),
                    impressions=_sum(part, c.impressions), clicks=_sum(part, c.clicks),
                    conversions=_sum(part, c.conversions), revenue=_sum(part, c.revenue))


def analyse(df: pd.DataFrame, cols: Optional[MarketingColumns] = None) -> Optional[MarketingReport]:
    c = cols or detect_columns(df)
    if not c.usable:
        return None
    rep = MarketingReport(columns=c, total=_row("All", df, c))
    if c.channel:
        rep.by_channel = sorted((_row(k, g, c) for k, g in df.groupby(c.channel, observed=True)),
                                key=lambda r: -r.spend)
    if c.campaign and c.campaign != c.channel:
        rep.by_campaign = sorted((_row(k, g, c) for k, g in df.groupby(c.campaign, observed=True)),
                                 key=lambda r: -r.spend)
    return rep


def _money(x: float) -> str:
    from app.engines.chart_exporter import fmt
    return fmt(x)


def _pct(x: float, digits=1) -> str:
    return ("{:.%df}%%" % digits).format(x * 100) if np.isfinite(x) else "n/a"


# ── insights ─────────────────────────────────────────────

def _summary_finding(rep: MarketingReport) -> str:
    t, c = rep.total, rep.columns
    parts = ["Spend {}".format(_money(t.spend))]
    if c.impressions and c.clicks:
        parts.append("{} impressions, {} clicks (CTR {})".format(
            _money(t.impressions), _money(t.clicks), _pct(t.ctr, 2)))
    if c.conversions:
        parts.append("{} conversions".format(_money(t.conversions)))
        if c.clicks:
            parts.append("conversion rate {}".format(_pct(t.cvr, 2)))
        parts.append("cost per conversion {}".format(_money(t.cpa)))
    if c.revenue:
        parts.append("revenue {} (ROAS {:.2f})".format(_money(t.revenue), t.roas))
    return "Marketing totals: " + " · ".join(parts) + "."


def _eligible(rows: List[GroupRow], total_spend: float) -> List[GroupRow]:
    return [r for r in rows if total_spend > 0 and r.spend / total_spend >= MIN_SPEND_SHARE]


def _channel_conversion_test(rows: List[GroupRow]):
    """Chi-square of converted vs not across groups with enough clicks, and
    each group against the rest (BH-adjusted). None if not testable."""
    from app.services.stat_guards import bh_adjust
    use = [r for r in rows if np.isfinite(r.clicks) and r.clicks >= MIN_CLICKS
           and np.isfinite(r.conversions) and r.conversions <= r.clicks]
    if len(use) < 2:
        return None
    table = np.array([[r.conversions, r.clicks - r.conversions] for r in use])
    if (table.sum(axis=0) == 0).any():
        return None
    _, p_all, _, _ = sps.chi2_contingency(table)
    pairs = []
    for i, r in enumerate(use):
        rest = table.sum(axis=0) - table[i]
        _, p, _, _ = sps.chi2_contingency(np.vstack([table[i], rest]))
        pairs.append(p)
    adj = bh_adjust(pairs)
    return use, float(p_all), dict(zip([r.name for r in use], adj))


def channel_insights(rep: MarketingReport, label: str, rows: List[GroupRow],
                     insights: List, findings: List, risks: List, opps: List) -> None:
    c, t = rep.columns, rep.total
    elig = _eligible(rows, t.spend)
    if len(elig) < 2:
        return

    # Return on spend: who earns it, who loses it.
    if c.revenue and np.isfinite(t.roas):
        ranked = sorted([r for r in elig if np.isfinite(r.roas)], key=lambda r: -r.roas)
        if len(ranked) >= 2:
            best, worst = ranked[0], ranked[-1]
            losing = [r for r in ranked if r.roas < 1]
            lost_spend = sum(r.spend for r in losing)
            detail = " · ".join("{}: {:.2f} on {}".format(r.name, r.roas, _money(r.spend))
                                for r in ranked[:8])
            if losing:
                risks.append(
                    "{} returning less than they cost (ROAS below 1): {} — {} of spend, "
                    "{:.0f}% of the total.".format(
                        label.capitalize() + "s", ", ".join(r.name for r in losing[:5]),
                        _money(lost_spend), lost_spend / t.spend * 100))
            # Under-funded winners: a bigger share of revenue than of spend.
            if np.isfinite(t.revenue) and t.revenue > 0:
                under = [r for r in ranked if r.roas >= 1.5 * t.roas
                         and r.spend / t.spend < r.revenue / t.revenue]
                for r in under[:2]:
                    opps.append(
                        "'{}' brings {:.0f}% of revenue on {:.0f}% of spend (ROAS {:.2f} "
                        "against {:.2f} overall): the first place to test more budget, "
                        "in steps, watching whether its return holds.".format(
                            r.name, r.revenue / t.revenue * 100, r.spend / t.spend * 100,
                            r.roas, t.roas))
            insights.append(build_insight(
                title="Return on ad spend ranges from {:.2f} ({}) to {:.2f} ({})".format(
                    best.roas, best.name, worst.roas, worst.name),
                problem="Across {}s with at least {:.0f}% of spend, every unit spent on "
                        "'{}' returned {:.2f} in revenue against {:.2f} on '{}'.".format(
                            label, MIN_SPEND_SHARE * 100, best.name, best.roas,
                            worst.roas, worst.name),
                cause=("{} of spend went to {}s returning less than they cost."
                       .format(_money(lost_spend), label) if losing else
                       "Every ranked {} returned more than it cost.".format(label)),
                evidence="ROAS (revenue ÷ spend, from totals) by {}: {}.".format(label, detail),
                action="1. Move budget in steps from the lowest-ROAS {} to the highest and "
                       "re-measure after each step  2. Check that revenue is attributed the "
                       "same way for every {} before comparing  3. Keep a floor on "
                       "awareness channels judged on reach, not revenue".format(label, label),
                impact="Returns fall as spend rises, so {:.2f} is the most the next unit "
                       "spent on '{}' can earn, not a promise.".format(best.roas, best.name),
                severity="high" if losing and lost_spend / t.spend >= 0.2 else "warning"
                if losing else "info",
                category="marketing_roas"))

    # Cost per conversion and whether conversion rates truly differ.
    if c.conversions:
        test = _channel_conversion_test(elig) if c.clicks else None
        ranked = sorted([r for r in elig if np.isfinite(r.cpa)], key=lambda r: r.cpa)
        if len(ranked) >= 2:
            best, worst = ranked[0], ranked[-1]
            shown = ranked if len(ranked) <= 6 else ranked[:3] + ranked[-3:]
            line = " · ".join("{}: {} per conversion".format(r.name, _money(r.cpa))
                              for r in shown)
            if len(ranked) > 6:
                line += ". The lowest and highest three of {} are shown".format(len(ranked))
            sig = ""
            if test:
                _, p_all, adj = test
                sig = (" Conversion rates differ by more than chance (chi-square "
                       "p{}).".format("<0.001" if p_all < 0.001 else "={:.3f}".format(p_all))
                   if p_all < 0.05 else
                       " The conversion rates do not differ by more than chance "
                       "(chi-square p={:.2f}); the cost gap comes from cost per click."
                       .format(p_all))
            findings.append("Cost per conversion by {}: {}.{}".format(label, line, sig))
            if worst.cpa >= 2 * best.cpa:
                insights.append(build_insight(
                    title="'{}' pays {:.1f}x as much per conversion as '{}'".format(
                        worst.name, worst.cpa / best.cpa, best.name),
                    problem="A conversion cost {} through '{}' and {} through '{}'.".format(
                        _money(worst.cpa), worst.name, _money(best.cpa), best.name),
                    cause=_funnel_cause(worst, best, c),
                    evidence="Spend ÷ conversions from totals.{}".format(sig),
                    action="1. Cap '{}' bids or budget at the cost per conversion the "
                           "business can afford  2. Fix the funnel stage named above before "
                           "adding spend  3. Re-test after two weeks".format(worst.name),
                    impact="At the overall cost per conversion ({}), its {} of spend would "
                           "have bought about {} conversions instead of {}.".format(
                               _money(t.cpa), _money(worst.spend),
                               _money(worst.spend / t.cpa), _money(worst.conversions)),
                    severity="warning", category="marketing_cpa"))

    # Spend with nothing to show.
    if c.conversions:
        dead = [r for r in rows if r.spend > 0 and r.conversions == 0]
        dead_spend = sum(r.spend for r in dead)
        if dead and dead_spend / t.spend >= 0.03:
            risks.append("{} with spend and no conversions: {} — {} ({:.0f}% of spend).".format(
                label.capitalize() + "s",
                ", ".join(r.name for r in sorted(dead, key=lambda r: -r.spend)[:5]),
                _money(dead_spend), dead_spend / t.spend * 100))


def _funnel_cause(worst: GroupRow, best: GroupRow, c: MarketingColumns) -> str:
    """Which funnel stage makes the expensive group expensive."""
    reasons = []
    if c.impressions and c.clicks and np.isfinite(worst.ctr) and np.isfinite(best.ctr) \
            and worst.ctr < 0.7 * best.ctr:
        reasons.append("its click-through rate is {} against {} (targeting or creative)"
                       .format(_pct(worst.ctr, 2), _pct(best.ctr, 2)))
    if c.clicks and np.isfinite(worst.cvr) and np.isfinite(best.cvr) and worst.cvr < 0.7 * best.cvr:
        reasons.append("fewer of its clicks convert, {} against {} (landing page or "
                       "offer)".format(_pct(worst.cvr, 2), _pct(best.cvr, 2)))
    if c.clicks and np.isfinite(worst.cpc) and np.isfinite(best.cpc) and worst.cpc > 1.3 * best.cpc:
        reasons.append("each click costs more, {} against {} (bidding or auction "
                       "competition)".format(_money(worst.cpc), _money(best.cpc)))
    if not reasons:
        return "No single funnel stage explains the gap; the costs compound across stages."
    return "Of the funnel stages: " + " · ".join(reasons) + "."


def trend_insight(df: pd.DataFrame, rep: MarketingReport, insights: List,
                  findings: List, risks: List) -> None:
    """Latest complete month against the one before, for cost per
    conversion and ROAS. Months only partly covered are left out."""
    c = rep.columns
    if not c.date or not (c.conversions or c.revenue):
        return
    d = pd.to_datetime(df[c.date], errors="coerce")
    if d.notna().sum() < 20:
        return
    month = d.dt.to_period("M")
    last = d.max()
    complete = month.max() if (month.max().end_time.normalize() - last.normalize()).days <= 3 \
        else month.max() - 1
    first = d.min()
    start_ok = month.min() if first.day <= 4 else month.min() + 1
    months = [m for m in sorted(month.dropna().unique()) if start_ok <= m <= complete]
    if len(months) < 2:
        return
    cur, prev = months[-1], months[-2]
    rc, rp = _row(str(cur), df[month == cur], c), _row(str(prev), df[month == prev], c)
    lines = []
    if c.conversions and np.isfinite(rc.cpa) and np.isfinite(rp.cpa):
        ch = (rc.cpa - rp.cpa) / rp.cpa * 100
        lines.append("cost per conversion {} ({:+.0f}% on {})".format(
            _money(rc.cpa), ch, prev.strftime("%b %Y")))
        if ch >= 20:
            risks.append("Cost per conversion rose {:.0f}% in {} against {}.".format(
                ch, cur.strftime("%b %Y"), prev.strftime("%b %Y")))
    if c.revenue and np.isfinite(rc.roas) and np.isfinite(rp.roas):
        lines.append("ROAS {:.2f} (from {:.2f})".format(rc.roas, rp.roas))
    if lines:
        findings.append("In {}, the latest complete month: {}.".format(
            cur.strftime("%b %Y"), " · ".join(lines)))


def run_marketing(df: pd.DataFrame, insights: List, findings: List, risks: List,
                  opps: List) -> Optional[MarketingReport]:
    try:
        rep = analyse(df)
    except Exception:
        logger.warning("marketing analysis failed", exc_info=True)
        return None
    if rep is None:
        return None
    findings.append(_summary_finding(rep))
    if rep.by_channel:
        channel_insights(rep, "channel", rep.by_channel, insights, findings, risks, opps)
    if rep.by_campaign:
        channel_insights(rep, "campaign", rep.by_campaign, insights, findings, risks, opps)
    try:
        trend_insight(df, rep, insights, findings, risks)
    except Exception:
        logger.warning("marketing trend failed", exc_info=True)
    return rep


def _insights_marketing(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    insights, findings, risks, opps = [], [], [], []
    run_marketing(df, insights, findings, risks, opps)
    return {"insights": insights, "findings": findings, "risks": risks,
            "opportunities": opps, "actions": []}
