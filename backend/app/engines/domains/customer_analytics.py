"""
domains/customer_analytics.py — the customer-level analyses a client
expects to see in an e-commerce or transactional review, and previously
did not get: cohort retention, RFM segmentation, and revenue
concentration.

These are deliberately separate from ecommerce.py, which works at the
product and rating level. Everything here needs a customer identifier,
and several analyses need an order date as well; each returns silently
when its inputs are absent rather than substituting a weaker proxy.

Nothing here reports a figure it cannot support. Retention needs enough
customers per cohort to be a rate rather than an anecdote; concentration
needs enough customers for a share to mean anything. The thresholds are
stated at each site.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from app.engines.domains.base import build_insight
from app.services.dtypes import is_text_dtype

logger = logging.getLogger(__name__)

# Below this a "rate" is an anecdote about a handful of people.
MIN_COHORT_CUSTOMERS = 25
MIN_CUSTOMERS = 50

_CUSTOMER_KW = ("customer_id", "customerid", "customer_key", "customer",
                "buyer_id", "user_id", "client_id", "account_id", "member_id")
_DATE_KW = ("order_date", "date", "created", "purchase", "timestamp",
            "invoice_date", "transaction_date")
_VALUE_KW = ("revenue", "amount", "sales", "total", "price", "value",
             "spend", "net_sales")


def _norm(col: str) -> str:
    return str(col).lower().replace(" ", "_").replace("-", "_")


def find_customer_col(df: pd.DataFrame) -> Optional[str]:
    """The column identifying who bought, if there is one.

    Matching is by name only. One row per value is not disqualifying — an
    already-aggregated customer table looks exactly like that, and
    concentration analysis is perfectly valid on it. What that grain does
    invalidate is retention, which is guarded at its own call site.
    """
    for c in df.columns:
        if not any(k in _norm(c) for k in _CUSTOMER_KW):
            continue
        if len(df[c].dropna()) < MIN_CUSTOMERS:
            continue
        return c
    return None


def find_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if not any(k in _norm(c) for k in _DATE_KW):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
        if is_text_dtype(df[c]):
            try:
                parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
            except (ValueError, TypeError):
                continue
            if parsed.notna().mean() > 0.9:
                return c
    return None


def find_value_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if (any(k in _norm(c) for k in _VALUE_KW)
                and pd.api.types.is_numeric_dtype(df[c])
                and not any(x in _norm(c) for x in ("_id", "count", "qty", "quantity"))):
            return c
    return None


def _as_datetime(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce", format="mixed")


# ══════════════════════════════════════════════════════════
#  COHORT RETENTION
# ══════════════════════════════════════════════════════════

def cohort_retention(df: pd.DataFrame, insights: List, findings: List,
                     risks: List, opps: List) -> None:
    """Do customers acquired in one month come back in later months?

    A blended repeat-purchase rate answers "do customers ever return"; it
    cannot say whether returns are improving or decaying, because a good
    old cohort and a bad new one average to something unremarkable.
    Grouping by acquisition month separates the two.

    Month 1 retention — the share of a cohort that ordered again in the
    month after acquisition — is the figure reported, because it is the
    one comparable across cohorts of different ages.
    """
    cust = find_customer_col(df)
    date = find_date_col(df)
    if not (cust and date):
        return
    try:
        work = df[[cust, date]].copy()
        work[date] = _as_datetime(work[date])
        work = work.dropna()
        n_customers = work[cust].nunique()
        if n_customers < MIN_CUSTOMERS:
            return
        # One row per customer means this is an aggregated customer table,
        # not an order log. Every cohort would return 0% — an artefact of
        # the grain, printed as though the customers had all churned.
        if len(work) < n_customers * 1.1:
            return

        work["_month"] = work[date].dt.to_period("M")
        first = work.groupby(cust)["_month"].min().rename("_cohort")
        work = work.join(first, on=cust)
        work["_age"] = ((work["_month"] - work["_cohort"])
                        .apply(lambda p: p.n if p is not pd.NaT else np.nan))

        cohort_size = work.groupby("_cohort")[cust].nunique()
        # Cohorts too small to give a rate, and the final cohort, which has
        # had no chance to return yet.
        usable = cohort_size[cohort_size >= MIN_COHORT_CUSTOMERS]
        if len(usable) < 2:
            return
        usable = usable.iloc[:-1] if len(usable) > 2 else usable

        m1 = {}
        for cohort, size in usable.items():
            returned = work[(work["_cohort"] == cohort) & (work["_age"] == 1)][cust].nunique()
            m1[cohort] = returned / size * 100

        if len(m1) < 2:
            return
        rates = pd.Series(m1).sort_index()
        overall = float(rates.mean())
        newest, oldest = float(rates.iloc[-1]), float(rates.iloc[0])
        drift = newest - oldest

        findings.append(
            "Month-1 retention averages {:.1f}% across {} monthly cohorts "
            "({:,} customers), ranging {:.1f}% to {:.1f}%.".format(
                overall, len(rates), int(usable.sum()),
                float(rates.min()), float(rates.max())))

        # A cohort trend is worth reporting when it is large enough not to
        # be month-to-month noise.
        if abs(drift) >= 5 and len(rates) >= 3:
            direction = "declined" if drift < 0 else "improved"
            sev = ("high" if drift <= -10 else "warning" if drift < 0 else "positive")
            detail = "; ".join(
                "{}: {:.1f}%".format(str(k), v) for k, v in rates.items())
            if drift < 0:
                risks.append(
                    "Month-1 retention has {} {:.1f} points across the cohorts in "
                    "this data, from {:.1f}% ({}) to {:.1f}% ({}). Later cohorts "
                    "are returning less often than earlier ones.".format(
                        direction, abs(drift), oldest, str(rates.index[0]),
                        newest, str(rates.index[-1])))
            else:
                opps.append(
                    "Month-1 retention has {} {:.1f} points across cohorts, from "
                    "{:.1f}% to {:.1f}% — whatever changed for the later cohorts "
                    "is worth identifying and holding.".format(
                        direction, abs(drift), oldest, newest))

            insights.append(build_insight(
                title="Month-1 retention {} from {:.1f}% to {:.1f}%".format(
                    direction, oldest, newest),
                problem="Customers acquired in {} returned the following month "
                        "{:.1f}% of the time; for {} it was {:.1f}%.".format(
                            str(rates.index[0]), oldest,
                            str(rates.index[-1]), newest),
                cause="A cohort trend of this size normally follows a change in "
                      "acquisition mix, onboarding, or the offer itself. Which one "
                      "applies is not established by this data — it needs the "
                      "channel or campaign behind each cohort.",
                evidence="Month-1 retention by cohort — {} (cohorts of "
                         "{}+ customers only)".format(detail, MIN_COHORT_CUSTOMERS),
                action="1. Split the weakest and strongest cohorts by acquisition "
                       "channel  2. Compare first-order contents between them  "
                       "3. Set the stronger cohort's rate as the retention target",
                impact="Month-1 retention compounds: a {:.1f} point difference "
                       "persists through every later month of a cohort's life.".format(
                           abs(drift)),
                severity=sev, category="ecommerce_retention",
            ))
    except Exception:
        logger.warning("cohort retention failed", exc_info=True)


# ══════════════════════════════════════════════════════════
#  RFM SEGMENTATION
# ══════════════════════════════════════════════════════════

def rfm_segments(df: pd.DataFrame, insights: List, findings: List,
                 risks: List, opps: List) -> None:
    """Score customers on recency, frequency and monetary value.

    Scoring is by quintile within this dataset, so the segments are
    relative to the client's own book rather than to an outside standard
    that may not apply to their category. The segment that matters
    commercially is the one that used to buy and has stopped — it is
    both the largest recoverable value and invisible in any average.
    """
    cust = find_customer_col(df)
    date = find_date_col(df)
    value = find_value_col(df)
    if not (cust and date and value):
        return
    try:
        work = df[[cust, date, value]].copy()
        work[date] = _as_datetime(work[date])
        work = work.dropna()
        work = work[work[value] > 0]
        if work[cust].nunique() < MIN_CUSTOMERS:
            return

        asof = work[date].max()
        agg = work.groupby(cust).agg(
            recency=(date, lambda s: (asof - s.max()).days),
            frequency=(cust, "size"),
            monetary=(value, "sum"),
        )
        if len(agg) < MIN_CUSTOMERS:
            return

        # Quintiles need enough distinct values to cut; frequency in
        # particular is often 1 for most of the book.
        def _score(series, ascending=True):
            try:
                ranked = series.rank(method="first", ascending=ascending)
                return pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5]).astype(int)
            except ValueError:
                return pd.Series(3, index=series.index)

        agg["r"] = _score(agg["recency"], ascending=False)   # recent = high
        agg["f"] = _score(agg["frequency"])
        agg["m"] = _score(agg["monetary"])

        champions = agg[(agg["r"] >= 4) & (agg["f"] >= 4) & (agg["m"] >= 4)]
        at_risk = agg[(agg["r"] <= 2) & (agg["f"] >= 4)]
        # High-value but lapsed — the recoverable money.
        lapsed_value = float(at_risk["monetary"].sum())
        total_value = float(agg["monetary"].sum())

        findings.append(
            "RFM segmentation of {:,} customers: {:,} champions (recent, frequent, "
            "high value) contributing {:.0f}% of revenue; {:,} previously frequent "
            "customers have not returned recently.".format(
                len(agg), len(champions),
                float(champions["monetary"].sum()) / total_value * 100 if total_value else 0,
                len(at_risk)))

        if len(at_risk) >= 10 and total_value > 0:
            share = lapsed_value / total_value * 100
            median_gap = float(at_risk["recency"].median())
            insights.append(build_insight(
                title="{:,} lapsed repeat customers hold {:.0f}% of revenue history".format(
                    len(at_risk), share),
                problem="{:,} customers who bought {:.0f}+ times have not ordered for "
                        "a median of {:.0f} days.".format(
                            len(at_risk), float(at_risk["frequency"].median()), median_gap),
                cause="Lapse after repeat purchasing is usually a service, stock or "
                      "competitive event rather than a pricing one, but this data "
                      "records orders only — the reason is not in it.",
                evidence="Recency/frequency/monetary quintiles over {:,} customers; "
                         "at-risk = recency quintile 1-2 with frequency quintile 4-5. "
                         "Their historical revenue: {:,.0f} of {:,.0f} "
                         "({:.1f}%).".format(len(agg), lapsed_value, total_value, share),
                action="1. Export the at-risk list and check for open service issues  "
                       "2. Contact the top decile by historical value first  "
                       "3. Measure reactivation against a held-back control group",
                impact="Recovering a quarter of this group would return roughly "
                       "{:,.0f} of annualised revenue at their historical rate.".format(
                           lapsed_value * 0.25),
                severity="high" if share > 15 else "warning",
                category="ecommerce_rfm",
            ))
        if len(champions) >= 10:
            opps.append(
                "{:,} champion customers ({:.0f}% of the base) account for {:.0f}% of "
                "revenue — the group to protect first in any pricing or service "
                "change.".format(
                    len(champions), len(champions) / len(agg) * 100,
                    float(champions["monetary"].sum()) / total_value * 100
                    if total_value else 0))
    except Exception:
        logger.warning("RFM segmentation failed", exc_info=True)


# ══════════════════════════════════════════════════════════
#  REVENUE CONCENTRATION
# ══════════════════════════════════════════════════════════

def revenue_concentration(df: pd.DataFrame, insights: List, findings: List,
                          risks: List, opps: List) -> None:
    """How much of the revenue rests on how few customers.

    Reported with a Gini coefficient alongside the top-decile share,
    because the two answer different questions: the share says how
    exposed the business is to a handful of accounts, the Gini says
    whether that is the shape of the whole book or one outlier.
    """
    cust = find_customer_col(df)
    value = find_value_col(df)
    if not (cust and value):
        return
    try:
        per = df.groupby(cust)[value].sum().sort_values(ascending=False)
        per = per[per > 0]
        n = len(per)
        if n < MIN_CUSTOMERS:
            return

        total = float(per.sum())
        top_n = max(1, int(round(n * 0.10)))
        top_share = float(per.iloc[:top_n].sum()) / total * 100

        # Gini over customer revenue: 0 = every customer spends the same,
        # 1 = one customer is the whole book.
        vals = np.sort(per.to_numpy(dtype=float))
        idx = np.arange(1, n + 1)
        gini = float((2 * (idx * vals).sum()) / (n * vals.sum()) - (n + 1) / n)

        findings.append(
            "Revenue concentration: the top 10% of customers ({:,} of {:,}) "
            "account for {:.0f}% of revenue (Gini {:.2f}).".format(
                top_n, n, top_share, gini))

        if top_share >= 50:
            largest = float(per.iloc[0]) / total * 100
            risks.append(
                "{:.0f}% of revenue comes from {:,} customers, and the single "
                "largest is {:.1f}% on its own. Losing the top decile would remove "
                "{:.0f}% of revenue.".format(top_share, top_n, largest, top_share))
            insights.append(build_insight(
                title="Top 10% of customers hold {:.0f}% of revenue".format(top_share),
                problem="{:,} of {:,} customers account for {:.0f}% of revenue; the "
                        "largest single customer is {:.1f}%.".format(
                            top_n, n, top_share, largest),
                cause="Concentration at this level is normal in some categories and "
                      "a dependency in others. Which this is depends on contract "
                      "length and switching cost, neither of which is in this data.",
                evidence="Customer revenue Gini {:.2f} over {:,} customers; top "
                         "decile {:,.0f} of {:,.0f} total.".format(
                             gini, n, float(per.iloc[:top_n].sum()), total),
                action="1. Confirm contract cover on the top decile  2. Track the "
                       "share monthly rather than annually  3. Set a concentration "
                       "ceiling for the sales plan",
                impact="A single top-decile loss would remove roughly {:.1f}% of "
                       "revenue at current mix.".format(top_share / max(top_n, 1)),
                severity="high" if top_share >= 70 else "warning",
                category="ecommerce_concentration",
            ))
        elif gini >= 0.45:
            # Below the concentration threshold but still an uneven book —
            # calling that "evenly spread" would contradict the coefficient
            # printed in the same sentence.
            opps.append(
                "No single account dominates (top decile {:.0f}% of revenue), but "
                "spend per customer is uneven (Gini {:.2f}): most customers are "
                "small, so growth depends on the middle of the book rather than "
                "the top.".format(top_share, gini))
        else:
            opps.append(
                "Revenue is spread across the customer base (top decile {:.0f}%, "
                "Gini {:.2f}) — no single account dominates.".format(top_share, gini))
    except Exception:
        logger.warning("revenue concentration failed", exc_info=True)


def run_customer_analytics(df: pd.DataFrame, insights: List, findings: List,
                           risks: List, opps: List) -> None:
    """All three, in the order a reader would want them."""
    cohort_retention(df, insights, findings, risks, opps)
    rfm_segments(df, insights, findings, risks, opps)
    revenue_concentration(df, insights, findings, risks, opps)
