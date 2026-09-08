"""
engines/domains/insurance.py — policies, claims and underwriting.

The one number that decides whether an insurance book is working is the
loss ratio: claims paid over premium earned. Everything else in a
portfolio review hangs off it, and no other domain in this app computes
anything like it, which is why this is a domain rather than a variant of
finance.

What an underwriting or claims lead asks:

**Is the book profitable before expenses?** Loss ratio above 100% means
the policies are paying out more than they bring in, and no amount of
volume fixes that.

**Which segment is carrying the loss?** A book at 78% overall with one
product at 140% is not a pricing problem across the board; it is one
product mispriced.

**How concentrated is the exposure?** A handful of claims carrying most
of the paid value is a reinsurance and reserving question, not a
frequency one.

**Is it frequency or severity?** More claims and bigger claims need
opposite responses — one is underwriting selection, the other is limits
and reserving — so the engine separates them rather than reporting
"claims are up".

Nothing here decides an individual claim or price. This is portfolio
analysis; an automated decision about one policyholder is a regulated
act with obligations this product does not carry.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from app.engines.domains.base import Insight, build_insight
from app.engines.domains._common import (
    benchmark_note, binary_rate, find_col, find_measure, fmt,
    grouping_columns,
)

logger = logging.getLogger(__name__)

# Above this the book is paying out more than it earns before a single
# expense is counted.
LOSS_RATIO_UNPROFITABLE = 100.0
# Above this it is not yet losing money but has no room for expenses.
LOSS_RATIO_CONCERN = 75.0
# A segment this much worse than the book is mispriced, not unlucky.
SEGMENT_LOSS_RATIO_GAP = 1.4
# Paid value this concentrated in the largest claims is a reserving and
# reinsurance question rather than a frequency one.
SEVERITY_CONCENTRATION = 50.0


def _segment_col(df: pd.DataFrame):
    """The dimension an underwriter re-prices: product, region, channel."""
    for kw in ("product", "policytype", "coverage", "line", "segment",
               "region", "channel", "broker", "vehicletype", "class"):
        for c in grouping_columns(df):
            if kw in c.lower().replace("_", "").replace(" ", ""):
                return c
    groups = grouping_columns(df)
    return groups[0] if groups else None


def _insights_insurance(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    findings, risks, opps, actions = [], [], [], []
    insights: List[Insight] = []

    premium = find_measure(df, ["premium", "annualpremium", "grosspremium",
                                "writtenpremium", "policyvalue"])
    claim_amount = find_measure(df, ["claimamount", "claimvalue", "paid",
                                     "lossamount", "incurred", "settlement"])
    claim_count = find_measure(df, ["claimcount", "numclaims", "claims"],
                               exclude=["claimamount", "claimvalue"])
    sum_insured = find_measure(df, ["suminsured", "coveramount", "limit",
                                    "insuredvalue"])
    segment = _segment_col(df)
    claimed = find_col(df, ["claimed", "hasclaim", "claimflag", "isclaim"])
    lapsed = find_col(df, ["lapsed", "cancelled", "canceled", "renewed",
                           "churned"])

    # ── Loss ratio: the number the book turns on ──────────
    loss_ratio = None
    if premium and claim_amount:
        try:
            total_premium = float(df[premium].sum())
            total_claims = float(df[claim_amount].sum())
            if total_premium > 0:
                loss_ratio = total_claims / total_premium * 100
        except Exception:
            logger.debug("insurance loss ratio failed", exc_info=True)

    if loss_ratio is not None:
        severity = ("critical" if loss_ratio >= LOSS_RATIO_UNPROFITABLE else
                    "high" if loss_ratio >= LOSS_RATIO_CONCERN else "positive")
        note = benchmark_note("insurance", "loss_ratio", loss_ratio)
        findings.append(
            "Loss ratio is {:.1f}% — {} in claims against {} in "
            "premium.{}".format(
                loss_ratio, fmt(float(df[claim_amount].sum())),
                fmt(float(df[premium].sum())), (" " + note) if note else ""))
        if loss_ratio >= LOSS_RATIO_UNPROFITABLE:
            risks.append(
                "At a {:.1f}% loss ratio the book pays out more than it "
                "earns before any expense is counted. Growing it makes the "
                "loss larger, not smaller.".format(loss_ratio))
        elif loss_ratio >= LOSS_RATIO_CONCERN:
            risks.append(
                "A {:.1f}% loss ratio leaves {:.1f} points for acquisition, "
                "administration and profit combined — thin enough that a "
                "single bad quarter turns it negative.".format(
                    loss_ratio, 100 - loss_ratio))
        insights.append(build_insight(
            title="Loss Ratio at {:.1f}%".format(loss_ratio),
            problem="Claims run at {:.1f}% of premium across {:,} "
                    "policies".format(loss_ratio, len(df)),
            cause="Pricing against the risk actually written, claims "
                  "severity, or selection — the segment breakdown separates "
                  "which",
            evidence="{} claims against {} premium over {:,} records{}".format(
                fmt(float(df[claim_amount].sum())),
                fmt(float(df[premium].sum())), len(df),
                (" — " + note) if note else ""),
            action="1. Split the ratio by product and region  2. Take the "
                   "worst segment and separate frequency from severity  "
                   "3. Re-rate or withdraw on that segment alone  "
                   "4. Re-measure over a full policy year",
            impact="Each point of loss ratio is about {} across this "
                   "book.".format(fmt(float(df[premium].sum()) / 100)),
            severity=severity, category="insurance_profitability",
        ))

    # ── Which segment carries it ──────────────────────────
    if segment and premium and claim_amount and loss_ratio:
        try:
            grouped = df.groupby(segment).agg(
                _prem=(premium, "sum"), _clm=(claim_amount, "sum"),
                _n=(premium, "size"))
            grouped = grouped[(grouped["_prem"] > 0) & (grouped["_n"] >= 20)]
            if len(grouped) >= 2:
                grouped["_lr"] = grouped["_clm"] / grouped["_prem"] * 100
                worst = grouped["_lr"].idxmax()
                worst_lr = float(grouped.loc[worst, "_lr"])
                best = grouped["_lr"].idxmin()
                best_lr = float(grouped.loc[best, "_lr"])
                if worst_lr >= loss_ratio * SEGMENT_LOSS_RATIO_GAP:
                    findings.append(
                        "'{}' runs at a {:.1f}% loss ratio against {:.1f}% "
                        "for the book and {:.1f}% for '{}'.".format(
                            worst, worst_lr, loss_ratio, best_lr, best))
                    opps.append(
                        "'{}' is the segment to re-rate: bringing it to the "
                        "book average would be about {} of claims cost on "
                        "{:,} policies.".format(
                            worst,
                            fmt(float(grouped.loc[worst, "_prem"]) *
                                (worst_lr - loss_ratio) / 100),
                            int(grouped.loc[worst, "_n"])))
                    insights.append(build_insight(
                        title="'{}' Runs at {:.1f}% Loss Ratio".format(
                            worst, worst_lr),
                        problem="One {} sits {:.1f} points above the book "
                                "at {:.1f}%".format(
                                    segment, worst_lr - loss_ratio, worst_lr),
                        cause="Mispricing against the risk in that segment, "
                              "adverse selection, or a claims pattern that "
                              "the rating basis does not see",
                        evidence="{:,} policies, {} premium, {} claims — "
                                 "against {:.1f}% for the book".format(
                                     int(grouped.loc[worst, "_n"]),
                                     fmt(float(grouped.loc[worst, "_prem"])),
                                     fmt(float(grouped.loc[worst, "_clm"])),
                                     loss_ratio),
                        action="1. Separate frequency from severity in that "
                               "segment  2. Check whether the rating factors "
                               "capture what is driving it  3. Re-rate, "
                               "restrict or withdraw  4. Hold the comparison "
                               "for a full policy year",
                        impact="About {} of claims cost above what the book "
                               "average would predict.".format(
                                   fmt(float(grouped.loc[worst, "_prem"]) *
                                       (worst_lr - loss_ratio) / 100)),
                        severity="critical" if worst_lr >= 100 else "high",
                        category="insurance_segment",
                    ))
        except Exception:
            logger.debug("insurance segment loss ratio failed", exc_info=True)

    # ── Frequency or severity? ────────────────────────────
    if claimed:
        freq = binary_rate(df[claimed])
        if freq is not None:
            findings.append(
                "{:.1f}% of policies made a claim.".format(freq))
            if claim_amount:
                try:
                    claimants = df[df[claim_amount] > 0][claim_amount]
                    if len(claimants) >= 20:
                        findings.append(
                            "Average claim among those who claimed is {}, "
                            "median {} — the gap between them is the "
                            "severity tail.".format(
                                fmt(float(claimants.mean())),
                                fmt(float(claimants.median()))))
                        actions.append(
                            "Separate frequency ({:.1f}% of policies) from "
                            "severity (mean claim {}) before responding — "
                            "selection fixes one, limits and reserving fix "
                            "the other.".format(
                                freq, fmt(float(claimants.mean()))))
                except Exception:
                    logger.debug("insurance severity failed", exc_info=True)

    # ── Exposure concentration ────────────────────────────
    if claim_amount:
        try:
            paid = df[df[claim_amount] > 0][claim_amount].sort_values(
                ascending=False)
            if len(paid) >= 20:
                top_decile = max(1, len(paid) // 10)
                share = float(paid.head(top_decile).sum()) / \
                    float(paid.sum()) * 100
                if share >= SEVERITY_CONCENTRATION:
                    findings.append(
                        "The largest {:.0f}% of claims carry {:.0f}% of the "
                        "paid value.".format(
                            top_decile / len(paid) * 100, share))
                    risks.append(
                        "{:.0f}% of paid value sits in the largest {:,} "
                        "claims. That is a reserving and reinsurance "
                        "question, and it will not respond to anything "
                        "aimed at claim frequency.".format(
                            share, top_decile))
                    insights.append(build_insight(
                        title="{:.0f}% of Paid Value in the Largest {:,} "
                              "Claims".format(share, top_decile),
                        problem="Claims cost is concentrated in a small "
                                "number of large losses",
                        cause="Severity tail — limits, aggregation, or a "
                              "small number of events rather than a broad "
                              "deterioration",
                        evidence="Top {:,} of {:,} paid claims carry {:.0f}% "
                                 "of value".format(
                                     top_decile, len(paid), share),
                        action="1. Read the largest claims individually  "
                               "2. Check whether they share a cause or a "
                               "segment  3. Review limits and reinsurance "
                               "attachment against them  4. Reserve on the "
                               "tail, not the mean",
                        impact="{} sits in those claims.".format(
                            fmt(float(paid.head(top_decile).sum()))),
                        severity="high", category="insurance_exposure",
                    ))
        except Exception:
            logger.debug("insurance concentration failed", exc_info=True)

    if lapsed:
        rate = binary_rate(df[lapsed])
        if rate is not None:
            findings.append("{:.1f}% of policies lapsed or cancelled.".format(
                rate))

    if sum_insured and premium:
        actions.append(
            "'{}' and '{}' are both present — premium as a rate on sum "
            "insured is the measure that makes segments comparable, and it "
            "can be derived here.".format(premium, sum_insured))

    if not findings and not insights:
        logger.info("insurance engine found no domain signal in %d columns",
                    len(df.columns))

    return {"findings": findings, "risks": risks, "opportunities": opps,
            "actions": actions, "insights": insights}
