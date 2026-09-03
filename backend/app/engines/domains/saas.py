"""
engines/domains/saas.py — SaaS & subscription analytics.

Subscription data used to land on the HR engine: `tenure` and a churn
flag were enough for HR to win detection at 0.07 confidence, and because
attrition ran for anything detected as HR, the report came back with an
"employee attrition rate" computed from customer churn.

Churn here is CUSTOMER churn. The distinction is the whole reason this
engine exists separately.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from app.engines.domains.base import Insight, build_insight
from app.engines.domains._common import (
    benchmark_note, binary_rate, concentration, find_measure, fmt,
    grouping_columns, segment_gap,
)

logger = logging.getLogger(__name__)

# Revenue concentrated above this in the top decile is a dependency risk.
TOP_ACCOUNT_CONCENTRATION = 40.0
# A plan churning this many times the overall rate is the leak to fix.
PLAN_CHURN_RATIO = 1.5


def _plan_col(df: pd.DataFrame):
    for kw in ("plan", "tier", "package", "segment", "subscription"):
        for c in grouping_columns(df):
            if kw in c.lower().replace("_", ""):
                return c
    groups = grouping_columns(df)
    return groups[0] if groups else None


def _insights_saas(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    findings, risks, opps, actions = [], [], [], []
    insights: List[Insight] = []

    mrr = find_measure(df, ["mrr", "monthlyrecurring", "monthlyrevenue",
                            "monthlycharges"])
    arr = find_measure(df, ["arr", "annualrecurring", "annualrevenue"])
    revenue = mrr or arr or find_measure(df, ["revenue", "totalcharges"])
    expansion = find_measure(df, ["expansion", "upsell", "upgraderevenue"])
    seats = find_measure(df, ["seats", "licenses", "activeusers", "users"])
    nps = find_measure(df, ["nps", "netpromoter", "csat"])
    tenure = find_measure(df, ["tenuremonths", "tenure", "monthssince",
                               "subscriptionage", "customerage"])
    plan = _plan_col(df)

    churn_col = None
    for c in df.columns:
        n = c.lower().replace("_", "")
        if any(k in n for k in ("churn", "cancelled", "canceled", "lost",
                                "unsubscribed")):
            if binary_rate(df[c]) is not None:
                churn_col = c
                break

    # ── Recurring revenue position ────────────────────────
    if revenue:
        total = float(df[revenue].sum())
        n_accounts = len(df)
        arpa = total / n_accounts if n_accounts else 0
        findings.append(
            "{:,} accounts carry {} of recurring revenue — {} per account "
            "on average.".format(n_accounts, fmt(total), fmt(arpa)))

    # ── Churn ─────────────────────────────────────────────
    if churn_col:
        rate = binary_rate(df[churn_col])
        if rate is not None:
            note = benchmark_note("saas", churn_col, rate)
            severity = ("critical" if rate > 15 else
                        "high" if rate > 8 else
                        "warning" if rate > 4 else "positive")
            lost_rev = None
            if revenue:
                try:
                    mask = df[churn_col].astype(str).str.strip().str.lower()
                    positives = {"1", "1.0", "yes", "y", "true", "t",
                                 "churned", "cancelled", "canceled", "lost"}
                    lost_rev = float(df.loc[mask.isin(positives), revenue].sum())
                except Exception:
                    logger.debug("saas churn revenue failed", exc_info=True)

            findings.append(
                "Customer churn is {:.1f}% of the base.{}".format(
                    rate, " " + note if note else ""))
            if rate > 8:
                risks.append(
                    "Churn at {:.1f}% means the base halves in under "
                    "{:.0f} periods without replacement.".format(
                        rate, max(1, round(69 / max(rate, 0.1)))))
            insights.append(build_insight(
                title="Customer Churn at {:.1f}%".format(rate),
                problem="{:.1f}% of accounts have churned{}".format(
                    rate,
                    ", carrying {} of recurring revenue".format(fmt(lost_rev))
                    if lost_rev else ""),
                cause="Onboarding gaps, unrealised value, or a pricing tier "
                      "mismatched to the segment buying it",
                evidence="{:,} of {:,} accounts churned.{}".format(
                    int(round(rate / 100 * len(df))), len(df),
                    " " + note if note else ""),
                action="1. Segment churn by plan and tenure  2. Interview "
                       "the last 20 churned accounts  3. Fix the largest "
                       "single cause before broadening",
                impact="Each point of churn recovered is worth about {} of "
                       "retained recurring revenue".format(
                           fmt(float(df[revenue].sum()) / 100)
                           if revenue else "one point of the base"),
                severity=severity, category="saas_retention",
            ))

    # ── Which plan leaks ──────────────────────────────────
    if churn_col and plan:
        try:
            work = df[[plan, churn_col]].dropna()
            rates = work.groupby(plan)[churn_col].apply(binary_rate).dropna()
            sizes = work.groupby(plan).size()
            rates = rates[sizes >= 5]
            if len(rates) >= 2:
                overall = binary_rate(df[churn_col]) or 0
                worst = rates.idxmax()
                worst_rate = float(rates.max())
                if overall > 0 and worst_rate >= overall * PLAN_CHURN_RATIO:
                    risks.append(
                        "'{}' churns at {:.1f}% against {:.1f}% overall — "
                        "the loss is concentrated, not spread.".format(
                            worst, worst_rate, overall))
                    insights.append(build_insight(
                        title="'{}' Churns at {:.1f}%, {:.1f}x the Base".format(
                            worst, worst_rate, worst_rate / max(overall, .1)),
                        problem="Churn concentrates in '{}' at {:.1f}% "
                                "against {:.1f}% overall".format(
                                    worst, worst_rate, overall),
                        cause="This tier's price, feature set or support "
                              "level does not match what the segment "
                              "buying it expects",
                        evidence="'{}' n={:,}, churn {:.1f}%; "
                                 "overall {:.1f}%".format(
                                     worst, int(sizes.loc[worst]),
                                     worst_rate, overall),
                        action="1. Review '{}' packaging and price  "
                               "2. Add a guided onboarding path for it  "
                               "3. Re-measure after one renewal cycle".format(
                                   worst),
                        impact="Bringing '{}' to base churn removes about "
                               "{:.0f}% of total losses".format(
                                   worst,
                                   (worst_rate - overall) / worst_rate * 100),
                        severity="critical", category="saas_retention",
                    ))
        except Exception:
            logger.debug("saas plan churn failed", exc_info=True)

    # ── Revenue concentration ─────────────────────────────
    if revenue and plan:
        conc = concentration(df, plan, revenue, top_n=1)
        if conc and conc["top_share"] >= TOP_ACCOUNT_CONCENTRATION \
                and conc["n_groups"] > 1:
            risks.append(
                "'{}' accounts for {:.0f}% of recurring revenue across {} "
                "tiers — revenue is dependent on one segment.".format(
                    conc["top_names"][0], conc["top_share"],
                    conc["n_groups"]))

    # ── Expansion opportunity ─────────────────────────────
    if expansion and revenue:
        try:
            exp_total = float(df[expansion].sum())
            base_total = float(df[revenue].sum())
            if base_total > 0:
                exp_pct = exp_total / base_total * 100
                findings.append(
                    "Expansion revenue is {} — {:.1f}% of the recurring "
                    "base.".format(fmt(exp_total), exp_pct))
                if exp_pct < 10:
                    opps.append(
                        "Expansion is only {:.1f}% of base revenue. Growing "
                        "existing accounts is the cheapest revenue available "
                        "and is currently under-worked.".format(exp_pct))
                    insights.append(build_insight(
                        title="Expansion Revenue at {:.1f}% of Base".format(
                            exp_pct),
                        problem="Only {} of expansion against {} of base "
                                "recurring revenue".format(
                                    fmt(exp_total), fmt(base_total)),
                        cause="No systematic upsell motion, or usage signals "
                              "not being surfaced to the account team",
                        evidence="Expansion {} vs base {} ({:.1f}%)".format(
                            fmt(exp_total), fmt(base_total), exp_pct),
                        action="1. Identify accounts at seat or usage limits  "
                               "2. Trigger an upgrade conversation at 80% of "
                               "limit  3. Measure expansion as its own metric",
                        impact="Reaching a 15% expansion rate adds about {} "
                               "without a single new customer".format(
                                   fmt(base_total * 0.15 - exp_total)),
                        severity="high", category="saas_growth",
                    ))
        except Exception:
            logger.debug("saas expansion failed", exc_info=True)

    # ── Engagement ────────────────────────────────────────
    if seats and churn_col:
        gap = segment_gap(df, churn_col, seats, agg="mean")
        if gap and gap["ratio"] and gap["ratio"] >= 1.3:
            findings.append(
                "Accounts differ on '{}' by {:.1f}x between churn states — "
                "usage is a visible leading indicator here.".format(
                    seats, gap["ratio"]))
            opps.append(
                "'{}' separates retained from churned accounts. It can drive "
                "an at-risk alert before renewal, not after.".format(seats))

    if nps is not None:
        try:
            mean_nps = float(df[nps].mean())
            findings.append("Average {} is {:.1f}.".format(nps, mean_nps))
        except Exception:
            logger.debug("saas nps failed", exc_info=True)

    if not findings and not insights:
        logger.info("saas engine found no domain signal in %d columns",
                    len(df.columns))

    return {"findings": findings, "risks": risks, "opportunities": opps,
            "actions": actions, "insights": insights}
