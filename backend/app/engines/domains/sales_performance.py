"""
domains/sales_performance.py — outcome-level sales analysis: who actually
converts, and how long winning takes compared with losing.

The sales engine could describe revenue and cycle length but had nothing
to say about the two questions a sales director asks first — is the gap
between my best and worst rep real, and do deals we win behave
differently from deals we lose. Its own cycle-length insight ended with
"compare won vs lost deals if outcome data exists"; this module is that
comparison.

Both analyses exist to avoid a specific mistake:

  - Ranking reps by raw win rate. A rep who wins 3 of 4 outranks one who
    wins 60 of 100, and the table is read as a performance league. Each
    rep is instead tested against the pooled rate and the results are
    FDR-corrected, so a twenty-rep team does not produce a "top performer"
    by multiplicity alone.
  - Comparing mean cycle length between won and lost deals. Cycle times
    are right-skewed and a handful of stalled deals move the mean;
    Mann-Whitney compares the distributions and the effect is reported as
    rank-biserial correlation, not as a difference of averages.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.engines.domains.base import build_insight, is_id_column
from app.services.dtypes import is_text_dtype
from app.services.stat_guards import bh_adjust

logger = logging.getLogger(__name__)

# A rep with fewer opportunities than this has no rate worth testing.
MIN_DEALS_PER_REP = 20
# Fewer reps than this is a conversation, not an analysis.
MIN_REPS = 3
MIN_DEALS_PER_GROUP = 15

_WON_TOKENS = {"won", "win", "closed won", "closed-won", "closedwon",
               "success", "successful", "converted", "yes", "true", "1"}
_LOST_TOKENS = {"lost", "loss", "closed lost", "closed-lost", "closedlost",
                "failed", "fail", "not converted", "no", "false", "0"}
_OUTCOME_KW = ("outcome", "status", "stage", "result", "won", "win",
               "deal_status", "opportunity_status", "converted", "closed")
_REP_KW = ("rep", "salesperson", "sales_person", "agent", "owner", "seller",
           "account_manager", "account manager", "employee", "consultant")


def _norm(col: str) -> str:
    return str(col).lower().replace(" ", "_").replace("-", "_")


def find_outcome_col(df: pd.DataFrame) -> Optional[Tuple[str, pd.Series]]:
    """A column recording whether each deal was won, as a boolean series.

    Open or in-progress deals are excluded rather than counted as losses:
    a pipeline that is 60% open would otherwise report a 40% win rate that
    is really a snapshot of timing.
    """
    for c in df.columns:
        if not any(k in _norm(c) for k in _OUTCOME_KW):
            continue
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            return c, s.astype(bool)
        if not is_text_dtype(s):
            continue
        vals = s.dropna().astype(str).str.strip().str.lower()
        if vals.empty:
            continue
        won = vals.isin(_WON_TOKENS)
        lost = vals.isin(_LOST_TOKENS)
        decided = won | lost
        # Needs both outcomes present, and enough decided deals for the
        # remainder to be a real population rather than leftovers.
        if won.any() and lost.any() and decided.mean() >= 0.5:
            outcome = pd.Series(np.nan, index=s.index, dtype="object")
            outcome[won[won].index] = True
            outcome[lost[lost].index] = False
            return c, outcome
    return None


def find_rep_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if not any(k in _norm(c) for k in _REP_KW):
            continue
        if is_id_column(c, df[c]):
            continue
        if not (is_text_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)):
            continue
        if MIN_REPS <= df[c].nunique() <= 200:
            return c
    return None


# ══════════════════════════════════════════════════════════
#  WIN RATE BY REP
# ══════════════════════════════════════════════════════════

def rep_win_rates(df: pd.DataFrame, insights: List, findings: List,
                  risks: List, opps: List) -> None:
    """Which reps convert at a rate the deal counts actually support.

    Each rep's wins are tested against the pooled win rate with an exact
    binomial test, and the p-values are Benjamini-Hochberg corrected. On a
    team of twenty reps all performing identically, an uncorrected scan at
    p < 0.05 produces a "standout" about two-thirds of the time — and that
    name ends up in a client's performance review.
    """
    found = find_outcome_col(df)
    rep_col = find_rep_col(df)
    if not found or not rep_col:
        return
    outcome_col, outcome = found
    if rep_col == outcome_col:
        return
    try:
        work = pd.DataFrame({"rep": df[rep_col], "won": outcome}).dropna()
        work["won"] = work["won"].astype(bool)
        if work.empty:
            return

        counts = work.groupby("rep")["won"].agg(["size", "sum"])
        counts = counts[counts["size"] >= MIN_DEALS_PER_REP]
        if len(counts) < MIN_REPS:
            return

        pooled = float(work["won"].sum()) / len(work)
        if not (0 < pooled < 1):
            return

        tested = []
        for rep, row in counts.iterrows():
            n, k = int(row["size"]), int(row["sum"])
            p = float(scipy_stats.binomtest(k, n, pooled).pvalue)
            tested.append({"rep": str(rep), "n": n, "wins": k,
                           "rate": k / n * 100, "p": p})

        qs = bh_adjust([t["p"] for t in tested])
        for t, q in zip(tested, qs):
            t["q"] = float(q)

        significant = [t for t in tested if t["q"] < 0.05]
        above = sorted([t for t in significant if t["rate"] > pooled * 100],
                       key=lambda t: -t["rate"])
        below = sorted([t for t in significant if t["rate"] < pooled * 100],
                       key=lambda t: t["rate"])

        rates = [t["rate"] for t in tested]
        findings.append(
            "Win rate {:.1f}% overall across {:,} decided deals and {} reps with "
            "{}+ deals each; individual rates range {:.1f}% to {:.1f}%.".format(
                pooled * 100, len(work), len(counts), MIN_DEALS_PER_REP,
                min(rates), max(rates)))

        if not significant:
            findings.append(
                "No rep's win rate differs from the team rate by more than deal "
                "volume explains (Benjamini-Hochberg corrected across {} reps). "
                "The spread between the best and worst rate is consistent with "
                "chance at these deal counts.".format(len(tested)))
            return

        detail = "; ".join(
            "{}: {}/{} = {:.1f}% (q={:.3f})".format(
                t["rep"], t["wins"], t["n"], t["rate"], t["q"])
            for t in (above + below)[:6])

        if below:
            worst = below[0]
            gap = pooled * 100 - worst["rate"]
            risks.append(
                "{} converts {:.1f}% of {} deals against a team rate of {:.1f}% — "
                "a {:.1f} point gap that survives correction for testing {} reps."
                .format(worst["rep"], worst["rate"], worst["n"], pooled * 100,
                        gap, len(tested)))

        if above:
            best = above[0]
            opps.append(
                "{} converts {:.1f}% of {} deals against a team rate of {:.1f}%. "
                "The difference is larger than deal volume explains, so what they "
                "do differently is worth documenting.".format(
                    best["rep"], best["rate"], best["n"], pooled * 100))

        lead = below[0] if below else above[0]
        insights.append(build_insight(
            title="{} of {} reps {} from the {:.1f}% team win rate".format(
                len(significant), len(tested),
                "differs" if len(significant) == 1 else "differ", pooled * 100),
            problem="{} converts {:.1f}% of {} deals; the team rate is {:.1f}%.".format(
                lead["rep"], lead["rate"], lead["n"], pooled * 100),
            cause="Territory, lead quality and deal mix all move a win rate and none "
                  "of them are held constant here. A difference this size is real in "
                  "the data, but it is not by itself evidence about the individual.",
            evidence="Exact binomial test of each rep against the pooled rate, "
                     "Benjamini-Hochberg corrected across {} reps at q<0.05: {}".format(
                         len(tested), detail),
            action="1. Compare lead source and deal size for {} against the team  "
                   "2. Rule out territory before drawing a conclusion about "
                   "performance  3. Review a sample of their lost deals".format(
                       lead["rep"]),
            impact="Bringing the below-rate reps to the team rate would add roughly "
                   "{:.0f} wins across the deals in this data.".format(
                       sum((pooled - t["rate"] / 100) * t["n"] for t in below)),
            severity="high" if below else "positive",
            category="sales_rep_performance",
        ))
    except Exception:
        logger.warning("rep win-rate analysis failed", exc_info=True)


# ══════════════════════════════════════════════════════════
#  CYCLE LENGTH: WON VS LOST
# ══════════════════════════════════════════════════════════

def cycle_by_outcome(df: pd.DataFrame, insights: List, findings: List,
                     risks: List, opps: List) -> None:
    """Do the deals that close behave differently from the ones that do not?

    Compared with Mann-Whitney rather than a t-test, because cycle times
    are right-skewed and a few stalled deals move a mean by more than the
    difference being measured. Reported as medians with a rank-biserial
    effect size — a statistically significant two-day difference on
    thousands of deals is not a finding worth a client's attention.
    """
    found = find_outcome_col(df)
    if not found:
        return
    _outcome_col, outcome = found
    try:
        cycle = _derive_cycle_days(df)
        if cycle is None:
            return

        work = pd.DataFrame({"days": cycle, "won": outcome}).dropna()
        work["won"] = work["won"].astype(bool)
        won = work.loc[work["won"], "days"]
        lost = work.loc[~work["won"], "days"]
        if len(won) < MIN_DEALS_PER_GROUP or len(lost) < MIN_DEALS_PER_GROUP:
            return

        u, p = scipy_stats.mannwhitneyu(won, lost, alternative="two-sided")
        # Rank-biserial: the share of won/lost pairs in which the won deal
        # closed faster, rescaled to -1..1.
        rbc = float(1 - (2 * u) / (len(won) * len(lost)))
        med_won, med_lost = float(won.median()), float(lost.median())

        findings.append(
            "Won deals close in a median {:.0f} days against {:.0f} for lost deals "
            "(n={:,} won, {:,} lost; Mann-Whitney p={:.3g}, rank-biserial "
            "{:+.2f}).".format(med_won, med_lost, len(won), len(lost), p, rbc))

        # Significance without an effect worth acting on is noise dressed up.
        if p >= 0.05 or abs(rbc) < 0.15:
            findings.append(
                "Cycle length does not usefully separate won from lost deals in "
                "this data, so it is not a signal to qualify on.")
            return

        slower_when_lost = med_lost > med_won
        insights.append(build_insight(
            title="Lost deals run {:.0f} days longer than won deals".format(
                abs(med_lost - med_won))
            if slower_when_lost else
            "Won deals run {:.0f} days longer than lost deals".format(
                abs(med_won - med_lost)),
            problem="Median cycle is {:.0f} days for won deals and {:.0f} for lost "
                    "ones.".format(med_won, med_lost),
            cause="A deal that drifts is usually one that was never qualified or has "
                  "lost its sponsor. Which applies here is not in this data — it "
                  "needs the stage history behind each opportunity."
            if slower_when_lost else
            "Longer cycles on won deals normally indicate larger or more complex "
            "purchases rather than a process problem. Deal size would settle it and "
            "is not tested here.",
            evidence="Mann-Whitney U={:.0f}, p={:.3g}, rank-biserial {:+.2f} over "
                     "{:,} won and {:,} lost deals. Medians {:.0f} vs {:.0f} days; "
                     "means {:.0f} vs {:.0f}.".format(
                         u, p, rbc, len(won), len(lost), med_won, med_lost,
                         float(won.mean()), float(lost.mean())),
            action="1. Set a stall threshold at the won-deal 75th percentile "
                   "({:.0f} days)  2. Review anything past it for qualification  "
                   "3. Track the gap monthly rather than as a one-off".format(
                       float(won.quantile(0.75))),
            impact="Deals still open past the point where comparable deals closed "
                   "convert at a lower rate; an age threshold turns that into an "
                   "action instead of a post-mortem."
            if slower_when_lost else
            "Longer winning cycles are a forecasting input, not a problem to fix — "
            "the risk is planning capacity on the blended figure.",
            severity="warning" if slower_when_lost else "info",
            category="sales_cycle_outcome",
        ))
        if slower_when_lost:
            opps.append(
                "Deals open longer than {:.0f} days (the won-deal 75th percentile) "
                "are candidates for requalification — won deals rarely take that "
                "long.".format(float(won.quantile(0.75))))
    except Exception:
        logger.warning("cycle-by-outcome analysis failed", exc_info=True)


def _derive_cycle_days(df: pd.DataFrame) -> Optional[pd.Series]:
    """Days between an opening and a closing date column, if both exist."""
    start_kw = ("created", "opened", "open_date", "start_date", "opportunity_date")
    end_kw = ("closed", "close_date", "won_date", "end_date")
    date_cols = [c for c in df.columns
                 if pd.api.types.is_datetime64_any_dtype(df[c])]
    if len(date_cols) < 2:
        return None
    start = next((c for c in date_cols if any(k in _norm(c) for k in start_kw)), None)
    end = next((c for c in date_cols
                if any(k in _norm(c) for k in end_kw) and c != start), None)
    if not start or not end:
        return None
    days = (df[end] - df[start]).dt.days
    # Negative means closed before opened, and beyond two years is almost
    # always a placeholder date rather than a real cycle.
    return days.where((days > 0) & (days <= 730))


def run_sales_performance(df: pd.DataFrame, insights: List, findings: List,
                          risks: List, opps: List) -> None:
    rep_win_rates(df, insights, findings, risks, opps)
    cycle_by_outcome(df, insights, findings, risks, opps)
