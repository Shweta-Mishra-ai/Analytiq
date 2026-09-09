"""
engines/domains/_common.py — column-finding and comparison helpers shared
by the domain insight engines.

Each domain engine needs the same three things before it can say anything:
locate the column that holds a concept ("which column is the spend?"),
turn a binary column into a rate, and compare groups on a measure. Written
once here so four engines cannot disagree about, say, what counts as a
usable grouping column.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from app.engines.domains.base import is_id_column

logger = logging.getLogger(__name__)

# A categorical column is only useful for grouping when it has more than
# one group and few enough to read in a table.
MIN_GROUPS = 2
MAX_GROUPS = 25


def _norm(col: str) -> str:
    """'MonthlyCharges' / 'monthly_charges' -> 'monthlycharges'."""
    text = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ',
                  re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', str(col)))
    return re.sub(r'[^a-z0-9]+', '', text.lower())


def find_col(df: pd.DataFrame, keywords: Sequence[str],
             exclude: Sequence[str] = (), numeric_only: bool = False
             ) -> Optional[str]:
    """First column whose normalised name contains any keyword.

    Keywords are tried in order, so callers list the most specific first
    ("costpercase" before "cost") — otherwise a loose keyword claims the
    column a precise one was meant to find.
    """
    if df is None or df.empty:
        return None
    excl = tuple(_norm(e) for e in exclude)
    cols = [c for c in df.columns
            if not numeric_only or pd.api.types.is_numeric_dtype(df[c])]
    for kw in keywords:
        k = _norm(kw)
        for c in cols:
            n = _norm(c)
            if k in n and not any(e in n for e in excl):
                return c
    return None


def find_measure(df: pd.DataFrame, keywords: Sequence[str],
                 exclude: Sequence[str] = ()) -> Optional[str]:
    """Like find_col but numeric and never an identifier — a 'total spend'
    that is really an account number produces confident nonsense."""
    col = find_col(df, keywords, exclude=exclude, numeric_only=True)
    if col is not None and is_id_column(col, df[col]):
        return None
    return col


def grouping_columns(df: pd.DataFrame) -> List[str]:
    """Categorical columns with a usable number of groups."""
    out = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or is_id_column(c, df[c]):
            continue
        n = df[c].nunique(dropna=True)
        if MIN_GROUPS <= n <= MAX_GROUPS:
            out.append(c)
    return out


_POSITIVE_MARKERS = {"1", "yes", "y", "true", "t", "churned", "left",
                     "exited", "attrited", "cancelled", "canceled", "lost",
                     "failed", "readmitted", "positive", "won", "returned",
                     "default", "defaulted", "fraud", "approved", "converted"}


def binary_mask(series: pd.Series) -> Optional[pd.Series]:
    """The 'yes' side of a binary column as a boolean Series, or None.

    Split out from binary_rate so that a rate and a breakdown of the same
    column can never disagree about which side is the outcome — a report
    that says "18% churn" over a table where the churned group is the
    other one is worse than either number alone.
    """
    if series is None:
        return None
    s = series.dropna()
    if s.empty:
        return None
    vals = set(str(v).strip().lower() for v in s.unique())
    if len(vals) != 2:
        return None
    hit = vals & _POSITIVE_MARKERS
    if not hit:
        # Numeric 0/1 stored as floats still counts.
        if vals <= {"0.0", "1.0", "0", "1"}:
            hit = {"1.0", "1"} & vals
        if not hit:
            return None
    marker = next(iter(hit))
    return s.astype(str).str.strip().str.lower() == marker


def binary_rate(series: pd.Series) -> Optional[float]:
    """Percentage of 'yes' in a binary column, or None if it isn't binary.

    Accepts 0/1, True/False and the usual word pairs, because a churn flag
    arrives as any of them.
    """
    mask = binary_mask(series)
    if mask is None:
        return None
    return round(float(mask.mean() * 100), 2)


def segment_gap(df: pd.DataFrame, group_col: str, value_col: str,
                agg: str = "mean", min_n: int = 5
                ) -> Optional[Dict]:
    """Best and worst group on a measure, with the size of the gap.

    Groups smaller than `min_n` are dropped: a "worst region" that is one
    record is noise, and printing it as a finding is how a report loses
    a reader's trust.
    """
    if group_col not in df.columns or value_col not in df.columns:
        return None
    try:
        work = df[[group_col, value_col]].dropna()
        if work.empty:
            return None
        grouped = work.groupby(group_col)[value_col]
        sizes = grouped.size()
        keep = sizes[sizes >= min_n].index
        if len(keep) < 2:
            return None
        agg_vals = getattr(grouped, agg)().loc[keep].sort_values()
        worst_name, worst_val = agg_vals.index[0], float(agg_vals.iloc[0])
        best_name, best_val = agg_vals.index[-1], float(agg_vals.iloc[-1])
        if not np.isfinite(best_val) or not np.isfinite(worst_val):
            return None
        # Ratio is undefined against a zero baseline — report the absolute
        # gap instead of dividing and emitting inf.
        ratio = (best_val / worst_val) if worst_val not in (0, 0.0) else None
        return {
            "group_col": group_col, "value_col": value_col,
            "best": str(best_name), "best_val": best_val,
            "worst": str(worst_name), "worst_val": worst_val,
            "gap": best_val - worst_val,
            "ratio": ratio,
            "n_groups": int(len(keep)),
            "n_best": int(sizes.loc[best_name]),
            "n_worst": int(sizes.loc[worst_name]),
        }
    except Exception:
        logger.debug("segment_gap failed for %s by %s", value_col, group_col,
                     exc_info=True)
        return None


# ══════════════════════════════════════════════════════════
#  WHO THE OUTCOME HAPPENS TO
# ══════════════════════════════════════════════════════════
#
# Every one of these domains has a binary outcome sitting in a column —
# returned, churned, attrition, readmission, won — and until now nothing
# computed the rate of that outcome across anything. The engines reached
# for segment_gap, which compares the MEAN OF A NUMERIC COLUMN between
# groups, so a dataset where apparel is returned four times as often came
# back as "Apparel and Electronics differ on Discount". True, and not the
# finding.
#
# This is the question an analyst asks first and the model asks last:
# given the thing we do not want, who does it happen to? It is a
# conditional rate with a test attached, not a model — which is why it
# still says something on the many datasets where a classifier honestly
# cannot.

# A group smaller than this is not reported. A "worst region" of four
# records is noise, and printing it as a finding is how a report loses
# its reader.
MIN_CELL = 25
# Below this many points of difference, a gap is not worth a paragraph
# however significant it is on a large enough sample.
MIN_SPREAD_PP = 5.0
# Bands for a numeric driver. Quartiles read naturally ("the top quarter
# of order values") and survive skew better than fixed-width bins.
NUMERIC_BANDS = 4
# Examining every column of a wide table costs more than it returns.
MAX_DRIVERS_EXAMINED = 40
# One groupby per driver is cheap; the tail of a very long frame is not.
RATE_SAMPLE = 50_000
# A numeric column that separates the two outcomes this cleanly is not a
# driver of them — it is the rule they were derived from. See
# _defines_outcome.
DEFINITIONAL_AUC = 0.98


@dataclass
class RateGap:
    """How much more often the outcome happens in one slice than another."""

    driver: str
    kind: str            # "group" (a category) | "band" (a numeric range)
    high_label: str
    high_rate: float     # percent
    high_n: int
    low_label: str
    low_rate: float
    low_n: int
    overall: float
    spread: float        # percentage points, high - low
    lift: Optional[float]  # high / low, None against a zero baseline
    p_value: float
    n_groups: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def _defines_outcome(values: pd.Series, mask: pd.Series) -> bool:
    """True when a column does not explain the outcome so much as encode it.

    A pass flag derived as `score >= 50` makes score a perfect predictor
    of passing, and the outcome pass duly reported "pass rate is 71.9%
    below a score of 60 against 100% above it" as a finding. It is a
    tautology dressed as an insight, and printing one in a client report
    costs more credibility than the rest of the page earns.

    The test is how cleanly a single cut on the column separates the two
    outcomes. Rank AUC above the threshold means one threshold reproduces
    the flag almost exactly, which is a definition, not a driver. Genuine
    drivers — overtime against attrition, discount against returns — sit
    far below it.
    """
    try:
        from scipy.stats import mannwhitneyu

        hits = values[mask]
        misses = values[~mask]
        if len(hits) < 10 or len(misses) < 10:
            return False
        u = mannwhitneyu(hits, misses, alternative="two-sided").statistic
        auc = u / (len(hits) * len(misses))
        return abs(auc - 0.5) >= DEFINITIONAL_AUC - 0.5
    except Exception:
        logger.debug("definitional check failed", exc_info=True)
        return False


def _rate_table(mask: pd.Series, grouper: pd.Series
                ) -> Optional[Tuple[pd.Series, pd.Series]]:
    """(rate per group in percent, count per group), groups large enough
    to report only."""
    frame = pd.DataFrame({"hit": mask.astype(int), "g": grouper}).dropna()
    if frame.empty:
        return None
    counts = frame.groupby("g", observed=True)["hit"].size()
    keep = counts[counts >= MIN_CELL].index
    if len(keep) < 2:
        return None
    rates = frame[frame["g"].isin(keep)].groupby(
        "g", observed=True)["hit"].mean() * 100
    return rates, counts.loc[keep]


def _gap_p_value(mask: pd.Series, grouper: pd.Series,
                 high, low, n_groups: int) -> float:
    """Probability of a gap this large between these two groups by chance.

    Two deliberate choices, both learned the hard way:

    *Fisher, not chi-square.* The claim printed is "this group against
    that one", so that is the table tested — a 2x2 of the two extremes.
    Fisher's exact test needs no expected-count assumption, which matters
    because the interesting group is usually the small one. An earlier
    version ran an omnibus chi-square and refused whenever any observed
    cell held fewer than five records; on subscription data where the
    enterprise tier churned four times out of 178, that rule threw away
    a 33-point spread — the single largest effect in the dataset — and
    reported nothing at all.

    *Corrected for the search.* The extremes are picked as the best and
    worst of `n_groups`, so the comparison was chosen after seeing the
    data. Across twelve sales reps there are 66 possible pairs and the
    widest of them looks impressive on noise alone. Multiplying by the
    number of pairs (Bonferroni) is the blunt correction, and blunt in
    the safe direction: it can only make a claim harder to make.
    """
    try:
        from scipy.stats import fisher_exact

        hi = mask[grouper == high]
        lo = mask[grouper == low]
        if len(hi) < 2 or len(lo) < 2:
            return 1.0
        table = [[int(hi.sum()), int((~hi).sum())],
                 [int(lo.sum()), int((~lo).sum())]]
        p = float(fisher_exact(table)[1])
        pairs = max(1, n_groups * (n_groups - 1) // 2)
        return min(1.0, p * pairs)
    except Exception:
        logger.debug("gap significance test failed", exc_info=True)
        return 1.0


def outcome_rates(df: pd.DataFrame, outcome_col: str,
                  max_results: int = 6) -> List[RateGap]:
    """Where a binary outcome concentrates, ranked by how much it varies.

    Returns only gaps that are both large (>= MIN_SPREAD_PP points) and
    unlikely to be chance (p < 0.05) on groups big enough to mean
    something. An empty list is a real answer: it says the outcome is
    spread evenly, which is worth knowing and is not the same as having
    looked for nothing.
    """
    if outcome_col not in df.columns:
        return []
    work = df if len(df) <= RATE_SAMPLE else df.sample(RATE_SAMPLE,
                                                       random_state=42)
    mask = binary_mask(work[outcome_col])
    if mask is None or mask.nunique() < 2:
        return []
    mask = mask.reindex(work.index).dropna()
    if len(mask) < MIN_CELL * 2:
        return []
    overall = float(mask.mean() * 100)

    gaps: List[RateGap] = []
    examined = 0
    for col in work.columns:
        if col == outcome_col or examined >= MAX_DRIVERS_EXAMINED:
            continue
        series = work[col].reindex(mask.index)
        bands: List = []
        try:
            if is_id_column(col, series):
                continue
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce")
                usable = numeric.notna()
                if _defines_outcome(numeric[usable], mask[usable]):
                    logger.debug(
                        "%s defines %s rather than driving it — skipped",
                        col, outcome_col)
                    continue
                # A numeric column with a handful of values is a category
                # wearing a number — band it and you get one band.
                if numeric.nunique(dropna=True) <= NUMERIC_BANDS + 1:
                    grouper = numeric.astype("object")
                    kind = "group"
                else:
                    grouper = pd.qcut(numeric, NUMERIC_BANDS,
                                      duplicates="drop")
                    if grouper.nunique(dropna=True) < 2:
                        continue
                    kind = "band"
                    bands = list(grouper.cat.categories)
            elif pd.api.types.is_datetime64_any_dtype(series):
                continue
            else:
                n_groups = series.nunique(dropna=True)
                if not (MIN_GROUPS <= n_groups <= MAX_GROUPS):
                    continue
                grouper = series.astype("object")
                kind = "group"
            examined += 1

            table = _rate_table(mask, grouper)
            if table is None:
                continue
            rates, counts = table
            rates = rates.sort_values()
            low_key, low_rate = rates.index[0], float(rates.iloc[0])
            high_key, high_rate = rates.index[-1], float(rates.iloc[-1])
            spread = high_rate - low_rate
            if spread < MIN_SPREAD_PP:
                continue
            # A quartile prints as "(-0.001, 77.257]" unless it is turned
            # into words here. A report that titles a finding with a raw
            # pandas interval has stopped talking to its reader.
            low_label = (_band_label(low_key, bands) if kind == "band"
                         else str(low_key))
            high_label = (_band_label(high_key, bands) if kind == "band"
                          else str(high_key))

            aligned = pd.Series(grouper).reindex(mask.index)
            p = _gap_p_value(mask, aligned, high_key, low_key, len(rates))
            if p >= 0.05:
                continue

            gaps.append(RateGap(
                driver=str(col), kind=kind,
                high_label=high_label, high_rate=round(high_rate, 1),
                high_n=int(counts.loc[high_key]),
                low_label=low_label, low_rate=round(low_rate, 1),
                low_n=int(counts.loc[low_key]),
                overall=round(overall, 1), spread=round(spread, 1),
                lift=round(high_rate / low_rate, 1) if low_rate > 0 else None,
                p_value=p, n_groups=int(len(rates)),
            ))
        except Exception:
            logger.debug("outcome_rates failed on %s", col, exc_info=True)
            continue

    gaps.sort(key=lambda g: g.spread, reverse=True)
    return gaps[:max_results]


def _band_label(interval, bands: Sequence) -> str:
    """A quartile in words. "under 77" beats "(-0.001, 77.257]" to every
    reader who did not write the code."""
    try:
        if bands and interval == bands[0]:
            return "under {}".format(fmt(interval.right))
        if bands and interval == bands[-1]:
            return "over {}".format(fmt(interval.left))
        return "{} to {}".format(fmt(interval.left), fmt(interval.right))
    except Exception:
        return str(interval)


def _slice_label(gap: RateGap, high: bool) -> str:
    """The slice, named with the column it came from.

    "concentrates in 'Yes'" is a finding the reader cannot act on: yes of
    what? The driver has to travel with the value.
    """
    label = gap.high_label if high else gap.low_label
    if gap.kind == "band":
        return "{} {}".format(gap.driver, label)
    return "{} '{}'".format(gap.driver, label)


def describe_gap(gap: RateGap, outcome_noun: str) -> str:
    """One sentence a non-technical reader can act on."""
    times = ("{:.1f}x".format(gap.lift) if gap.lift
             else "{:.1f} points above".format(gap.spread))
    return ("{} runs at {:.1f}% in {} against {:.1f}% in {} — {} the rate, "
            "on {:,} and {:,} records.".format(
                outcome_noun.capitalize(), gap.high_rate,
                _slice_label(gap, True), gap.low_rate,
                _slice_label(gap, False), times, gap.high_n, gap.low_n))


def rate_insights(df: pd.DataFrame, outcome_col: str, outcome_noun: str,
                  category: str = "concentration", good: bool = False,
                  max_insights: int = 2, verdict: bool = True) -> Dict:
    """Turn the sharpest outcome concentrations into report material.

    `good` says which end of the gap is the problem, and it changes every
    sentence. For attrition, churn, returns or readmission the worry is
    the group at the top. For a win rate or a conversion rate it is the
    group at the bottom — and without this flag the report led a sales
    analysis with "The win concentrates in Rep 12: 42.1%" marked
    critical, which reads as an alarm about the best performer in the
    business.

    `verdict` says whether the direction is known at all. A column found
    by shape rather than by name — an `approved_b` flag nobody declared —
    gets the same table and the same Fisher test, and no opinion: the
    concentration is a fact, whereas "critical risk" is a claim about
    which end of it the business wants, and this function has no basis
    for that one. With verdict False the finding is stated, capped at
    warning, and left for the reader to judge.

    Returns the same {insights, findings, risks, opportunities, actions}
    shape the domain engines already assemble, so an engine gets the
    analysis a reader expects first: not "what is the average of this
    column" but "who is this happening to, and how much more often than
    everyone else".
    """
    from app.engines.domains.base import build_insight

    out: Dict = {"insights": [], "findings": [], "risks": [],
                 "opportunities": [], "actions": []}
    gaps = outcome_rates(df, outcome_col)
    if not gaps:
        return out

    for gap in gaps[:max_insights]:
        out["findings"].append(describe_gap(gap, outcome_noun))

        # The slice that needs attention, and the one it should look like.
        problem = _slice_label(gap, high=not good)
        target = _slice_label(gap, high=good)
        problem_rate = gap.low_rate if good else gap.high_rate
        target_rate = gap.high_rate if good else gap.low_rate
        problem_n = gap.low_n if good else gap.high_n

        multiple = ("{:.1f}x".format(gap.lift) if gap.lift
                    else "{:.1f} points".format(gap.spread))

        # The gap to the average, in records rather than percentages —
        # a count is what gets a plan approved.
        excess = int(round(problem_n * abs(problem_rate - gap.overall) / 100))
        if excess > 0 and verdict:
            out["opportunities"].append(
                "Moving {} to the {:.1f}% average is about {:,} {} {} across "
                "the {:,} records in that group.".format(
                    problem, gap.overall, excess,
                    "more" if good else "fewer", outcome_noun, problem_n))

        severity = ("critical" if gap.spread >= 20 else
                    "high" if gap.spread >= 10 else "warning")
        if not verdict:
            # Nothing here says the high group is the wrong end.
            severity = "warning" if severity == "critical" else severity
        if severity == "critical" and verdict:
            out["risks"].append(
                "{} in {} is {:.1f}% against {:.1f}% overall. A gap this "
                "wide between slices of the same book is a difference in "
                "how they are handled, not a coincidence.".format(
                    outcome_noun.capitalize(), problem, problem_rate,
                    gap.overall))

        out["actions"].append(
            "Compare {} against {} — {} separates them on {}.".format(
                problem, target, multiple, outcome_noun))

        out["insights"].append(build_insight(
            # A noun phrase, not a sentence. The outcome noun is
            # sometimes plural (returns) and sometimes singular (churn),
            # so any verb here agrees with one and not the other —
            # "Where returns concentrates" was the result of trying.
            title="{} {}: {} at {:.1f}% against {:.1f}%".format(
                "Lowest" if good else "Highest", outcome_noun,
                problem, problem_rate, target_rate),
            problem="{} runs at {:.1f}% in {}, against {:.1f}% overall and "
                    "{:.1f}% in {}".format(
                        outcome_noun.capitalize(), problem_rate, problem,
                        gap.overall, target_rate, target),
            cause="Something about that group differs — the data shows "
                  "where, not why, and the difference is worth naming "
                  "before it is modelled",
            evidence="{} — Fisher's exact test, p = {:.2g} after "
                     "correcting for the {} groups the comparison was "
                     "chosen from".format(
                         describe_gap(gap, outcome_noun), gap.p_value,
                         gap.n_groups),
            action="1. Pull the {:,} records in {} and read a sample "
                   "against {}  2. Name the difference in process, not in "
                   "the data  3. Change it for that group only  "
                   "4. Re-measure after one full cycle".format(
                       problem_n, problem, target),
            impact=("About {:,} records in that group sit {} the overall "
                    "rate.".format(excess, "below" if good else "above")
                    if verdict else
                    "{:,} of the {:,} records in that group carry the flag, "
                    "against {:.1f}% across the file. Whether that is the "
                    "good end or the bad one is a question about the "
                    "business, not the data.".format(
                        int(round(problem_n * problem_rate / 100)),
                        problem_n, gap.overall)),
            severity=severity, category=category,
        ))
    return out


def concentration(df: pd.DataFrame, group_col: str, value_col: str,
                  top_n: int = 1) -> Optional[Dict]:
    """Share of a total held by the largest group(s) — the Pareto check
    behind 'one channel carries 68% of spend'."""
    if group_col not in df.columns or value_col not in df.columns:
        return None
    try:
        totals = df.groupby(group_col)[value_col].sum().sort_values(
            ascending=False)
        grand = float(totals.sum())
        if grand <= 0 or totals.empty:
            return None
        top = totals.head(top_n)
        return {
            "top_names": [str(i) for i in top.index],
            "top_share": round(float(top.sum()) / grand * 100, 1),
            "n_groups": int(len(totals)),
            "total": grand,
        }
    except Exception:
        logger.debug("concentration failed", exc_info=True)
        return None


def variability(series: pd.Series) -> Optional[float]:
    """Coefficient of variation as a percentage — how unstable a process
    measure is, independent of its units."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 5:
        return None
    mean = float(s.mean())
    if mean == 0:
        return None
    return round(float(s.std() / abs(mean)) * 100, 1)


def fmt(value: float, unit: str = "") -> str:
    """Numbers a reader can scan: thousands separated, sensible precision."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if abs(value) >= 1_000_000:
        body = f"{value/1_000_000:,.1f}M"
    elif abs(value) >= 1000:
        body = f"{value:,.0f}"
    elif abs(value) >= 10:
        body = f"{value:,.1f}"
    else:
        body = f"{value:,.2f}"
    return f"{body}{unit}"


def benchmark_note(domain: str, column: str, value: float) -> str:
    """One sentence placing a value against its published range, or ''.

    Returns empty rather than inventing a comparison when no range applies
    — an unsupported benchmark is worse than none.
    """
    try:
        from app.engines.industry_benchmarks import (
            lookup_benchmark, format_benchmark_context)
        bm = lookup_benchmark(domain, column)
        if bm is None:
            return ""
        return format_benchmark_context(bm)
    except Exception:
        logger.debug("benchmark lookup failed for %s/%s", domain, column,
                     exc_info=True)
        return ""
