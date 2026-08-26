"""
engines/eda_depth.py — the analysis a senior analyst adds on top of the
descriptive pass.

The existing EDA reports distributions, correlations, group comparisons
with effect sizes, VIF and trends, and corrects for multiple testing. All
of that is sound and none of it is what makes a finding senior. Four
things were missing:

  * An estimate with no interval around it is a point guess presented as
    a fact. "Average order value is 412" reads very differently from
    "412, and on this sample it could reasonably be anywhere from 388 to
    436".
  * Interactions. The finding that changes a decision is usually not "A
    affects M" but "A affects M for one group and not for another" — and
    a main-effects-only analysis reports the average of the two, which
    can be near zero even when both halves are large.
  * Rare categories. A "worst-performing region" with four rows in it is
    noise wearing the costume of a finding.
  * Class imbalance stated plainly, because it changes how every
    subsequent number should be read.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.engines.domains.base import is_id_column

logger = logging.getLogger(__name__)

# A group smaller than this cannot carry a finding on its own.
MIN_GROUP_N = 15
# Two subgroup effects must differ by at least this much of the larger of
# them before the difference is worth calling an interaction.
INTERACTION_RATIO = 2.0
# A category holding less than this share of rows is rare.
RARE_SHARE = 0.02
# The larger of the two subgroup effects must be at least this many
# standard deviations of the metric. Without a magnitude floor the ratio
# test alone fires on differences that are real and meaningless: a 0.27
# gap on a 1-5 education scale beat a 0.01 gap by 27x and led the report.
MIN_EFFECT_SD = 0.25


@dataclass
class Estimate:
    """A number with the uncertainty that belongs to it."""
    column: str
    statistic: str
    value: float
    ci_low: float
    ci_high: float
    n: int

    @property
    def margin(self) -> float:
        return (self.ci_high - self.ci_low) / 2

    def describe(self) -> str:
        return ("{} of '{}' is {:,.2f} (95% CI {:,.2f} to {:,.2f}, "
                "n={:,})").format(self.statistic, self.column, self.value,
                                  self.ci_low, self.ci_high, self.n)


@dataclass
class Interaction:
    """An effect that is not the same for everyone."""
    metric: str
    factor: str
    moderator: str
    effect_by_level: Dict[str, float]
    reverses: bool
    ratio: float
    description: str
    # The larger effect in standard deviations of the metric, so a reader
    # can judge whether it is worth acting on independent of the units.
    effect_sd: float = 0.0


@dataclass
class RareCategory:
    column: str
    level: str
    n: int
    share: float


@dataclass
class ImbalanceNote:
    column: str
    majority_level: str
    majority_share: float
    note: str


def mean_with_ci(series: pd.Series, confidence: float = 0.95
                 ) -> Optional[Estimate]:
    """Mean and its confidence interval.

    Uses the t distribution, so a small sample widens the interval instead
    of quietly pretending to the same precision as a large one.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n < 3:
        return None
    mean = float(s.mean())
    sem = float(s.std(ddof=1)) / np.sqrt(n) if n > 1 else 0.0
    if sem == 0 or not np.isfinite(sem):
        return Estimate(str(series.name), "Mean", mean, mean, mean, n)
    half = float(scipy_stats.t.ppf(0.5 + confidence / 2, n - 1)) * sem
    return Estimate(str(series.name), "Mean", round(mean, 4),
                    round(mean - half, 4), round(mean + half, 4), n)


def proportion_with_ci(successes: int, n: int,
                       confidence: float = 0.95) -> Optional[Tuple[float, float, float]]:
    """A rate and its interval, by the Wilson method.

    The textbook normal interval misbehaves badly at rates near 0 or 1 and
    on small samples — it can produce a lower bound below zero for a churn
    rate, which is visibly wrong in a client report.
    """
    if n <= 0 or successes < 0 or successes > n:
        return None
    z = float(scipy_stats.norm.ppf(0.5 + confidence / 2))
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (round(p * 100, 2), round(max(centre - half, 0) * 100, 2),
            round(min(centre + half, 1) * 100, 2))


def _usable_categoricals(df: pd.DataFrame, max_levels: int = 8) -> List[str]:
    out = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or is_id_column(c, df[c]):
            continue
        n = df[c].nunique(dropna=True)
        if 2 <= n <= max_levels:
            out.append(c)
    return out


def find_interactions(df: pd.DataFrame, max_results: int = 3
                      ) -> List[Interaction]:
    """Effects that differ across a second factor.

    Main-effects analysis reports the average of the subgroups. When a
    factor helps one group and hurts another, that average can sit near
    zero while both halves are large — the analysis reports "no effect"
    about the most important thing in the data. A reversal is the version
    of this worth leading with.
    """
    results: List[Interaction] = []
    try:
        cats = _usable_categoricals(df)
        nums = [c for c in df.select_dtypes(include="number").columns
                if not is_id_column(c, df[c])
                and not str(c).endswith("__was_missing")]
        if len(cats) < 2 or not nums:
            return results

        for metric in nums[:4]:
            metric_sd = float(pd.to_numeric(df[metric],
                                            errors="coerce").std(ddof=1) or 0)
            if not np.isfinite(metric_sd) or metric_sd <= 0:
                continue
            floor = MIN_EFFECT_SD * metric_sd
            for factor in cats[:4]:
                for moderator in cats[:4]:
                    if factor == moderator:
                        continue
                    try:
                        effects: Dict[str, float] = {}
                        for level, chunk in df.groupby(moderator, dropna=True):
                            if len(chunk) < MIN_GROUP_N * 2:
                                continue
                            means = chunk.groupby(factor)[metric].mean().dropna()
                            sizes = chunk.groupby(factor)[metric].size()
                            means = means[sizes >= MIN_GROUP_N]
                            if len(means) < 2:
                                continue
                            effects[str(level)] = float(means.max() - means.min())
                        if len(effects) < 2:
                            continue
                        vals = list(effects.values())
                        hi, lo = max(vals, key=abs), min(vals, key=abs)
                        if abs(hi) < 1e-9:
                            continue
                        # Real but trivial is not a finding. Judge the
                        # bigger effect against the metric's own spread,
                        # so the test means the same thing on a salary in
                        # thousands and a rating from 1 to 5.
                        if abs(hi) < floor:
                            continue
                        ratio = abs(hi) / max(abs(lo), 1e-9)
                        reverses = (hi > 0) != (lo > 0) and abs(lo) > 1e-9
                        if ratio < INTERACTION_RATIO and not reverses:
                            continue
                        best = max(effects, key=lambda k: abs(effects[k]))
                        least = min(effects, key=lambda k: abs(effects[k]))
                        results.append(Interaction(
                            metric=str(metric), factor=str(factor),
                            moderator=str(moderator), effect_by_level=effects,
                            reverses=bool(reverses), ratio=round(ratio, 2),
                            effect_sd=round(abs(hi) / metric_sd, 2),
                            description=(
                                "The effect of '{}' on '{}' is not the same "
                                "across '{}': it is {:.2f} for {} and {:.2f} "
                                "for {}{}. Reporting a single overall effect "
                                "here would average these together and "
                                "describe neither."
                            ).format(
                                factor, metric, moderator, effects[best], best,
                                effects[least], least,
                                " — the direction reverses" if reverses else "")))
                    except Exception:
                        logger.debug("interaction check failed for %s/%s/%s",
                                     metric, factor, moderator, exc_info=True)
        # A reversal first, then the largest effect relative to spread.
        results.sort(key=lambda i: (i.reverses, i.effect_sd), reverse=True)
    except Exception:
        logger.warning("interaction analysis failed", exc_info=True)
    return results[:max_results]


def find_rare_categories(df: pd.DataFrame,
                         max_results: int = 12) -> List[RareCategory]:
    """Levels too thin to support a finding.

    A 'worst-performing region' holding four rows is noise. Naming these
    lets the report exclude them from comparisons rather than quietly
    ranking them alongside groups a hundred times their size.
    """
    out: List[RareCategory] = []
    n_rows = len(df)
    if n_rows == 0:
        return out
    for col in _usable_categoricals(df, max_levels=50):
        try:
            counts = df[col].value_counts(dropna=True)
            for level, n in counts.items():
                share = n / n_rows
                if n < MIN_GROUP_N or share < RARE_SHARE:
                    out.append(RareCategory(str(col), str(level), int(n),
                                            round(share * 100, 2)))
        except Exception:
            logger.debug("rare-category scan failed for %r", col, exc_info=True)
    out.sort(key=lambda r: r.n)
    return out[:max_results]


def describe_imbalance(df: pd.DataFrame,
                       max_results: int = 3) -> List[ImbalanceNote]:
    """Categorical columns dominated by one level.

    Worth stating plainly: on a column that is 95% one value, an accuracy
    figure, a correlation and a group comparison all mean something
    different from what a reader assumes.
    """
    notes: List[ImbalanceNote] = []
    for col in _usable_categoricals(df, max_levels=10):
        try:
            shares = df[col].value_counts(normalize=True, dropna=True)
            if shares.empty:
                continue
            top_share = float(shares.iloc[0])
            if top_share < 0.85:
                continue
            notes.append(ImbalanceNote(
                column=str(col), majority_level=str(shares.index[0]),
                majority_share=round(top_share * 100, 1),
                note=(
                    "'{}' is {:.0f}% '{}'. Any model predicting it can reach "
                    "{:.0f}% accuracy by always answering '{}' without "
                    "learning anything, and a comparison between its groups "
                    "rests on the {:.0f}% minority."
                ).format(col, top_share * 100, shares.index[0],
                         top_share * 100, shares.index[0],
                         (1 - top_share) * 100)))
        except Exception:
            logger.debug("imbalance check failed for %r", col, exc_info=True)
    notes.sort(key=lambda n: n.majority_share, reverse=True)
    return notes[:max_results]


def key_estimates(df: pd.DataFrame, max_results: int = 5) -> List[Estimate]:
    """Headline measures with intervals, most informative first."""
    out: List[Estimate] = []
    try:
        from app.engines.chart_exporter import _rank_measures
        numeric = df.select_dtypes(include="number").columns.tolist()
        for col in _rank_measures(df, numeric)[:max_results]:
            est = mean_with_ci(df[col])
            if est is not None:
                out.append(est)
    except Exception:
        logger.warning("key estimates failed", exc_info=True)
    return out
