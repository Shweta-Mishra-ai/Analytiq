"""
engines/bi/scenario.py — what a proposed change would be worth.
"""
from __future__ import annotations

import logging

import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


from typing import Optional

from app.engines.bi.results import ScenarioResult
from app.services.stat_guards import is_restatement



# ══════════════════════════════════════════════════════════
#  SCENARIO PROJECTION
# ══════════════════════════════════════════════════════════

# How far past the observed edge a projection may go before it stops
# being a forecast. A tenth of the observed span is a step beyond what
# was seen; several times it is invention.
OUTSIDE_RANGE_TOLERANCE = 0.10


def analyze_scenario(
    df: pd.DataFrame,
    driver_col: str,
    target_col: str,
    change_pct: float = 10.0,
) -> Optional[ScenarioResult]:
    """
    'What if driver_col improved by change_pct%?' — projects the effect on
    target_col using the historical linear relationship between the two
    (ordinary least squares on the observed data, via scipy.stats.linregress).

    This is a PROJECTION from an association, not a causal guarantee — the
    interpretation/caveat text is deliberately explicit about that, and the
    `reliable` flag tells the caller whether the projection is worth showing
    as actionable at all. It requires three things, not two:

      * the relationship explains something (r_squared >= 0.1),
      * it is statistically significant (p < 0.05), and
      * the projection lands inside the range of the driver the data
        actually covers.

    The third used to be missing, and only the caveat text mentioned it.
    A fitted line returns a value for any input, so asking for a 500%
    increase in a discount rate never observed above 20% produced a 223%
    discount and revenue of -1,665 — reported as reliable. A weak, an
    insignificant or an out-of-range projection still returns a result,
    so the caller can say why rather than showing nothing.

    Returns None if either column isn't usable (non-numeric, all-null, or
    fewer than 10 overlapping non-null rows — too little data to fit a
    trend line that means anything).
    """
    if driver_col not in df.columns or target_col not in df.columns:
        return None
    if driver_col == target_col:
        return None
    if not pd.api.types.is_numeric_dtype(df[driver_col]) or \
       not pd.api.types.is_numeric_dtype(df[target_col]):
        return None

    paired = df[[driver_col, target_col]].dropna()
    if len(paired) < 10:
        return None
    if paired[driver_col].nunique() < 2 or paired[target_col].nunique() < 2:
        return None

    try:
        reg = scipy_stats.linregress(paired[driver_col], paired[target_col])
    except Exception:
        logger.warning("analyze_scenario: linregress failed for %s -> %s",
                       driver_col, target_col, exc_info=True)
        return None

    slope, intercept, r_value, p_value, _std_err = reg
    r_squared = float(r_value ** 2)

    driver_mean  = float(paired[driver_col].mean())
    target_mean  = float(paired[target_col].mean())
    driver_delta = driver_mean * (change_pct / 100.0)
    projected_target = target_mean + slope * driver_delta
    projected_change_pct = (
        (projected_target - target_mean) / abs(target_mean) * 100
        if target_mean != 0 else 0.0
    )

    # Where the projection actually lands on the driver, and whether the
    # data ever went there. A fitted line will happily return a value for
    # any input; that it does so says nothing about whether the business
    # has ever operated at that level. Asking for a 500% rise in a
    # discount rate observed only between 0% and 20% projected a 223%
    # discount and revenue of -1,665 — and reported it as reliable.
    projected_driver = driver_mean + driver_delta
    driver_min = float(paired[driver_col].min())
    driver_max = float(paired[driver_col].max())
    span = driver_max - driver_min
    # A little beyond the edge is a forecast; far beyond it is fiction.
    margin = span * OUTSIDE_RANGE_TOLERANCE
    within_range = bool(driver_min - margin <= projected_driver
                        <= driver_max + margin)

    # A driver that restates the target is not a lever. Ask "what if
    # revenue_k rose 10%?" against revenue and the fit is perfect, the
    # p-value vanishing, and the projection exactly 10% — a flawless,
    # meaningless result, and the most convincing-looking output the
    # scenario engine can produce.
    restates = is_restatement(paired[driver_col], paired[target_col])

    reliable = bool(r_squared >= 0.1 and p_value < 0.05
                    and within_range and not restates)

    direction = "increases" if change_pct > 0 else "decreases"
    interpretation = (
        "If {} {} by {:.0f}% (from an average of {:.2f} to {:.2f}), the "
        "historical relationship in this dataset projects {} would move "
        "from {:.2f} to {:.2f} ({:+.1f}%). This relationship explains "
        "{:.0f}% of the variance in {} (R\u00b2={:.2f}, p={:.4f}).".format(
            driver_col, direction, abs(change_pct), driver_mean,
            driver_mean + driver_delta, target_col, target_mean,
            projected_target, projected_change_pct,
            r_squared * 100, target_col, r_squared, p_value)
    )
    if restates:
        interpretation += (
            " But {} and {} are the same measurement recorded twice — one "
            "is the other rescaled, copied or transformed, which is why the "
            "fit is near-perfect. Moving one moves the other by definition, "
            "so this projects nothing about the business. Pick a driver "
            "that can be changed independently of {}.".format(
                driver_col, target_col, target_col)
        )
    elif not within_range:
        interpretation += (
            " This projection puts {} at {:.2f}, which is outside the {:.2f} "
            "to {:.2f} range the data actually covers. Nothing here records "
            "what happens at that level, so the figure above is the fitted "
            "line extended past the evidence, not a finding. Ask for a change "
            "the data has seen.".format(
                driver_col, projected_driver, driver_min, driver_max)
        )
    elif not reliable:
        interpretation += (
            " This relationship is too weak or not statistically significant "
            "to treat this projection as reliable — treat it as a rough "
            "directional signal at most."
        )

    caveat = (
        "This is a projection from a historical association, not a causal "
        "guarantee — changing {} may not actually cause {} to move this way "
        "in practice. Valid only within (or near) the range of {} actually "
        "observed in this dataset; do not extrapolate far beyond it."
    ).format(driver_col, target_col, driver_col)

    return ScenarioResult(
        driver_col=driver_col,
        target_col=target_col,
        change_pct=round(float(change_pct), 2),
        current_driver_mean=round(driver_mean, 4),
        current_target_mean=round(target_mean, 4),
        projected_target_mean=round(float(projected_target), 4),
        projected_change_pct=round(float(projected_change_pct), 2),
        r_squared=round(r_squared, 4),
        slope=round(float(slope), 6),
        p_value=round(float(p_value), 4),
        reliable=reliable,
        interpretation=interpretation,
        caveat=caveat,
        projected_driver_value=round(float(projected_driver), 4),
        driver_observed_min=round(driver_min, 4),
        driver_observed_max=round(driver_max, 4),
        within_observed_range=within_range,
        driver_restates_target=restates,
    )
