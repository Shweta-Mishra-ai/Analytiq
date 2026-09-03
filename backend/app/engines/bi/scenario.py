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



# ══════════════════════════════════════════════════════════
#  COHORT ANALYSIS
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
    `reliable` flag (r_squared >= 0.1 and p_value < 0.05) tells the caller
    whether the relationship is strong enough to make the projection worth
    showing at all. A weak or non-significant relationship still returns a
    result (so the caller can show 'not reliable enough' rather than nothing),
    but reliable=False should gate whether a report presents the number
    as actionable.

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

    reliable = bool(r_squared >= 0.1 and p_value < 0.05)

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
    if not reliable:
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
    )
