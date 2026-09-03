"""
engines/bi/results.py — what a business-intelligence run returns.

Shapes only, shared by every analysis in the package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ══════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════



@dataclass
class BenchmarkResult:
    column:          str
    mean:            float
    median:          float
    p25:             float
    p75:             float
    p90:             float
    top_10_pct:      float    # value threshold for top 10%
    bottom_10_pct:   float    # value threshold for bottom 10%
    above_avg_pct:   float    # % of rows above average
    cv:              float    # coefficient of variation
    benchmark_label: str      # "High variance", "Consistent", etc.
    interpretation:  str


@dataclass
class ScenarioResult:
    driver_col:      str
    target_col:      str
    change_pct:      float          # hypothetical % change applied to driver
    current_driver_mean:  float
    current_target_mean:  float
    projected_target_mean: float
    projected_change_pct: float     # resulting % change in target
    r_squared:       float          # how much of target's variance driver explains
    slope:           float          # linear regression slope (target per unit driver)
    p_value:         float
    reliable:        bool           # r_squared/p_value clear the reliability bar
    interpretation:  str
    caveat:          str


@dataclass
class RootCauseResult:
    target_col:      str
    low_performer_threshold: float
    n_low_performers: int
    low_pct:         float
    drivers:         List[Dict]   # [{factor, impact, direction, detail}]
    top_driver:      str
    interpretation:  str
    recommendations: List[str]


@dataclass
class CohortResult:
    cohort_col:      str
    metric_col:      str
    cohorts:         List[Dict]   # [{name, n, mean, median, rank, vs_avg_pct}]
    best_cohort:     str
    worst_cohort:    str
    gap:             float        # best - worst mean
    gap_pct:         float        # gap as % of worst
    is_significant:  bool
    p_value:         float
    test_used:       str
    interpretation:  str
    recommendations: List[str]


@dataclass
class ParetoResult:
    group_col:       str
    value_col:       str
    agg_fn:          str         # "sum" or "mean"
    groups:          List[Dict]  # [{name, value, cumulative_pct, in_top_20}]
    top_20_pct_groups: int       # how many groups make up 80% of value
    top_groups_share:  float     # % of total value from top 20% groups
    pareto_holds:    bool        # True if top 20% drives >= 60% of value
    interpretation:  str


@dataclass
class SegmentHealth:
    segment_name:    str
    segment_col:     str
    n:               int
    metrics:         Dict[str, Dict]  # {metric: {mean, vs_avg, rank, status}}
    health_score:    float    # 0-100
    strengths:       List[str]
    weaknesses:      List[str]
    opportunity:     str


@dataclass
class BIReport:
    benchmarks:      List[BenchmarkResult]   = field(default_factory=list)
    root_causes:     List[RootCauseResult]   = field(default_factory=list)
    cohorts:         List[CohortResult]      = field(default_factory=list)
    pareto:          List[ParetoResult]      = field(default_factory=list)
    segments:        List[SegmentHealth]     = field(default_factory=list)
    key_insights:    List[str]               = field(default_factory=list)
    executive_brief: str                     = ""


# ══════════════════════════════════════════════════════════