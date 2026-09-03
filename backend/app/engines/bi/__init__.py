"""
engines/bi — business intelligence, split by the question being asked.

benchmark, root_cause, scenario, cohort, pareto and segments each answer
one; insights writes them up and runner sequences them.
app.engines.bi_engine remains as the import path callers already use.
"""
from app.engines.bi.results import (                            # noqa: F401
    BenchmarkResult, BIReport, CohortResult, ParetoResult, RootCauseResult,
    ScenarioResult, SegmentHealth,
)
from app.engines.bi.runner import run_bi                        # noqa: F401
