"""
engines/bi_engine.py — compatibility shim.

Business intelligence lives in app/engines/bi/ now, split by the
question each analysis answers: benchmark, root_cause, scenario, cohort,
pareto and segments, with insights writing them up and runner sequencing
them. It had grown to 977 lines covering all of it.

This keeps `from app.engines.bi_engine import run_bi` working. What it
forwards is what something actually imports — checked against the
codebase. New code should import from app.engines.bi.
"""
from app.engines.bi.results import (                            # noqa: F401
    BenchmarkResult, BIReport, CohortResult, ParetoResult, RootCauseResult,
    ScenarioResult, SegmentHealth,
)
from app.engines.bi.benchmark import analyze_benchmark          # noqa: F401
from app.engines.bi.root_cause import analyze_root_cause        # noqa: F401
from app.engines.bi.scenario import analyze_scenario            # noqa: F401
from app.engines.bi.cohort import analyze_cohort                # noqa: F401
from app.engines.bi.pareto import analyze_pareto                # noqa: F401
from app.engines.bi.segments import analyze_segment_health      # noqa: F401
from app.engines.bi.runner import (                             # noqa: F401
    _is_performance_metric, run_bi,
)

__all__ = ["run_bi", "BIReport"]
