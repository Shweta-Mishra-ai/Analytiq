"""
engines/eda_engine.py — compatibility shim.

Exploratory data analysis lives in app/engines/eda/ now, split by what
is being analysed: results.py holds the shapes, univariate/bivariate/
multivariate do the work on one, two and all columns, findings.py
decides which results are worth telling someone about, and runner.py
orchestrates. It had grown to 1,055 lines covering all of that, with
the section banners inside it already marking where the seams were.

This keeps `from app.engines.eda_engine import run_eda` working. What it
forwards is what something actually imports — checked against the
codebase. New code should import from app.engines.eda.
"""
from app.engines.eda.results import (                           # noqa: F401
    BivariateResult, EDAReport, GroupComparisonResult,
    MulticollinearityResult, TimeSeriesResult, UnivariateResult,
)
from app.engines.eda.univariate import (                        # noqa: F401
    FIT_SAMPLE_SIZE, _fit_distribution, analyze_univariate,
)
from app.engines.eda.bivariate import (                         # noqa: F401
    analyze_bivariate_numeric, analyze_group_comparison,
)
from app.engines.eda.multivariate import (                      # noqa: F401
    analyze_time_series, analyze_vif,
)
from app.engines.eda.runner import run_eda                      # noqa: F401

__all__ = ["run_eda", "EDAReport"]
