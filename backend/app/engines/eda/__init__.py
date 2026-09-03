"""
engines/eda — exploratory data analysis, split by what is being analysed.

results.py holds the shapes; univariate/bivariate/multivariate do the
work on one, two and all columns; findings.py decides what is worth
saying; runner.py orchestrates. app.engines.eda_engine remains as the
import path callers already use.
"""
from app.engines.eda.results import (                           # noqa: F401
    BivariateResult, EDAReport, GroupComparisonResult,
    MulticollinearityResult, TimeSeriesResult, UnivariateResult,
)
from app.engines.eda.runner import run_eda                      # noqa: F401
