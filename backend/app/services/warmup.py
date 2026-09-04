"""services/warmup.py — pay the import cost before a user does.

The engines import scikit-learn, statsmodels and ReportLab lazily, inside
the request handler, to keep `uvicorn` startup fast. That works, but the
bill does not disappear: it lands on whoever clicks Predict first, who
waits about five seconds for an endpoint that then answers in 70ms. The
cost is the import, not the work.

So import them on a daemon thread at startup instead. Nothing waits on
this thread; a request arriving mid-warmup simply performs the import
itself, exactly as it did before, because Python's import lock makes the
second importer wait for the first rather than duplicate it.
"""
from __future__ import annotations

import importlib
import logging
import threading
import time

logger = logging.getLogger(__name__)

# Ordered by what a user reaches first, so an early click benefits even if
# the sweep has not finished.
MODULES = (
    "app.engines.eda_engine",
    "app.engines.stats_engine",
    "app.engines.bi_engine",
    "app.engines.ml_engine",
    "app.engines.health_engine",
    "app.engines.chart_exporter",
    "app.engines.pdf_builder",
)


def warm(modules=MODULES) -> dict:
    """Import each module, returning the seconds each one took.

    Never raises: an engine that cannot import here would have failed in
    the request handler too, and the warmup is not the place to discover
    that fatally."""
    timings = {}
    for name in modules:
        started = time.perf_counter()
        try:
            importlib.import_module(name)
        except Exception:
            logger.warning("warmup: %s did not import", name, exc_info=True)
            continue
        timings[name] = round(time.perf_counter() - started, 3)
    total = round(sum(timings.values()), 2)
    logger.info("warmup: %d engines imported in %ss", len(timings), total)
    return timings


def start(modules=MODULES) -> threading.Thread:
    """Run `warm` on a daemon thread and return it. Daemon, so a shutdown
    during warmup is immediate rather than blocked on an import."""
    thread = threading.Thread(target=warm, args=(modules,),
                              name="engine-warmup", daemon=True)
    thread.start()
    return thread
