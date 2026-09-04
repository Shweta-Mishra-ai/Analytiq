"""Building charts from several threads at once.

The dashboard asks for five charts in parallel, and React's development
double-mount makes that ten. Roughly one request in seventeen came back
500 "Invalid value", so a user saw one blank tile among four good ones —
and the handler swallowed the traceback, so the server log said only that
a 500 had happened.

`template="plotly_dark"` is a name, not a value: plotly.express resolves
it to the single Template object registered on the module and fills in
that object's prototype trace properties on first use. Two threads
reaching that lazily-built state together trip over each other inside
plotly's own property tree — which is why it only ever bit just after a
restart, on the first dashboard anyone opened.
"""
import subprocess
import sys
import textwrap

import concurrent.futures
import numpy as np
import pandas as pd

from app.engines import chart_engine


def _frame():
    rng = np.random.default_rng(0)
    n = 4000
    return pd.DataFrame({
        "region": rng.choice(["North", "South", "East", "West"], n),
        "day": pd.date_range("2024-01-01", periods=n, freq="h"),
        "units": rng.integers(1, 40, n),
        "price": rng.gamma(3, 20, n).round(2),
        "cost": rng.gamma(2, 15, n).round(2),
    })


def test_charts_build_correctly_under_concurrency():
    frame = _frame()
    jobs = [
        lambda: chart_engine.make_bar(frame, "region", "units"),
        lambda: chart_engine.make_line(frame, "day", "units"),
        lambda: chart_engine.make_histogram(frame, "price"),
        lambda: chart_engine.make_pie(frame, "region", "units"),
        lambda: chart_engine.make_heatmap(frame),
        lambda: chart_engine.make_scatter(frame, "price", "cost"),
    ] * 3
    failures = []
    for _ in range(6):
        with concurrent.futures.ThreadPoolExecutor(18) as pool:
            for fut in [pool.submit(j) for j in jobs]:
                try:
                    fut.result()
                except Exception as exc:      # noqa: BLE001 — that is the point
                    failures.append(repr(exc))
    assert not failures, "{} concurrent builds failed: {}".format(
        len(failures), failures[:3])


# The race needs plotly's template registry in its cold, not-yet-populated
# state, which is why it only appeared on the first dashboard after a
# restart. Reimporting plotly restores that — in a subprocess, so nothing
# else in the suite is holding a stale reference to the old module.
_PROBE = textwrap.dedent('''
    import concurrent.futures, contextlib, importlib, sys
    import numpy as np, pandas as pd

    USE_LOCK = sys.argv[1] == "lock"
    rng = np.random.default_rng(0); n = 3000
    f = pd.DataFrame({"region": rng.choice(list("ABCD"), n),
                      "day": pd.date_range("2024-01-01", periods=n, freq="h"),
                      "units": rng.integers(1, 40, n),
                      "price": rng.gamma(3, 20, n).round(2),
                      "cost": rng.gamma(2, 15, n).round(2)})

    def trial():
        for m in [k for k in list(sys.modules) if k.startswith("plotly")]:
            del sys.modules[m]
        sys.modules.pop("app.engines.chart_engine", None)
        ce = importlib.import_module("app.engines.chart_engine")
        if not USE_LOCK:
            ce._FIGURE_LOCK = contextlib.nullcontext()
        jobs = [lambda: ce.make_bar(f, "region", "units"),
                lambda: ce.make_line(f, "day", "units"),
                lambda: ce.make_histogram(f, "price"),
                lambda: ce.make_pie(f, "region", "units"),
                lambda: ce.make_heatmap(f),
                lambda: ce.make_scatter(f, "price", "cost")] * 3
        bad = 0
        with concurrent.futures.ThreadPoolExecutor(18) as p:
            for fut in [p.submit(j) for j in jobs]:
                try:
                    fut.result()
                except Exception:
                    bad += 1
        return bad

    print(sum(trial() for _ in range(int(sys.argv[2]))))
''')


def _cold_start_failures(mode: str, rounds: int = 8) -> int:
    out = subprocess.run([sys.executable, "-c", _PROBE, mode, str(rounds)],
                         capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-800:]
    return int(out.stdout.strip().splitlines()[-1])


def test_the_lock_is_what_makes_that_true():
    """Without the lock the same cold-start workload fails; with it, none
    do. A future change that drops the lock is caught here rather than as
    an intermittent blank tile on someone's dashboard.

    The unlocked arm is a race, so it is not guaranteed to lose on any
    given run — on a loaded or single-core machine the threads can simply
    take turns. That arm therefore skips rather than fails when it comes
    back clean: a scheduling accident in the *reproduction* must never be
    reported as a regression in the code. The locked arm is asserted
    unconditionally, and `test_charts_build_correctly_under_concurrency`
    above is the check that has to pass every time.
    """
    assert _cold_start_failures("lock") == 0, (
        "chart building raced even with the lock held")

    if _cold_start_failures("nolock") == 0:
        pytest.skip("the race did not surface this run — the unlocked arm "
                    "is inherently probabilistic; the locked arm passed")


def test_every_figure_is_still_styled():
    """Routing construction through the lock must not skip the styling
    that used to wrap it."""
    fig = chart_engine.make_bar(_frame(), "region", "units")
    assert fig.layout.paper_bgcolor == "#07080f"
    assert fig.layout.plot_bgcolor == "#0e0f1a"
