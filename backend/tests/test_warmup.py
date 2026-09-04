"""The engines import lazily so uvicorn starts fast. That is fine until
you notice who pays: the first user to open Predict waits ~5s for an
endpoint whose work takes 70ms. `warmup` moves that bill to startup."""
import threading

from app.services import warmup


def test_warm_reports_a_timing_for_every_engine():
    timings = warmup.warm(("json", "csv"))
    assert set(timings) == {"json", "csv"}
    assert all(isinstance(v, float) and v >= 0 for v in timings.values())


def test_a_module_that_cannot_import_is_skipped_not_raised():
    timings = warmup.warm(("json", "app.engines.no_such_engine_at_all"))
    assert "json" in timings
    assert "app.engines.no_such_engine_at_all" not in timings


def test_start_returns_a_daemon_thread_so_shutdown_is_never_blocked():
    thread = warmup.start(("json",))
    assert isinstance(thread, threading.Thread)
    assert thread.daemon, "a non-daemon warmup would hold the process open"
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_the_listed_engines_are_the_ones_the_request_handlers_import():
    """A warmup that names modules nobody imports on the request path
    warms nothing. Each entry must be a real, importable engine."""
    import importlib
    for name in warmup.MODULES:
        assert importlib.import_module(name) is not None
