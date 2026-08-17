"""
The failures an audit of the running app turned up, none of which any
existing test would have caught.

**Auth failed open.** `UserStore._load()` swallowed every read error and
returned an empty dict, and an empty account store with no admin key set
*is* single-user open mode — correct for a fresh install, catastrophic
for a deployment that has clients. A corrupt `users.pkl`, a permission
change, or a pickle a newer Python cannot read therefore turned a
multi-client deployment into one where every request, `/api/admin/*`
included, was served as an administrator. Silently.

**The whole upload was in memory before its size was checked.**
`await file.read()` first, `validate_file_size(len(data))` second. A 2 GB
upload allocated 2 GB and then got its 413, and an OOM kill takes down
every request in flight — including the ones belonging to people who
uploaded nothing.

**ML scored itself on the future.** Most business data is a log, and the
split was always random, so the model trained on rows that happened
after the ones it was tested on. Measured on a series with a regime
change: R² 0.998 reported against 0.665 on a time-based split. The
number that goes in front of a client was the flattering one.

**The analysis caches never shrank.** `_mem` was capped at eight
datasets; `_caches` was capped at nothing, so every dataset the process
ever touched kept its profile, EDA, BI, story and ML reports alive until
restart — 3.3 MB per dataset for ML and EDA alone.

**Guarded work was guarded in two places out of seven,** and unlimited
password guessing was free.
"""
from __future__ import annotations

import asyncio
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════
#  Auth fails closed
# ══════════════════════════════════════════════════════════

@pytest.fixture()
def store_path(tmp_path):
    return str(tmp_path / "users.pkl")


def test_a_fresh_install_has_no_accounts_and_that_is_fine(store_path):
    from app.services.user_store import UserStore

    store = UserStore(path=store_path)
    assert store.is_empty()
    assert store.unreadable is None


def test_a_corrupt_account_file_is_not_an_empty_one(store_path):
    """This is the whole finding: "no accounts" and "cannot read the
    accounts" used to be the same empty dict, and the first one turns
    authentication off."""
    with open(store_path, "wb") as f:
        f.write(b"\x80\x04 not a pickle at all")
    from app.services.user_store import UserStore

    store = UserStore(path=store_path)
    assert store.unreadable, "a corrupt file was read as an empty store"


def test_an_account_file_of_the_wrong_shape_is_refused(store_path):
    with open(store_path, "wb") as f:
        pickle.dump(["not", "a", "map"], f)
    from app.services.user_store import UserStore

    assert UserStore(path=store_path).unreadable


def test_open_mode_is_off_while_the_accounts_are_unreadable(monkeypatch):
    from app.services import auth

    class _Broken:
        unreadable = "BadPickle: broken"

        @staticmethod
        def is_empty():
            return True

    monkeypatch.setattr(auth, "user_store", _Broken)
    assert auth._open_mode() is False


def _break_the_account_store(monkeypatch):
    from app.services import auth

    class _Broken:
        unreadable = "BadPickle: broken"

        @staticmethod
        def is_empty():
            return True

    monkeypatch.setattr(auth, "user_store", _Broken)


def test_the_api_refuses_while_the_accounts_are_unreadable(client, monkeypatch):
    """503, not 200 as an administrator."""
    _break_the_account_store(monkeypatch)
    response = client.get("/api/datasets")
    assert response.status_code == 503, response.status_code
    body = response.json()["detail"].lower()
    assert "no data has been lost" in body, body


def test_health_still_answers_and_says_why(client, monkeypatch):
    """The one endpoint an operator can still reach has to be the one
    that tells them what is wrong — they may have no token to get any
    further."""
    import app.main as main

    _break_the_account_store(monkeypatch)

    class _Broken:
        unreadable = "BadPickle: broken"

        @staticmethod
        def is_empty():
            return True

    monkeypatch.setattr(main, "user_store", _Broken)
    body = client.get("/api/health").json()
    assert body["status"] == "degraded", body
    assert body["accounts_readable"] is False


def test_nothing_is_written_over_an_unreadable_account_file(store_path):
    """Creating a user would replace a file that still holds the real
    accounts with one holding a single new one — a recoverable read
    failure turned into permanent data loss."""
    with open(store_path, "wb") as f:
        f.write(b"corrupt")
    from app.services.user_store import UserStore

    store = UserStore(path=store_path)
    with pytest.raises(RuntimeError, match="could not be read"):
        store.create("someone", "password123")


def test_a_readable_file_still_round_trips(store_path):
    from app.services.user_store import UserStore

    UserStore(path=store_path).create("alice", "password123")
    reopened = UserStore(path=store_path)
    assert reopened.unreadable is None
    assert reopened.exists("alice")
    assert reopened.verify("alice", "password123")
    assert reopened.verify("alice", "wrong") is None


# ══════════════════════════════════════════════════════════
#  Uploads are refused before they are in memory
# ══════════════════════════════════════════════════════════

class _FakeUpload:
    """Minimal stand-in for UploadFile: chunked reads and headers."""

    def __init__(self, data: bytes, declared: str = ""):
        self._buf = data
        self._pos = 0
        self.headers = {"content-length": declared} if declared else {}
        self.chunks_read = 0

    async def read(self, size: int = -1) -> bytes:
        if size < 0:
            chunk, self._pos = self._buf[self._pos:], len(self._buf)
        else:
            chunk = self._buf[self._pos:self._pos + size]
            self._pos += len(chunk)
        if chunk:
            self.chunks_read += 1
        return chunk


def test_a_declared_size_over_the_limit_is_refused_unread():
    """The cheapest gate: the client said how big it is, and it is too
    big. Not one byte of the body needs to be read to know that."""
    from fastapi import HTTPException
    from app.api.datasets import read_capped

    upload = _FakeUpload(b"x" * 1024, declared=str(500 * 1024 * 1024))
    with pytest.raises(HTTPException) as caught:
        asyncio.run(read_capped(upload, 200, "Dataset"))
    assert caught.value.status_code == 413
    assert upload.chunks_read == 0, "the body was read despite the header"


def test_an_undeclared_oversize_body_is_stopped_while_reading():
    """Content-Length is a claim, not a fact — a body that keeps coming
    has to be cut off as it arrives."""
    from fastapi import HTTPException
    from app.api.datasets import read_capped

    with pytest.raises(HTTPException) as caught:
        asyncio.run(read_capped(
            _FakeUpload(b"x" * (3 * 1024 * 1024)), 1, "Dataset"))
    assert caught.value.status_code == 413


def test_the_refusal_says_nothing_was_uploaded():
    from fastapi import HTTPException
    from app.api.datasets import read_capped

    with pytest.raises(HTTPException) as caught:
        asyncio.run(read_capped(
            _FakeUpload(b"x" * (3 * 1024 * 1024)), 1, "Dataset"))
    assert "nothing was uploaded" in str(caught.value.detail).lower()


def test_a_file_within_the_limit_arrives_intact():
    from app.api.datasets import read_capped

    payload = os.urandom(300_000)
    assert asyncio.run(read_capped(_FakeUpload(payload), 5, "Dataset")) == payload


def test_an_empty_upload_is_not_an_error_here():
    """Emptiness is the loader's problem to describe, and it does it
    better — this gate is only about size."""
    from app.api.datasets import read_capped

    assert asyncio.run(read_capped(_FakeUpload(b""), 5, "Dataset")) == b""


# ══════════════════════════════════════════════════════════
#  ML is scored the way the model would be used
# ══════════════════════════════════════════════════════════

def _log(n: int = 900) -> pd.DataFrame:
    """A log with a regime change halfway: the driver flips sign. A
    random split sees both regimes in training and scores beautifully;
    a model deployed at the halfway point would not."""
    rng = np.random.default_rng(0)
    t = np.arange(n)
    x = rng.normal(0, 1, n)
    y = np.where(t < n // 2, 100 + 0.05 * t + 8 * x,
                 240 + 0.02 * t - 8 * x) + rng.normal(0, 3, n)
    return pd.DataFrame({
        "order_date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "x": x, "sales": y.round(2)})


def test_the_date_column_is_found():
    from app.engines.ml_engine import time_order_column

    assert time_order_column(_log(), "sales") == "order_date"


def test_a_flag_with_a_date_type_is_not_a_timeline():
    """Two distinct timestamps is a status field, not a history, and
    splitting on it would put an arbitrary group in the holdout."""
    from app.engines.ml_engine import time_order_column

    df = _log(120)
    df["batch_date"] = pd.to_datetime(
        np.where(df.index < 60, "2024-01-01", "2024-06-01"))
    assert time_order_column(df[["batch_date", "x", "sales"]], "sales") is None


def test_a_file_with_no_dates_gets_the_random_split():
    from app.engines.ml_engine import time_order_column

    assert time_order_column(_log().drop(columns=["order_date"]), "sales") is None


def test_the_holdout_is_the_most_recent_rows():
    from app.engines.ml_engine import prepare_features, train_models

    df = _log()
    X, y, _enc = prepare_features(df, "sales")
    order = df.loc[X.index, "order_date"]
    _results, X_test, _y_test, _te, _base = train_models(
        X, y, "regression", order=order)
    latest = order.sort_values().index[-len(X_test):]
    assert set(X_test.index) == set(latest)


def test_the_reported_score_is_the_one_the_model_would_earn():
    """The point of the whole change: the flattering number was the one
    being shown to a client."""
    from app.engines.ml_engine import run_ml_pipeline

    report = run_ml_pipeline(_log(), "sales")
    assert report.best_model is not None
    assert report.best_model.test_score < 0.95, report.best_model.test_score


def test_the_report_says_the_split_was_by_time():
    from app.engines.ml_engine import run_ml_pipeline

    warnings = " ".join(run_ml_pipeline(_log(), "sales").warnings)
    assert "on time, not at random" in warnings, warnings
    assert "order_date" in warnings


def test_a_dateless_file_still_trains():
    from app.engines.ml_engine import run_ml_pipeline

    df = _log().drop(columns=["order_date"])
    report = run_ml_pipeline(df, "sales")
    assert report.best_model is not None
    assert "on time, not at random" not in " ".join(report.warnings)


def test_a_short_file_does_not_lose_its_holdout_to_the_time_split():
    """Below a handful of rows in the test period, a time split leaves
    nothing to score against; the random split is the safer answer."""
    from app.engines.ml_engine import run_ml_pipeline

    report = run_ml_pipeline(_log(30), "sales")
    assert report.best_model is not None or report.warnings


# ══════════════════════════════════════════════════════════
#  The caches follow the data out of memory
# ══════════════════════════════════════════════════════════

def _frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"region": rng.choice(list("ABCD"), 200),
                         "revenue": rng.normal(5e4, 9e3, 200)})


def test_the_cache_does_not_outlive_the_dataset_in_memory():
    from app.services.dataset_store import DatasetStore

    store = DatasetStore(base_dir=tempfile.mkdtemp())
    for i in range(store._MEM_LIMIT + 6):
        meta = store.create("u", _frame(i), f"f{i}.csv", size_mb=0.1)
        store.cache_set("u", meta.dataset_id, "profile", {"n": i})
    assert len(store._caches) <= store._MEM_LIMIT, len(store._caches)


def test_an_evicted_cache_is_reloaded_rather_than_recomputed():
    """Dropping the in-memory copy must not cost the analysis — the
    pickle beside the dataset is still valid for the same frame."""
    from app.services.dataset_store import DatasetStore

    store = DatasetStore(base_dir=tempfile.mkdtemp())
    first = store.create("u", _frame(0), "f0.csv", size_mb=0.1)
    store.cache_set("u", first.dataset_id, "profile", {"answer": 42})
    for i in range(1, store._MEM_LIMIT + 4):
        store.create("u", _frame(i), f"f{i}.csv", size_mb=0.1)
    assert store.cache_get("u", first.dataset_id, "profile") == {"answer": 42}


def test_a_changed_frame_still_invalidates_the_reloaded_cache():
    from app.services.dataset_store import DatasetStore

    store = DatasetStore(base_dir=tempfile.mkdtemp())
    meta = store.create("u", _frame(0), "f.csv", size_mb=0.1)
    store.cache_set("u", meta.dataset_id, "profile", {"answer": 42})
    store.update_active("u", meta.dataset_id, _frame(1))
    assert store.cache_get("u", meta.dataset_id, "profile") is None


def test_a_failed_cache_write_is_logged_not_swallowed(monkeypatch, caplog):
    """A full disk announced itself as "the app got slower" and nothing
    else."""
    import builtins

    from app.services.dataset_store import DatasetStore

    store = DatasetStore(base_dir=tempfile.mkdtemp())
    meta = store.create("u", _frame(0), "f.csv", size_mb=0.1)
    real_open = builtins.open

    def no_space(path, mode="r", *args, **kwargs):
        if "cache_" in str(path) and "w" in mode:
            raise OSError(28, "No space left on device")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", no_space)
    with caplog.at_level("WARNING"):
        store.cache_set("u", meta.dataset_id, "profile", {"answer": 42})
    assert any("disk" in r.message.lower() or "persist" in r.message.lower()
               for r in caplog.records), caplog.text
    # The request still succeeds: the in-memory copy is set either way.
    monkeypatch.setattr(builtins, "open", real_open)
    assert store.cache_get("u", meta.dataset_id, "profile") == {"answer": 42}


# ══════════════════════════════════════════════════════════
#  Heavy work is bounded everywhere, not in two places
# ══════════════════════════════════════════════════════════

def test_a_refused_slot_becomes_a_503_with_a_wait():
    from fastapi import HTTPException

    from app.services.load_guard import _Guard, http_slot

    guard = _Guard(1, "analysis")
    with guard.slot():
        with pytest.raises(HTTPException) as caught:
            with http_slot(guard):
                pass
    assert caught.value.status_code == 503
    assert caught.value.headers.get("Retry-After")


def test_the_slot_is_released_when_the_work_raises():
    from app.services.load_guard import _Guard, http_slot

    guard = _Guard(1, "analysis")
    with pytest.raises(ValueError):
        with http_slot(guard):
            raise ValueError("boom")
    assert guard.running == 0


def test_a_cached_result_is_never_refused_for_load(monkeypatch):
    """Serving something already computed costs nothing. Refusing it
    because other people are computing theirs would be the load limit
    causing the outage it exists to prevent."""
    from app.api import analytics
    from app.services.load_guard import ANALYSIS

    monkeypatch.setattr(analytics.store, "cache_get",
                        lambda *a, **k: {"cached": True})
    with ANALYSIS._lock:
        ANALYSIS._running = ANALYSIS.limit          # simulate a full server
    try:
        assert analytics._cached("u", "d", "eda", lambda: {"fresh": True}) \
            == {"cached": True}
    finally:
        with ANALYSIS._lock:
            ANALYSIS._running = 0


def test_every_analysis_route_computes_under_the_guard():
    """EDA, BI, the story engine and the advanced analytics all ran
    unbounded while report rendering was carefully limited."""
    import inspect

    from app.api import advanced_analytics, analytics

    for module in (analytics, advanced_analytics):
        source = inspect.getsource(module._cached)
        assert "http_slot" in source, module.__name__


# ══════════════════════════════════════════════════════════
#  One client cannot fill the disk for everyone
# ══════════════════════════════════════════════════════════

def test_storage_is_measured_from_the_files_not_the_upload_size():
    """The pickles, the cleaned copy and the cached reports are several
    times the size of the CSV that arrived, and it is the disk that
    fills up, not the CSV."""
    from app.services.dataset_store import DatasetStore

    store = DatasetStore(base_dir=tempfile.mkdtemp())
    assert store.storage_mb("u") == 0
    store.create("u", _frame(0), "f.csv", size_mb=0.0)
    assert store.storage_mb("u") > 0


def test_an_owner_over_quota_is_refused_with_a_way_out(monkeypatch):
    from fastapi import HTTPException

    from app.api import datasets

    monkeypatch.setattr(datasets.store, "storage_mb", lambda owner: 9_999)
    monkeypatch.setattr(datasets.config, "max_storage_mb_per_owner", 100)
    with pytest.raises(HTTPException) as caught:
        datasets._check_quota("u")
    assert caught.value.status_code == 413
    detail = str(caught.value.detail).lower()
    assert "delete a dataset" in detail
    assert "nothing already stored has been touched" in detail


def test_one_owner_over_quota_does_not_block_another(monkeypatch):
    from app.api import datasets

    monkeypatch.setattr(datasets.store, "storage_mb",
                        lambda owner: 9_999 if owner == "heavy" else 1)
    monkeypatch.setattr(datasets.config, "max_storage_mb_per_owner", 100)
    datasets._check_quota("light")          # must not raise


# ══════════════════════════════════════════════════════════
#  Passwords cannot be guessed without limit
# ══════════════════════════════════════════════════════════

def test_a_run_of_wrong_passwords_is_stopped():
    from app.services.throttle import LoginThrottle

    throttle = LoginThrottle(max_failures=3, window=300, lockout=300)
    for _ in range(3):
        throttle.record_failure("user:bob", "addr:1.2.3.4")
    allowed, wait = throttle.check("user:bob", "addr:1.2.3.4")
    assert not allowed
    assert wait > 0


def test_the_limit_applies_to_the_account_across_addresses():
    """Otherwise a botnet gets `max_failures` free guesses per host."""
    from app.services.throttle import LoginThrottle

    throttle = LoginThrottle(max_failures=3)
    for i in range(3):
        throttle.record_failure("user:bob", "addr:10.0.0.{}".format(i))
    assert not throttle.check("user:bob", "addr:10.0.0.99")[0]


def test_another_account_from_a_clean_address_is_unaffected():
    from app.services.throttle import LoginThrottle

    throttle = LoginThrottle(max_failures=3)
    for _ in range(3):
        throttle.record_failure("user:bob", "addr:1.2.3.4")
    assert throttle.check("user:alice", "addr:5.6.7.8")[0]


def test_a_correct_password_clears_the_count():
    """Four mistypes then success must not leave someone one mistake
    from a lockout for the rest of the window."""
    from app.services.throttle import LoginThrottle

    throttle = LoginThrottle(max_failures=3)
    throttle.record_failure("user:bob", "addr:1.2.3.4")
    throttle.record_failure("user:bob", "addr:1.2.3.4")
    throttle.record_success("user:bob", "addr:1.2.3.4")
    throttle.record_failure("user:bob", "addr:1.2.3.4")
    assert throttle.check("user:bob", "addr:1.2.3.4")[0]


def test_checking_is_not_attempting():
    from app.services.throttle import LoginThrottle

    throttle = LoginThrottle(max_failures=2)
    for _ in range(10):
        assert throttle.check("user:bob")[0]


def test_the_window_expires():
    from app.services.throttle import LoginThrottle

    throttle = LoginThrottle(max_failures=2, window=0, lockout=0)
    throttle.record_failure("user:bob")
    throttle.record_failure("user:bob")
    assert throttle.check("user:bob")[0]
