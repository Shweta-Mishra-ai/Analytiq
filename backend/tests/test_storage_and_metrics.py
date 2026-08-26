"""
tests/test_storage_and_metrics.py — the storage directory must be data,
not a code-execution path, and the server must be able to say what it is
doing.

The rule these tests defend: nothing under DATA_DIR is unpickled on the
strength of being there. Datasets and account records are parquet/JSON,
and the one remaining pickle (analysis caches, which hold fitted models)
is only read back when it carries this server's signature.
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest

from app.services.dataset_store import DatasetStore
from app.services.frame_io import read_frame, write_frame
from app.services.llm_cache import LLMCache
from app.services.metrics import Metrics
from app.services.user_store import UserStore


@pytest.fixture()
def store(tmp_path) -> DatasetStore:
    return DatasetStore(base_dir=str(tmp_path / "datasets"))


# ── frames survive the trip ──────────────────────────────

def test_frame_round_trips_dtypes_and_values(tmp_path):
    df = pd.DataFrame({
        "n": [1, 2, 3],
        "f": [1.5, np.nan, 3.25],
        "s": ["a", "b", None],
        "b": [True, False, True],
        "d": pd.to_datetime(["2024-01-01", "2024-06-30", "2025-01-01"]),
    })
    path = str(tmp_path / "f.parquet")
    assert write_frame(path, df) == []
    back = read_frame(path)
    pd.testing.assert_frame_equal(back, df)


def test_duplicate_and_non_string_column_names_survive(tmp_path):
    # `Region, Region, 2024` is an ordinary spreadsheet header row, and
    # arrow addresses columns by unique string name — so this is the case
    # a naive to_parquet would refuse or silently rename.
    df = pd.DataFrame([[1, "x", 10], [2, "y", 20]],
                      columns=["Region", "Region", 2024])
    path = str(tmp_path / "dupes.parquet")
    write_frame(path, df)
    back = read_frame(path)
    assert list(back.columns) == ["Region", "Region", 2024]
    assert back.iloc[1, 2] == 20


def test_meaningful_index_is_not_dropped(tmp_path):
    df = pd.DataFrame({"v": [1, 2]},
                      index=pd.to_datetime(["2024-01-01", "2024-02-01"]))
    path = str(tmp_path / "dated.parquet")
    write_frame(path, df)
    back = read_frame(path)
    assert list(back.index) == list(df.index)


def test_mixed_type_column_is_stored_as_text_and_named(tmp_path):
    df = pd.DataFrame({"m": [12, "n/a", None, 3.5], "ok": ["a", "b", "c", "d"]})
    path = str(tmp_path / "mixed.parquet")
    coerced = write_frame(path, df)
    assert coerced == ["m"], "the caller must be told which column changed"
    back = read_frame(path)
    # The values are still there and still distinguishable; only their
    # type changed, and the null stayed a null.
    assert back["m"].iloc[0] == "12"
    assert back["m"].iloc[1] == "n/a"
    assert pd.isna(back["m"].iloc[2])
    assert list(back["ok"]) == ["a", "b", "c", "d"]


# ── the store writes data, not code ──────────────────────

def test_dataset_is_stored_without_pickling_it(store, hr_df):
    meta = store.create("acme", hr_df, "hr.csv", 0.4)
    written = os.listdir(store._dir("acme", meta.dataset_id))
    assert not any(f.endswith(".pkl") for f in written), written
    assert "raw.parquet" in written and "active.parquet" in written
    assert "meta.json" in written


def test_metadata_is_readable_json(store, hr_df):
    meta = store.create("acme", hr_df, "hr.csv", 0.4)
    with open(store._path("acme", meta.dataset_id, "meta.json")) as fh:
        raw = json.load(fh)
    assert raw["filename"] == "hr.csv"
    assert raw["rows"] == len(hr_df)


def test_dataset_survives_a_restart(tmp_path, hr_df):
    base = str(tmp_path / "datasets")
    first = DatasetStore(base_dir=base)
    meta = first.create("acme", hr_df, "hr.csv", 0.4)

    second = DatasetStore(base_dir=base)  # fresh process, cold memory cache
    back = second.get_df("acme", meta.dataset_id)
    assert back is not None
    pd.testing.assert_frame_equal(back, hr_df)
    assert second.get_meta("acme", meta.dataset_id).filename == "hr.csv"


def test_cleaning_a_dataset_leaves_the_raw_upload_intact(store, hr_df):
    meta = store.create("acme", hr_df, "hr.csv", 0.4)
    store.update_active("acme", meta.dataset_id, hr_df.head(10))
    assert len(store.get_df("acme", meta.dataset_id)) == 10
    assert len(store.get_raw_df("acme", meta.dataset_id)) == len(hr_df)
    assert store.get_meta("acme", meta.dataset_id).rows == 10


def test_mixed_column_coercion_reaches_the_upload_warnings(store):
    df = pd.DataFrame({"amount": [1, "unknown", 3], "id": [1, 2, 3]})
    meta = store.create("acme", df, "messy.csv", 0.1)
    assert any("amount" in w for w in meta.warnings), meta.warnings


# ── caches: signed, or not read at all ───────────────────

def test_cache_round_trips_an_arbitrary_object(store, hr_df):
    meta = store.create("acme", hr_df, "hr.csv", 0.4)
    store.cache_set("acme", meta.dataset_id, "stats", {"mean": 3.5})
    store._caches.clear()  # force the disk path
    assert store.cache_get("acme", meta.dataset_id, "stats") == {"mean": 3.5}


def test_cache_invalidates_when_the_data_changes(store, hr_df):
    meta = store.create("acme", hr_df, "hr.csv", 0.4)
    store.cache_set("acme", meta.dataset_id, "stats", {"mean": 3.5})
    store.update_active("acme", meta.dataset_id, hr_df.head(10))
    assert store.cache_get("acme", meta.dataset_id, "stats") is None


def test_an_unsigned_cache_file_is_never_unpickled(store, hr_df):
    """The point of the whole exercise: a pickle someone else dropped in
    the storage directory must not be loaded."""
    meta = store.create("acme", hr_df, "hr.csv", 0.4)
    store.cache_set("acme", meta.dataset_id, "stats", {"mean": 3.5})
    store._caches.clear()

    path = store._path("acme", meta.dataset_id,
                       f"cache_{store._safe_key('stats')}.bin")

    detonated = []

    class Boom:
        def __reduce__(self):
            return (detonated.append, (True,))

    with open(path, "wb") as fh:
        fh.write(b"\x00" * 32)  # a signature this server did not produce
        fh.write(pickle.dumps((store._hash_df(hr_df), Boom())))

    assert store.cache_get("acme", meta.dataset_id, "stats") is None
    assert detonated == [], "an unsigned cache entry was executed"


def test_a_cache_signed_by_another_install_is_rejected(tmp_path, hr_df):
    a = DatasetStore(base_dir=str(tmp_path / "a"))
    meta = a.create("acme", hr_df, "hr.csv", 0.4)
    a.cache_set("acme", meta.dataset_id, "stats", {"mean": 3.5})

    b = DatasetStore(base_dir=str(tmp_path / "a"))
    b._sign_key = b"different-key-entirely-32-bytes!!"
    b._caches.clear()
    assert b.cache_get("acme", meta.dataset_id, "stats") is None


def test_cache_key_from_user_text_cannot_escape_the_dataset_dir(store, hr_df):
    # `ml_{target}` puts a user-chosen column name into a filename.
    meta = store.create("acme", hr_df, "hr.csv", 0.4)
    store.cache_set("acme", meta.dataset_id, "ml_../../escape", {"x": 1})
    written = os.listdir(store._dir("acme", meta.dataset_id))
    assert all(".." not in f and "/" not in f for f in written), written
    store._caches.clear()
    assert store.cache_get("acme", meta.dataset_id, "ml_../../escape") == {"x": 1}


# ── accounts ─────────────────────────────────────────────

def test_accounts_persist_as_json_not_pickle(tmp_path):
    path = str(tmp_path / "users.json")
    first = UserStore(path)
    first.create("alice", "password123")

    with open(path) as fh:
        raw = json.load(fh)          # readable as JSON at all
    assert "alice" in raw
    assert "password123" not in json.dumps(raw), "password stored in the clear"

    second = UserStore(path)          # survives a restart
    assert second.verify("alice", "password123") is not None
    assert second.verify("alice", "wrong") is None


def test_account_file_is_not_world_readable(tmp_path):
    path = str(tmp_path / "users.json")
    UserStore(path).create("alice", "password123")
    assert os.stat(path).st_mode & 0o077 == 0, "credentials readable by others"


def test_an_unreadable_account_file_does_not_look_like_first_run(tmp_path):
    """An empty store means "no accounts yet", which means open mode.
    A corrupt file must not be mistaken for one."""
    path = tmp_path / "users.json"
    path.write_text("{not json")
    with pytest.raises(Exception):
        UserStore(str(path))


# ── narrative cache ──────────────────────────────────────

def test_llm_cache_returns_the_same_prompt_and_misses_a_different_one(tmp_path):
    cache = LLMCache(base_dir=str(tmp_path / "llm"))
    assert cache.get("sys", "user", "exec", "m") is None
    cache.put("sys", "user", "exec", "m", "A generated paragraph.")
    assert cache.get("sys", "user", "exec", "m") == "A generated paragraph."
    # A cleaned dataset produces a different prompt, and must not be
    # served last week's narrative.
    assert cache.get("sys", "different data", "exec", "m") is None


def test_llm_cache_expires(tmp_path):
    cache = LLMCache(base_dir=str(tmp_path / "llm"), ttl=0)
    cache.put("sys", "user", "exec", "m", "stale")
    assert cache.get("sys", "user", "exec", "m") is None


def test_llm_cache_survives_a_restart(tmp_path):
    base = str(tmp_path / "llm")
    LLMCache(base_dir=base).put("sys", "user", "exec", "m", "kept")
    assert LLMCache(base_dir=base).get("sys", "user", "exec", "m") == "kept"


def test_chat_task_serves_the_second_identical_call_from_cache(monkeypatch,
                                                              tmp_path):
    from app.ai import llm_client as mod
    from app.services import llm_cache as cache_mod
    monkeypatch.setattr(cache_mod, "llm_cache",
                        cache_mod.LLMCache(base_dir=str(tmp_path / "llm")))

    calls = []
    monkeypatch.setattr(mod, "Groq", lambda api_key: object())
    client = mod.LLMClient(api_key="fake")
    monkeypatch.setattr(mod.gemini_client, "is_configured", lambda: False)

    def fake(*a, **k):
        calls.append(1)
        return "narrative"
    monkeypatch.setattr(client, "_groq_report", fake)

    first = client.chat_task("sys", "user", task="executive_summary")
    second = client.chat_task("sys", "user", task="executive_summary")
    assert first == second == "narrative"
    assert len(calls) == 1, "the identical second call was billed again"


# ── metrics ──────────────────────────────────────────────

def test_metrics_time_an_operation():
    m = Metrics()
    with m.timed("report.pdf"):
        pass
    snap = m.snapshot()
    assert snap["operations"]["report.pdf"]["count"] == 1
    assert snap["operations"]["report.pdf"]["median_sec"] >= 0


def test_timed_counts_a_failure_and_re_raises():
    m = Metrics()
    with pytest.raises(ValueError):
        with m.timed("report.pptx"):
            raise ValueError("boom")
    snap = m.snapshot()
    assert snap["failures"]["report.pptx"]["count"] == 1
    assert "boom" in snap["failures"]["report.pptx"]["last_error"]
    # It still timed the attempt — a slow failure is worth seeing.
    assert snap["operations"]["report.pptx"]["count"] == 1


def test_metrics_do_not_grow_without_bound():
    m = Metrics()
    for i in range(500):
        m.record_duration("report.pdf", 0.1)
        m.record_failure(f"engine.{i}", "nope")
    snap = m.snapshot()
    assert snap["operations"]["report.pdf"]["samples"] <= 200
    assert snap["operations"]["report.pdf"]["count"] == 500
    assert len(snap["failures"]) <= 200


def test_metrics_report_what_the_narrative_cache_saved(monkeypatch, tmp_path):
    from app.services import llm_cache as cache_mod
    cache = cache_mod.LLMCache(base_dir=str(tmp_path / "llm"))
    monkeypatch.setattr(cache_mod, "llm_cache", cache)
    cache.put("sys", "user", "exec", "m", "text")
    cache.get("sys", "user", "exec", "m")
    assert Metrics().snapshot()["llm_cache"]["calls_avoided"] == 1


def test_metrics_endpoint_is_admin_scoped():
    from app.services import auth
    # /api/admin/* is gated by the auth middleware; this asserts the
    # endpoint sits behind that prefix rather than beside it.
    from app.main import app
    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/api/admin/metrics" in paths
    assert "/api/metrics" not in paths
    assert "/api/health" in auth.PUBLIC_PATHS


def test_a_skipped_report_section_is_counted_as_an_engine_failure():
    from app.api import reports
    reports.metrics.reset()
    reports._count_skipped(["forecast", "stats"])
    failures = reports.metrics.snapshot()["failures"]
    assert failures["engine.forecast"]["count"] == 1
    assert failures["engine.stats"]["count"] == 1
