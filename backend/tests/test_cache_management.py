"""
Memory and disk held by caches stay bounded: working frames within a byte
budget (least recently used evicted), analysis results released with
their frame, stale cache files and expired report files deleted, and
login-failure records pruned.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from app.services.dataset_store import DatasetStore


def _frame(rows=20_000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.normal(size=rows), "b": rng.integers(0, 9, rows),
                         "c": rng.choice(["x", "y", "z"], rows)})


def _store(tmp_path, budget_mb):
    s = DatasetStore(base_dir=str(tmp_path))
    s._MEM_BUDGET = int(budget_mb * 1024 * 1024)
    return s


def test_frames_are_held_within_a_byte_budget(tmp_path):
    s = _store(tmp_path, budget_mb=3)
    ids = [s.create("u", _frame(seed=i), f"f{i}.csv", 1).dataset_id for i in range(6)]
    assert s.memory_bytes() <= s._MEM_BUDGET
    assert len(s._mem) < 6
    # Evicted frames come back from disk, unchanged.
    first = s.get_df("u", ids[0])
    pd.testing.assert_frame_equal(first.reset_index(drop=True),
                                  _frame(seed=0).reset_index(drop=True), check_dtype=False)


def test_the_least_recently_used_frame_goes_first(tmp_path):
    s = _store(tmp_path, budget_mb=100)
    a = s.create("u", _frame(seed=1), "a.csv", 1).dataset_id
    b = s.create("u", _frame(seed=2), "b.csv", 1).dataset_id
    s.get_df("u", a)                                   # a is now the most recent
    s._MEM_BUDGET = s._mem[s._mkey("u", a)]["bytes"] + 5000     # room for a and the small c
    s.create("u", _frame(rows=10, seed=3), "c.csv", 1)
    assert s._mkey("u", a) in s._mem
    assert s._mkey("u", b) not in s._mem


def test_the_original_upload_is_not_held_in_memory(tmp_path):
    s = _store(tmp_path, budget_mb=100)
    ds = s.create("u", _frame(), "a.csv", 1).dataset_id
    assert set(s._mem[s._mkey("u", ds)]) >= {"active", "meta", "bytes"}
    assert "raw" not in s._mem[s._mkey("u", ds)]
    assert s.get_raw_df("u", ds) is not None


def test_analysis_results_leave_memory_with_their_frame(tmp_path):
    s = _store(tmp_path, budget_mb=100)
    a = s.create("u", _frame(seed=1), "a.csv", 1).dataset_id
    s.cache_set("u", a, "stats", {"big": "result"})
    assert s._mkey("u", a) in s._caches
    s._MEM_BUDGET = 1                                  # everything but the newest goes
    s.create("u", _frame(rows=10, seed=2), "b.csv", 1)
    assert s._mkey("u", a) not in s._caches
    assert s.cache_get("u", a, "stats") == {"big": "result"}   # still on disk


def test_stale_cache_files_are_deleted_on_a_new_revision(tmp_path):
    s = _store(tmp_path, budget_mb=100)
    ds = s.create("u", _frame(), "a.csv", 1).dataset_id
    s.cache_set("u", ds, "stats", 1)
    s.cache_set("u", ds, "bi", 2)
    folder = os.path.join(str(tmp_path), "u", ds)
    assert sum(n.startswith("cache_") for n in os.listdir(folder)) == 2
    s.update_active("u", ds, _frame(seed=9))
    assert not [n for n in os.listdir(folder) if n.startswith("cache_")]
    assert s.cache_get("u", ds, "stats") is None


def test_expired_report_downloads_are_removed(tmp_path, monkeypatch):
    from app import mcp_server
    monkeypatch.setattr(mcp_server, "report_dir", lambda: str(tmp_path))
    old, new = tmp_path / "old.pdf", tmp_path / "new.pdf"
    old.write_bytes(b"%PDF")
    new.write_bytes(b"%PDF")
    past = time.time() - mcp_server.DOWNLOAD_TTL - 60
    os.utime(old, (past, past))
    assert mcp_server.sweep_reports() == 1
    assert not old.exists() and new.exists()


def test_login_failure_records_do_not_grow_without_bound(monkeypatch):
    from app.services.auth import LoginThrottle
    t = LoginThrottle(window_sec=60)
    monkeypatch.setattr(LoginThrottle, "_MAX_KEYS", 100)
    clock = [1000.0]
    monkeypatch.setattr("app.services.auth.time.time", lambda: clock[0])
    for i in range(100):
        t.record_failure(f"made-up-{i}")
    clock[0] += 120                                     # all of those expired
    t.record_failure("one-more")
    assert len(t._failures) <= 2


def test_memory_stats_are_reported(tmp_path):
    s = _store(tmp_path, budget_mb=100)
    s.create("u", _frame(), "a.csv", 1)
    stats = s.memory_stats()
    assert stats["datasets_in_memory"] == 1 and stats["memory_mb"] > 0
