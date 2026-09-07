"""
A stored frame is a parquet file AND a sidecar naming its columns, and
the two only mean anything as a pair.

Both writes used a single shared temp name, `<path>.tmp`, and happened
outside any lock. Measured with four writers of a 60,000-row frame and
four of a 10-row frame hitting one path: **39 of 96 writes failed** with

    FileNotFoundError: ... 'race.parquet.tmp' -> 'race.parquet'

because one writer replaced the temp file out from under another.

The half nobody would have caught in a log is worse. The sidecar was
written *after* the parquet was replaced, so writer A's parquet could
come to rest beside writer B's column list — and every column in that
frame would then be read back under the wrong name. Silent, plausible,
and wrong.

Two writers land on one path whenever anything reads a dataset while
something else saves it: a cleaning step and a report build, two browser
tabs, a scheduled export.
"""
import os
import threading

import numpy as np
import pandas as pd
import pytest

from app.services.frame_io import read_frame, write_frame


@pytest.fixture()
def path(tmp_path) -> str:
    return str(tmp_path / "frame.parquet")


BIG_COLS = ["alpha", "beta"]
SMALL_COLS = ["gamma", "delta"]


def _big() -> pd.DataFrame:
    return pd.DataFrame({"alpha": np.arange(20_000),
                         "beta": np.random.rand(20_000)})


def _small() -> pd.DataFrame:
    return pd.DataFrame({"gamma": np.arange(10),
                         "delta": np.random.rand(10)})


def test_concurrent_writers_to_one_path_all_succeed(path):
    failures: list = []

    def writer(frame_factory):
        for _ in range(10):
            try:
                write_frame(path, frame_factory())
            except Exception as exc:                    # noqa: BLE001
                failures.append("{}: {}".format(type(exc).__name__, exc))

    threads = ([threading.Thread(target=writer, args=(_big,)) for _ in range(3)]
               + [threading.Thread(target=writer, args=(_small,)) for _ in range(3)])
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert failures == [], failures[:3]


def test_a_frame_is_never_paired_with_another_writes_columns(path):
    """The silent half: a parquet beside the wrong sidecar renames every
    column in the frame."""
    mismatches: list = []
    stop = threading.Event()

    def writer(frame_factory):
        for _ in range(10):
            write_frame(path, frame_factory())

    def reader():
        while not stop.is_set():
            try:
                got = read_frame(path)
            except Exception:                            # noqa: BLE001
                continue
            if got is None:
                continue
            expected = BIG_COLS if len(got) > 100 else SMALL_COLS
            if list(got.columns) != expected:
                mismatches.append((len(got), list(got.columns)))

    readers = [threading.Thread(target=reader) for _ in range(3)]
    writers = ([threading.Thread(target=writer, args=(_big,)) for _ in range(2)]
               + [threading.Thread(target=writer, args=(_small,)) for _ in range(2)])
    for t in readers + writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    for t in readers:
        t.join()

    assert mismatches == [], mismatches[:3]


def test_no_temp_files_are_left_behind(path, tmp_path):
    """A per-writer temp name is only an improvement if it is cleaned
    up; otherwise the data directory fills with orphans."""
    for _ in range(5):
        write_frame(path, _small())
    leftovers = [f for f in os.listdir(tmp_path) if ".tmp" in f]
    assert leftovers == []


def test_the_frame_still_round_trips(path):
    """The locking must not have changed what a write means."""
    original = pd.DataFrame({
        "name": ["a", "b", "c"],
        "when": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "value": [1.5, 2.5, 3.5],
    })
    write_frame(path, original)
    back = read_frame(path)
    pd.testing.assert_frame_equal(original, back)


# ══════════════════════════════════════════════════════════
#  The store on top of it
# ══════════════════════════════════════════════════════════

def test_the_store_survives_mixed_concurrent_traffic(tmp_path):
    """Reads, cache writes and active-frame updates from many threads at
    once — the shape of a busy few seconds in the real app."""
    # A fresh store rooted at tmp_path, NOT importlib.reload(app.config):
    # reloading builds a second config object, so modules holding
    # `from app.config import config` keep the first one and later tests
    # in the same session see a different object than the code does.
    # That leak made an unrelated warehouse test fail only when the
    # suite ran in full.
    from app.services.dataset_store import DatasetStore
    store = DatasetStore(base_dir=str(tmp_path / "datasets"))

    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"a": rng.normal(0, 1, 400), "b": range(400)})
    ids = [store.create("local", frame, f"d{i}.csv", 0.01).dataset_id
           for i in range(4)]

    errors: list = []

    def worker(n):
        import random
        try:
            for _ in range(60):
                ds = random.choice(ids)
                roll = random.random()
                if roll < 0.5:
                    got = store.get_df("local", ds)
                    if got is not None:
                        assert len(got) == 400, "short frame: %d" % len(got)
                elif roll < 0.75:
                    store.cache_set("local", ds, f"k{n}", {"v": n})
                else:
                    got = store.get_df("local", ds)
                    if got is not None:
                        store.update_active("local", ds, got.assign(c=n))
        except Exception as exc:                         # noqa: BLE001
            errors.append("{}: {}".format(type(exc).__name__, exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], errors[:3]
    for ds in ids:
        assert store.get_df("local", ds) is not None


def test_deleting_a_dataset_reports_that_it_deleted_it(tmp_path):
    """`delete` returned False whenever only the on-disk directory was
    missing, so the API answered 404 about a dataset it had just removed
    from memory — the row vanished and an error appeared at once."""
    import shutil

    from app.services.dataset_store import DatasetStore
    store = DatasetStore(base_dir=str(tmp_path / "datasets"))

    meta = store.create("local", pd.DataFrame({"a": [1, 2, 3]}), "x.csv", 0.01)
    shutil.rmtree(store._dir("local", meta.dataset_id), ignore_errors=True)

    assert store.delete("local", meta.dataset_id) is True
    assert store.get_df("local", meta.dataset_id) is None
    # And a genuinely unknown id is still not found.
    assert store.delete("local", "does-not-exist") is False
