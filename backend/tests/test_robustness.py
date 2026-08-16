"""
Whether the app survives the files people actually upload.

Everything in the suite up to here feeds the engines data shaped the way
the engines expect. This file does the opposite: it takes every public
entry point and hands it every degenerate frame a real upload produces —
an empty parse, one row, a column that is entirely null, a repeated
header, a mixed-type column, infinities, a NaT date, 60,000 rows.

Run the first time, it found four crash classes:

  - **Duplicate column names**, which broke six of nine entry points from
    one upload. Two columns called "Revenue" out of a join is ordinary;
    `df["Revenue"]` then returns a DataFrame, and every `.dtype`,
    `.nunique()` and `to_numeric` call downstream raises.
  - **An empty parse** — a header-only CSV, or the wrong delimiter —
    divided by zero in the profiler.
  - **The cleaner** on a frame with no columns, calling `.str` on an
    empty index.
  - **The readiness report** swallowing any check that raised and then
    reporting the data as ready: a pass reached by not running the tests
    that would have contradicted it.

A crash is the visible failure. The one to care about is the last kind,
where the app carries on and answers anyway.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════
#  The frames
# ══════════════════════════════════════════════════════════

def _frames():
    rng = np.random.default_rng(1)
    return {
        "empty": pd.DataFrame(),
        "no_rows": pd.DataFrame({"a": pd.Series([], dtype=float),
                                 "g": pd.Series([], dtype=object)}),
        "one_row": pd.DataFrame({"revenue": [5.0], "region": ["N"],
                                 "d": [pd.Timestamp("2024-01-01")]}),
        "all_null": pd.DataFrame({"revenue": [np.nan] * 50,
                                  "region": [None] * 50}),
        "all_constant": pd.DataFrame({"revenue": [7.0] * 100,
                                      "region": ["N"] * 100}),
        "only_text": pd.DataFrame({"a": [f"x{i}" for i in range(100)],
                                   "b": ["p", "q"] * 50}),
        "only_ids": pd.DataFrame({"order_id": range(200),
                                  "ref": [f"R{i}" for i in range(200)]}),
        "dupe_colnames": pd.DataFrame(np.arange(200).reshape(100, 2),
                                      columns=["revenue", "revenue"]),
        "infinities": pd.DataFrame({
            "revenue": [np.inf, -np.inf] + list(rng.normal(1, 1, 98)),
            "g": ["a", "b"] * 50}),
        "unicode": pd.DataFrame({"région": ["Nörd", "Süd"] * 50,
                                 "revenü€": rng.normal(1, 1, 100)}),
        "mixed_types": pd.DataFrame({"revenue": [1, "two", 3.0, None] * 25,
                                     "g": ["a", "b"] * 50}),
        "negative_money": pd.DataFrame({
            "revenue": rng.normal(-100, 50, 100),
            "cost": rng.normal(50, 10, 100), "g": ["a", "b"] * 50}),
        "nat_dates": pd.DataFrame({
            "d": [pd.NaT] * 50 + list(pd.date_range("2024-01-01", periods=50)),
            "revenue": rng.normal(1, 1, 100)}),
        "very_wide": pd.DataFrame(
            {f"col{i}": rng.normal(1, 1, 40) for i in range(120)}),
    }


FRAMES = _frames()
NAMES = sorted(FRAMES)


def _entry_points():
    from app.engines.chart_engine import recommend_charts
    from app.engines.chart_exporter import generate_all_charts
    from app.engines.column_roles import left_mask, resolve
    from app.engines.dashboard_export import build_dashboard_html
    from app.engines.dashboard_spec import build_spec, layout_tiles
    from app.engines.data_cleaner import auto_clean
    from app.engines.data_profiler import profile_dataset
    from app.engines.domain_detect import detect
    from app.engines.readiness import assess_readiness
    from app.engines.story_engine import generate_story

    return {
        "detect": detect,
        "resolve_roles": resolve,
        "left_mask": left_mask,
        "profile": profile_dataset,
        "readiness": assess_readiness,
        "clean": auto_clean,
        "story": generate_story,
        "pdf_charts": generate_all_charts,
        "recommend": recommend_charts,
        "spec": lambda df: layout_tiles(build_spec(df, detect(df).domain)),
        "export": lambda df: build_dashboard_html(df, [], [], title="t"),
    }


ENTRY_POINTS = _entry_points()


@pytest.mark.parametrize("entry", sorted(ENTRY_POINTS))
@pytest.mark.parametrize("frame", NAMES)
def test_no_entry_point_raises(entry, frame):
    """The whole point of the file: an upload should never produce a
    traceback, whatever is in it."""
    ENTRY_POINTS[entry](FRAMES[frame].copy())


# ══════════════════════════════════════════════════════════
#  Duplicate column names
# ══════════════════════════════════════════════════════════

def test_duplicate_names_are_suffixed_not_dropped():
    """Silently discarding a column is the one thing a loader must never
    do."""
    from app.engines.data_loader import _clean_columns

    df = pd.DataFrame(np.arange(12).reshape(4, 3),
                      columns=["revenue", "revenue", "region"])
    out, warnings = _clean_columns(df)
    assert out.shape[1] == 3
    assert len(set(out.columns)) == 3
    assert any("duplicate" in w.lower() for w in warnings), warnings


def test_the_user_is_told_the_columns_were_renamed():
    from app.engines.data_loader import _clean_columns

    _out, warnings = _clean_columns(
        pd.DataFrame(np.arange(4).reshape(2, 2), columns=["a", "a"]))
    assert warnings and "no data was dropped" in warnings[0]


def test_the_store_hands_out_unique_names():
    from app.services.dtypes import dedupe_columns

    df = pd.DataFrame(np.arange(6).reshape(3, 2), columns=["x", "x"])
    assert list(dedupe_columns(df).columns) == ["x", "x_1"]


def test_a_frame_that_is_already_unique_is_returned_untouched():
    from app.services.dtypes import dedupe_columns

    df = pd.DataFrame({"a": [1], "b": [2]})
    assert dedupe_columns(df) is df


# ══════════════════════════════════════════════════════════
#  The cache tells the truth about the data
# ══════════════════════════════════════════════════════════

def test_a_cleaning_step_past_the_first_page_invalidates_the_cache():
    """The hash covered shape, dtypes and `df.head(100)`. Capping
    outliers on a 5,000-row file changed 2,990 values and the hash did
    not move, so the story, the ML report and the charts were all served
    from before the clean — the user cleaned their data and the report
    kept showing the old numbers."""
    from app.services.dataset_store import DatasetStore

    rng = np.random.default_rng(1)
    n = 5_000
    df = pd.DataFrame({"revenue": rng.normal(100, 10, n),
                       "g": rng.choice(list("abc"), n)})
    cleaned = df.copy()
    tail = cleaned.index[200:]
    cleaned.loc[tail, "revenue"] = cleaned.loc[tail, "revenue"].clip(95, 105)

    assert int((df.revenue != cleaned.revenue).sum()) > 1_000
    assert DatasetStore._hash_df(df) != DatasetStore._hash_df(cleaned)


def test_the_same_frame_hashes_the_same_twice():
    from app.services.dataset_store import DatasetStore

    rng = np.random.default_rng(2)
    df = pd.DataFrame({"x": rng.normal(0, 1, 500), "g": ["a", "b"] * 250})
    assert DatasetStore._hash_df(df) == DatasetStore._hash_df(df.copy())


def test_a_single_changed_cell_moves_the_hash():
    from app.services.dataset_store import DatasetStore

    df = pd.DataFrame({"x": list(range(1_000))})
    before = DatasetStore._hash_df(df)
    df.loc[900, "x"] = -1
    assert DatasetStore._hash_df(df) != before


def test_a_frame_with_unhashable_cells_still_hashes():
    from app.services.dataset_store import DatasetStore

    df = pd.DataFrame({"a": [[1, 2], [3, 4]], "b": [1, 2]})
    assert DatasetStore._hash_df(df)


def test_the_memory_cache_evicts_the_least_recently_used():
    """Re-assigning an existing dict key does not move it, so the dataset
    being actively worked on was the next one evicted."""
    from app.services.dataset_store import DatasetStore

    store = DatasetStore.__new__(DatasetStore)
    store._mem = {}
    store._MEM_LIMIT = 3
    for name in ("a", "b", "c"):
        store._touch_mem("o", name, meta=None)
    store._touch_mem("o", "a", meta=None)          # "a" used again
    store._touch_mem("o", "d", meta=None)          # forces an eviction
    assert "o/a" in store._mem, list(store._mem)
    assert "o/b" not in store._mem, list(store._mem)


# ══════════════════════════════════════════════════════════
#  Refusing beats answering anyway
# ══════════════════════════════════════════════════════════

def test_a_readiness_check_that_cannot_run_withholds_the_pass():
    """It swallowed the exception and reported the data ready — a verdict
    reached by not running the tests that would have contradicted it."""
    import app.engines.readiness as readiness

    def _explode(_df, _issues):
        raise RuntimeError("simulated")

    original = readiness._check_missing
    readiness._check_missing = _explode
    try:
        report = readiness.assess_readiness(
            pd.DataFrame({"revenue": [1.0, 2.0, 3.0] * 20,
                          "g": ["a", "b", "c"] * 20}))
    finally:
        readiness._check_missing = original

    assert not report.ready, "reported ready with a check that never ran"
    assert any("could not run" in i.issue for i in report.issues), \
        [i.issue for i in report.issues]


def test_a_working_file_still_passes_readiness():
    """The fix must not turn every report into a refusal."""
    from app.engines.readiness import assess_readiness

    rng = np.random.default_rng(3)
    df = pd.DataFrame({"revenue": rng.normal(100, 10, 300),
                       "region": rng.choice(["N", "S"], 300),
                       "d": pd.date_range("2024-01-01", periods=300)})
    report = assess_readiness(df)
    assert not [i for i in report.issues if "could not run" in i.issue]


def test_an_empty_parse_is_reported_not_crashed():
    from app.engines.data_profiler import profile_dataset

    recs = profile_dataset(pd.DataFrame()).recommendations
    assert any("parsed" in r.lower() or "delimiter" in r.lower()
               for r in recs), recs


def test_cleaning_an_empty_frame_returns_a_report():
    from app.engines.data_cleaner import auto_clean

    out, report = auto_clean(pd.DataFrame())
    assert out.empty
    assert report is not None
