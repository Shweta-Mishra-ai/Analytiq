"""
Image and video table extraction, judged on one question: does the
dataset that comes out say the same thing as the table that went in?

Two ways it can lie, and both were happening:

  - Rows go missing with nothing to say so. A screenshot of the first 25
    rows of a 480-row table extracted perfectly and was analysed as
    though it were the whole table. Frames of a scrolled spreadsheet
    whose header was transcribed with different capitalisation were
    treated as "a different table" and dropped.
  - Rows are counted twice. The overlapping region of two scroll frames
    survived `drop_duplicates()` whenever the model wrote "1,234" in one
    frame and "1234" in the next, so every total was inflated.

The merge tests below are the important ones: whatever the merge
produces is what every figure in the report is computed from.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.services.table_extractor import (
    ExtractionError,
    _merge_frame_tables,
    _norm_cell,
    _norm_header,
    _uses_comma_decimal,
    table_json_to_df,
)


def _payload(columns, rows, **extra):
    p = {"found": True, "columns": columns, "rows": rows}
    p.update(extra)
    return p


def _frame(columns, rows):
    return pd.DataFrame(rows, columns=columns)


# ══════════════════════════════════════════════════════════
#  Completeness — a partial table must never look complete
# ══════════════════════════════════════════════════════════

def test_a_cut_off_table_says_so():
    """25 visible rows of a longer table extract perfectly and are still
    the wrong dataset to average."""
    df, warnings = table_json_to_df(_payload(
        ["region", "revenue"], [["N", "100"], ["S", "200"]],
        cut_off={"top": False, "bottom": True, "left": False, "right": False}))
    assert len(df) == 2
    assert any("cut off" in w.lower() for w in warnings), warnings


def test_a_stated_total_is_compared_against_what_was_captured():
    _df, warnings = table_json_to_df(_payload(
        ["region", "revenue"], [["N", "100"], ["S", "200"]],
        stated_total_rows=480))
    joined = " ".join(warnings)
    assert "480" in joined, warnings
    assert "%" in joined, "no indication of how much of the table was captured"


def test_a_stated_total_matching_the_capture_is_not_flagged():
    _df, warnings = table_json_to_df(_payload(
        ["region", "revenue"], [["N", "100"], ["S", "200"]],
        stated_total_rows=2))
    assert not any("480" in w or "remaining rows" in w for w in warnings)


def test_other_tables_in_the_image_are_reported():
    _df, warnings = table_json_to_df(_payload(
        ["a", "b"], [["1", "2"]], other_tables=2))
    assert any("other table" in w.lower() for w in warnings), warnings


def test_unreadable_cells_are_reported():
    _df, warnings = table_json_to_df(_payload(
        ["a", "b"], [["1", None]], unreadable_cells=1))
    assert any("could not be read" in w.lower() for w in warnings), warnings


def test_a_complete_table_carries_no_completeness_noise():
    """Warning on every upload trains the reader to ignore all of them."""
    _df, warnings = table_json_to_df(_payload(
        ["a", "b"], [["1", "2"]],
        cut_off={"top": False, "bottom": False, "left": False, "right": False},
        other_tables=0, unreadable_cells=0, stated_total_rows=1))
    assert warnings == [], warnings


# ══════════════════════════════════════════════════════════
#  Merging frames — no loss, no double counting
# ══════════════════════════════════════════════════════════

def test_overlapping_scroll_frames_count_a_row_once():
    """Frame 1 shows rows 1-4, frame 2 shows rows 3-6. The two shared
    rows must appear once — and the model transcribes them differently in
    each frame, which is exactly what drop_duplicates could not catch."""
    f1 = _frame(["id", "amount"],
                [["1", "1000"], ["2", "2000"], ["3", "3,000"], ["4", "4,000"]])
    f2 = _frame(["id", "amount"],
                [["3", "3000"], ["4", "4000"], ["5", "5000"], ["6", "6000"]])
    merged, warnings = _merge_frame_tables([(f1, []), (f2, [])], 2)
    assert len(merged) == 6, \
        "expected 6 distinct rows, got {}:\n{}".format(len(merged), merged)
    assert sorted(merged["id"].astype(str)) == ["1", "2", "3", "4", "5", "6"]
    assert any("more than one frame" in w for w in warnings), warnings


def test_genuinely_repeated_rows_are_not_collapsed():
    """Two orders with identical values are two orders. Collapsing them
    silently changes every count and total in the report."""
    f1 = _frame(["product", "qty"],
                [["widget", "1"], ["widget", "1"], ["gizmo", "2"]])
    merged, _warnings = _merge_frame_tables([(f1, [])], 1)
    assert len(merged) == 3, "a legitimate duplicate row was dropped"
    assert (merged["product"] == "widget").sum() == 2


def test_repeated_rows_seen_in_two_frames_are_still_counted_once_each():
    f1 = _frame(["product", "qty"], [["widget", "1"], ["widget", "1"]])
    f2 = _frame(["product", "qty"], [["widget", "1"], ["widget", "1"]])
    merged, _w = _merge_frame_tables([(f1, []), (f2, [])], 2)
    assert len(merged) == 2, \
        "the same two rows filmed twice became {} rows".format(len(merged))


def test_a_header_read_differently_does_not_split_the_table():
    """"Order Date" and "order  date" are one column. Treated as two
    tables, the second frame's rows are discarded as "another table"."""
    f1 = _frame(["Order Date", "Revenue"], [["2024-01-01", "10"]])
    f2 = _frame(["order  date", "revenue"], [["2024-01-02", "20"]])
    merged, warnings = _merge_frame_tables([(f1, []), (f2, [])], 2)
    assert len(merged) == 2, \
        "capitalisation split one table in two and lost a row"
    assert not any("different tables" in w for w in warnings)


def test_genuinely_different_tables_are_reported_not_hidden():
    f1 = _frame(["region", "revenue"], [["N", "10"], ["S", "20"]])
    f2 = _frame(["employee", "salary", "dept"], [["A", "1", "X"]])
    merged, warnings = _merge_frame_tables([(f1, []), (f2, [])], 2)
    assert list(merged.columns) == ["region", "revenue"], \
        "the smaller table was used instead of the larger one"
    assert any("NOT in this dataset" in w for w in warnings), warnings


def test_frames_that_failed_extraction_are_disclosed():
    """Rows visible only in a skipped frame are gone, and the user has to
    know that before trusting a total."""
    f1 = _frame(["a"], [["1"]])
    _merged, warnings = _merge_frame_tables([(f1, [])], 4)
    assert any("skipped" in w for w in warnings), warnings
    assert any("missing" in w for w in warnings), warnings


def test_per_frame_warnings_survive_the_merge():
    f1 = _frame(["a"], [["1"]])
    _merged, warnings = _merge_frame_tables(
        [(f1, ["The table is cut off in the source image"])], 1)
    assert any("cut off" in w for w in warnings)


def test_merged_frames_keep_a_consistent_column_order():
    f1 = _frame(["id", "amount"], [["1", "10"]])
    f2 = _frame(["amount", "id"], [["20", "2"]])
    merged, _w = _merge_frame_tables([(f1, []), (f2, [])], 2)
    assert list(merged.columns) == ["id", "amount"]
    # the row from the reordered frame must land in the right columns
    assert set(merged["id"].astype(str)) == {"1", "2"}
    assert set(merged["amount"].astype(str)) == {"10", "20"}


# ══════════════════════════════════════════════════════════
#  Normalisation helpers
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("a,b", [
    ("Order Date", "order  date"),
    ("Revenue ($)", "revenue $"),
    ("Total_Sales", "total sales"),
])
def test_headers_that_differ_only_in_noise_normalise_together(a, b):
    assert _norm_header(a) == _norm_header(b)


def test_headers_with_different_words_stay_distinct():
    assert _norm_header("revenue") != _norm_header("revenues_2024")


@pytest.mark.parametrize("a,b", [
    ("1,234", "1234"),
    (" 1234 ", "1234"),
    ("$1,234", "1234"),
    ("1234.0", "1234"),
    (None, ""),
])
def test_cells_that_differ_only_in_formatting_normalise_together(a, b):
    assert _norm_cell(a) == _norm_cell(b)


def test_different_values_do_not_normalise_together():
    assert _norm_cell("1234") != _norm_cell("12345")
    assert _norm_cell("north") != _norm_cell("south")


# ══════════════════════════════════════════════════════════
#  European decimals — a factor-of-1000 error
# ══════════════════════════════════════════════════════════

def test_european_decimal_column_is_read_correctly():
    """"1.234,56" stripped of commas becomes 1.23456 — every figure
    derived from the column is then wrong by a factor of a thousand."""
    df, warnings = table_json_to_df(_payload(
        ["item", "price"],
        [["a", "1.234,56"], ["b", "2.500,00"], ["c", "999,90"],
         ["d", "12.000,10"]]))
    assert df["price"].iloc[0] == pytest.approx(1234.56)
    assert df["price"].iloc[1] == pytest.approx(2500.00)
    assert df["price"].iloc[3] == pytest.approx(12000.10)
    assert any("European" in w for w in warnings), warnings


def test_us_format_is_still_read_correctly():
    df, warnings = table_json_to_df(_payload(
        ["item", "price"],
        [["a", "1,234.56"], ["b", "2,500.00"], ["c", "999.90"],
         ["d", "12,000.10"]]))
    assert df["price"].iloc[0] == pytest.approx(1234.56)
    assert df["price"].iloc[3] == pytest.approx(12000.10)
    assert not any("European" in w for w in warnings)


def test_indian_grouping_is_read_correctly():
    df, _w = table_json_to_df(_payload(
        ["item", "price"],
        [["a", "1,23,456.78"], ["b", "12,34,567.00"], ["c", "999.90"]]))
    assert df["price"].iloc[0] == pytest.approx(123456.78)
    assert df["price"].iloc[1] == pytest.approx(1234567.00)


def test_comma_decimal_detection_needs_evidence():
    """A single "1.234" is ambiguous and must not flip the whole column
    into European mode."""
    assert not _uses_comma_decimal(pd.Series(["1.234"]))
    assert not _uses_comma_decimal(pd.Series(["1,234.56", "2,000.00"]))
    assert _uses_comma_decimal(pd.Series(["1.234,56", "2.000,00", "99,90"]))


def test_percentage_columns_state_what_was_stored():
    """45% stored as 45 is fine; 45% stored as 45 and reported as a
    proportion is not, and the reader cannot tell which happened."""
    df, warnings = table_json_to_df(_payload(
        ["region", "share"],
        [["N", "45%"], ["S", "30%"], ["E", "25%"]]))
    assert df["share"].iloc[0] == 45
    assert any("45" in w and "0.45" in w for w in warnings), warnings


# ══════════════════════════════════════════════════════════
#  Existing guarantees still hold
# ══════════════════════════════════════════════════════════

def test_no_table_found_is_a_clear_error():
    with pytest.raises(ExtractionError):
        table_json_to_df({"found": False, "columns": [], "rows": []})


def test_short_rows_are_padded_and_long_rows_reported():
    df, warnings = table_json_to_df(_payload(
        ["a", "b", "c"], [["1", "2"], ["1", "2", "3", "4"]]))
    assert df.shape == (2, 3)
    assert any("truncated" in w for w in warnings)


def test_non_numeric_cells_in_a_numeric_column_are_preserved():
    """Coercing "Pending" to NaN destroys the only record that the value
    was pending."""
    df, warnings = table_json_to_df(_payload(
        ["item", "amount"],
        [["a", "100"], ["b", "200"], ["c", "300"], ["d", "400"],
         ["e", "500"], ["f", "Pending"]]))
    assert "Pending" in df["amount"].astype(str).tolist()
    assert any("kept as text" in w for w in warnings)
