"""
A mostly-numeric column with a few legitimate non-numeric values —
"Pending", "N/A" — was silently having those values destroyed into NaN
by the >80%-numeric auto-conversion heuristic, both on the ordinary
upload path and in AI table extraction from an image or a video. A cell
that fails to convert now keeps what it said.

This was a script that printed ticks and called sys.exit(); it is now
ordinary pytest, so a failure names itself and the rest still run.
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.table_extractor import table_json_to_df


@pytest.fixture()
def client():
    return TestClient(app)


# ══════════════════════════════════════════════════════════
#  The upload path
# ══════════════════════════════════════════════════════════

@pytest.fixture()
def mostly_numeric_upload(client):
    """Nine numbers and one 'Pending' — over the 80% threshold, so the
    column converts, and the tenth value is the one at risk."""
    csv = "Order,Discount\n" + "".join(
        "{},{}\n".format(i + 1, (i + 1) * 10) for i in range(9)) + "10,Pending\n"

    res = client.post(
        "/api/datasets/upload",
        files={"file": ("orders.csv", io.BytesIO(csv.encode()), "text/csv")})
    assert res.status_code == 200, res.text

    ds_id = res.json()["meta"]["dataset_id"]
    return client.get("/api/datasets/{}/preview".format(ds_id)).json()["records"]


def test_a_90_percent_numeric_column_keeps_its_one_word(mostly_numeric_upload):
    assert mostly_numeric_upload[-1]["Discount"] == "Pending"


def test_and_the_numbers_in_it_still_convert(mostly_numeric_upload):
    """The fix must not be "give up on the column"."""
    assert mostly_numeric_upload[0]["Discount"] == 10.0


# ══════════════════════════════════════════════════════════
#  Table extraction from an image or a video frame
# ══════════════════════════════════════════════════════════

@pytest.fixture()
def extracted():
    payload = {
        "found": True, "columns": ["Region", "Revenue"],
        "rows": [["R1", "12500"], ["R2", "8200"], ["R3", "9800"],
                 ["R4", "1000"], ["R5", "4300"], ["R6", "7700"],
                 ["R7", "2200"], ["R8", "5600"], ["R9", "3300"],
                 ["R10", "Pending"]],
    }
    return table_json_to_df(payload)


def test_extraction_keeps_a_non_numeric_cell(extracted):
    df, _ = extracted
    assert df.loc[df["Region"] == "R10", "Revenue"].iloc[0] == "Pending"


def test_extraction_still_converts_the_numeric_rows(extracted):
    df, _ = extracted
    assert df.loc[df["Region"] == "R1", "Revenue"].iloc[0] == 12500.0


def test_extraction_says_it_kept_the_value(extracted):
    """Silently correct is not enough — the reader has to know one cell
    in a numeric column is text."""
    _, warnings = extracted
    assert any("kept as text" in w for w in warnings)


# ══════════════════════════════════════════════════════════
#  Rows the model got wrong
# ══════════════════════════════════════════════════════════

@pytest.fixture()
def malformed():
    payload = {
        "found": True, "columns": ["A", "B"],
        "rows": [["x", "1"], ["y", "2"], ["z", "3", "extra-cell"], "not-a-row"],
    }
    return table_json_to_df(payload)


def test_a_malformed_row_is_dropped_not_guessed_at(malformed):
    df, _ = malformed
    assert len(df) == 3


def test_the_skipped_row_is_reported(malformed):
    _, warnings = malformed
    assert any("unexpected format" in w for w in warnings)


def test_the_oversized_row_is_reported_as_truncated(malformed):
    _, warnings = malformed
    assert any("truncated" in w for w in warnings)


def test_an_accounting_negative_parses_as_negative():
    """(500) is how a finance export writes -500. Reading it as 500
    flips the sign on every figure derived from the column."""
    payload = {
        "found": True, "columns": ["Item", "Amount"],
        "rows": [["A", "1000"], ["B", "2000"], ["C", "3000"], ["D", "4000"],
                 ["E", "(500)"]],
    }
    df, _ = table_json_to_df(payload)

    assert df.loc[df["Item"] == "E", "Amount"].iloc[0] == -500
