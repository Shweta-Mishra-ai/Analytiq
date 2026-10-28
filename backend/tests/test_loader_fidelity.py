"""
tests/test_loader_fidelity.py — what arrives in the dataset is what was in
the file: the right characters, every row or a count of those skipped, the
dates as dates, and one column per name.
"""
from __future__ import annotations

import io
import json

import pandas as pd
import pytest

from app.api.datasets import _UploadShim
from app.engines.data_loader import load_file


def _load(name: str, data: bytes):
    r = load_file(_UploadShim(name, data))
    assert r.success, r.error
    return r


def test_a_windows_encoded_csv_keeps_its_accented_characters():
    r = _load("cp.csv", "city,amount\ncafé,1\nZürich,2\n".encode("cp1252"))
    assert r.df["city"].tolist() == ["café", "Zürich"]
    assert any("not UTF-8" in w for w in r.warnings)


def test_utf8_and_bom_files_load_without_an_encoding_warning():
    r = _load("u.csv", "﻿id,city\n1,café\n2,b\n".encode("utf-8"))
    assert list(r.df.columns) == ["id", "city"]
    assert r.df["city"].iloc[0] == "café"
    assert not any("UTF-8" in w for w in r.warnings)


def test_skipped_malformed_lines_are_counted_for_the_user():
    r = _load("ragged.csv", b"a,b\n1,2\n3,4,5,6\n7,8\n9,10,11\n12,13\n")
    assert len(r.df) == 3
    assert any("2 line(s)" in w and "skipped" in w for w in r.warnings)


def test_day_first_dates_are_parsed_as_dates():
    pairs = [(1, 2), (5, 3), (13, 1), (20, 4), (28, 2), (2, 6), (15, 7), (30, 8)]
    body = "order_date,amount\n" + "\n".join(
        f"{d:02d}/{m:02d}/2024,{i}" for i, (d, m) in enumerate(pairs))
    r = _load("dm.csv", body.encode())
    col = r.df["order_date"]
    assert pd.api.types.is_datetime64_any_dtype(col)
    assert col.iloc[0] == pd.Timestamp("2024-02-01")     # 01/02 is 1 February
    assert col.iloc[2] == pd.Timestamp("2024-01-13")


def test_month_first_dates_are_unchanged():
    body = "order_date,v\n" + "\n".join(f"{m:02d}/{d:02d}/2024,1"
                                         for m, d in [(1, 31), (2, 15), (12, 25), (3, 1)])
    col = _load("md.csv", body.encode()).df["order_date"]
    assert col.iloc[0] == pd.Timestamp("2024-01-31")


def test_nested_json_records_are_flattened_and_lists_kept_as_text():
    data = [{"id": 1, "user": {"name": "a", "age": 30}, "tags": ["x", "y"]},
            {"id": 2, "user": {"name": "b", "age": 40}, "tags": []}]
    df = _load("n.json", json.dumps(data).encode()).df
    assert {"user.name", "user.age"} <= set(df.columns)
    assert df["tags"].iloc[0] == '["x", "y"]'
    df["tags"].nunique()            # hashable: no TypeError


@pytest.mark.parametrize("header", [b"a,a ,b", b"Column_1,,b"])
def test_column_names_are_unique_after_cleaning(header):
    df = _load("d.csv", header + b"\n1,2,3\n4,5,6\n").df
    assert df.columns.is_unique
    assert df.shape[1] == 3


def test_an_uploaded_file_with_duplicate_names_survives_analysis(client):
    body = b"amount,amount ,region\n" + b"\n".join(
        f"{i},{i * 2},{'ns'[i % 2]}".encode() for i in range(40))
    r = client.post("/api/datasets/upload",
                    files={"file": ("d.csv", io.BytesIO(body), "text/csv")})
    assert r.status_code == 200, r.text
    ds = r.json()["meta"]["dataset_id"]
    for path in ("stats", "eda", "insights"):
        assert client.get(f"/api/analytics/{ds}/{path}").status_code == 200


def test_legacy_xls_reader_is_installed():
    # The upload page accepts .xls; without xlrd every such upload failed.
    import xlrd  # noqa: F401


# ── dates by content, names by word, codes as written ─────

@pytest.mark.parametrize("header,rows", [
    ("day", ["2024-01-31", "2024-02-01", "2024-03-05", "2024-04-09"]),
    ("logged", ["2024-01-31T10:00:00Z", "2024-02-01T11:30:00Z", "2024-02-03T09:00:00Z"]),
    ("period", ['"Jan 5, 2024"', '"Feb 6, 2024"', '"Mar 7, 2024"']),
    ("paid_date", ["31/01/2024", "05/02/2024", "13/03/2024"]),     # 'id' inside 'paid'
])
def test_date_columns_are_recognised_whatever_they_are_called(header, rows):
    body = f"{header},v\n" + "\n".join(f"{r},{i}" for i, r in enumerate(rows))
    col = _load("d.csv", body.encode()).df[header]
    assert pd.api.types.is_datetime64_any_dtype(col), col.tolist()


@pytest.mark.parametrize("header,rows", [
    ("version", ["1.2.3", "2.0.1", "3.1.4"]),
    ("note", ["2024-01-01", "hello", "world", "foo"]),
    ("year", ["2021", "2022", "2023"]),
])
def test_values_that_are_not_dates_stay_as_they_are(header, rows):
    body = f"{header},v\n" + "\n".join(f"{r},{i}" for i, r in enumerate(rows))
    col = _load("d.csv", body.encode()).df[header]
    assert not pd.api.types.is_datetime64_any_dtype(col)


@pytest.mark.parametrize("header,rows", [
    ("zip", ["02134", "10001", "00501"]),
    ("account_no", ["000123", "000456", "000789"]),
])
def test_zero_padded_codes_keep_their_zeros(header, rows):
    body = f"{header},v\n" + "\n".join(f"{r},{i}" for i, r in enumerate(rows))
    r = _load("z.csv", body.encode())
    assert r.df[header].tolist() == rows
    assert any("leading zeros" in w for w in r.warnings)


def test_ordinary_integers_are_still_numbers():
    df = _load("n.csv", b"qty,delta\n0,-5\n10,0\n5,3\n").df
    assert pd.api.types.is_integer_dtype(df["qty"]) and pd.api.types.is_integer_dtype(df["delta"])


def test_a_mostly_numeric_column_named_with_id_inside_a_word_is_converted():
    # "amount_paid" used to be skipped because "paid" contains "id".
    body = "amount_paid,v\n" + "\n".join(f"{x},{i}" for i, x in enumerate(["10.5"] * 9 + ["Pending"]))
    col = _load("p.csv", body.encode()).df["amount_paid"]
    assert col.iloc[0] == 10.5 and col.iloc[-1] == "Pending"
