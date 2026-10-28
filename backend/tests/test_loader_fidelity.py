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
