"""Tests for services/serialize.to_jsonable."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

from app.services.serialize import df_records, to_jsonable


class Range(NamedTuple):
    low: float
    high: float
    unit: str


@dataclass
class Boxed:
    name: str
    rng: Range


def test_namedtuple_serialises_as_object_not_array():
    """Regression: a NamedTuple is a tuple, so without explicit handling it
    serialised positionally and every field name was lost. The industry
    benchmark UI read .low/.high off what arrived as [10, 15, "%"] and
    rendered blanks with no error anywhere."""
    out = to_jsonable(Range(10, 15, "%"))
    assert out == {"low": 10, "high": 15, "unit": "%"}


def test_namedtuple_nested_in_dataclass_also_becomes_an_object():
    out = to_jsonable(Boxed(name="attrition", rng=Range(1, 2, "x")))
    assert out["rng"] == {"low": 1, "high": 2, "unit": "x"}


def test_plain_tuple_still_serialises_as_a_list():
    assert to_jsonable((1, 2, 3)) == [1, 2, 3]


def test_numpy_and_nan_handling():
    assert to_jsonable(np.int64(5)) == 5
    assert to_jsonable(np.float64(1.5)) == 1.5
    assert to_jsonable(float("nan")) is None
    assert to_jsonable(float("inf")) is None
    assert to_jsonable(np.bool_(True)) is True


def test_dataframe_becomes_records_payload():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out = to_jsonable(df)
    assert out["columns"] == ["a", "b"]
    assert out["total_rows"] == 2
    assert out["records"][0]["a"] == 1


def test_df_records_nulls_become_none():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [np.inf, 2.0]})
    out = df_records(df)
    assert out["records"][1]["a"] is None
    assert out["records"][0]["b"] is None


def test_timestamp_becomes_iso_string():
    out = to_jsonable(pd.Timestamp("2026-01-02T03:04:05"))
    assert isinstance(out, str) and out.startswith("2026-01-02")


def test_deeply_nested_structure_terminates():
    obj: object = {"x": 1}
    for _ in range(20):
        obj = {"nested": obj}
    assert to_jsonable(obj) is not None  # must not recurse forever


def test_nan_inside_nested_container_is_nulled():
    out = to_jsonable({"vals": [1.0, math.nan, 3.0]})
    assert out["vals"] == [1.0, None, 3.0]
