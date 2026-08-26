"""
services/frame_io.py — write a DataFrame to disk without making the
storage directory a code-execution path.

Datasets used to be `pickle.dump`ed. Unpickling runs whatever the file
says to run, so anything that could drop a file into the data directory
— a path-traversal bug, a shared volume, a restored-from-backup
directory — got remote code execution for free. Parquet is data: the
worst a malformed file can do is fail to parse.

Two things parquet will not do on its own, both of which turn up in real
uploads:

  * Duplicate or non-string column names. `Region, Region, 2024` is a
    perfectly ordinary spreadsheet header row and pandas keeps it, but
    arrow addresses columns by unique string name. So columns are stored
    positionally as ``c0…cN`` and the real names — with their Python
    types — go in a sidecar, which round-trips duplicates and integer
    headers exactly.
  * Object columns holding genuinely mixed types (``12`` next to
    ``"n/a"``). Arrow refuses those. Rather than fail the upload, the
    offending column is stored as text and *named* in the sidecar, so
    the caller can tell the user which column changed shape instead of
    the change happening silently.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_SIDECAR_SUFFIX = ".schema.json"


def _encode_name(name: Any) -> dict:
    """Column names survive a round trip only if their type does too:
    a spreadsheet with a 2024 header must not come back as '2024'."""
    if isinstance(name, str):
        return {"t": "str", "v": name}
    if isinstance(name, bool):
        return {"t": "bool", "v": name}
    if isinstance(name, int):
        return {"t": "int", "v": int(name)}
    if isinstance(name, float):
        return {"t": "float", "v": float(name)}
    if isinstance(name, tuple):  # MultiIndex level
        return {"t": "tuple", "v": [_encode_name(p) for p in name]}
    return {"t": "str", "v": str(name)}


def _decode_name(enc: dict) -> Any:
    kind = enc.get("t")
    val = enc.get("v")
    if kind == "int":
        return int(val)
    if kind == "float":
        return float(val)
    if kind == "bool":
        return bool(val)
    if kind == "tuple":
        return tuple(_decode_name(p) for p in val)
    return val


def _as_text(value: Any) -> Any:
    """Text for anything that is not already text, nulls left alone —
    ``pd.isna`` on a list or array raises rather than answering, so the
    check is guarded."""
    if value is None or isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return value
    except (TypeError, ValueError):
        pass
    return str(value)


def _writable(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return a frame arrow will accept, plus the names of any columns
    that had to be stored as text to get there."""
    import pyarrow as pa

    out = df
    coerced: list[str] = []
    for pos, col in enumerate(df.columns):
        series = df.iloc[:, pos]
        if series.dtype != object:
            continue
        try:
            pa.array(series, from_pandas=True)
        except Exception:
            if out is df:
                out = df.copy()
            # Only the values that are not already text change form;
            # nulls stay null so missingness is not invented.
            out.isetitem(pos, series.map(_as_text))
            coerced.append(str(col))
    return out, coerced


def write_frame(path: str, df: pd.DataFrame) -> list[str]:
    """Write `df` to `path` as parquet plus a sidecar. Returns the list
    of columns that had to be stored as text (empty in the normal case)."""
    safe, coerced = _writable(df)
    stored = safe.copy()
    stored.columns = [f"c{i}" for i in range(stored.shape[1])]

    index_name = None
    if not isinstance(stored.index, pd.RangeIndex):
        # A meaningful index (a date index, an id index) is data, so it
        # is carried across as a column rather than quietly dropped.
        index_name = "__analytiq_index__"
        stored = stored.reset_index(drop=False)
        stored.columns = [index_name] + list(stored.columns[1:])

    tmp = path + ".tmp"
    stored.to_parquet(tmp, engine="pyarrow", index=False)
    os.replace(tmp, path)

    with open(path + _SIDECAR_SUFFIX, "w", encoding="utf-8") as fh:
        json.dump({
            "columns": [_encode_name(c) for c in df.columns],
            "coerced_to_text": coerced,
            "index_column": index_name,
        }, fh)
    if coerced:
        logger.info("stored %d mixed-type column(s) as text: %s",
                    len(coerced), ", ".join(coerced[:5]))
    return coerced


def read_frame(path: str) -> pd.DataFrame | None:
    """Read back what `write_frame` wrote. Returns None if absent."""
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path, engine="pyarrow")

    sidecar = path + _SIDECAR_SUFFIX
    if not os.path.exists(sidecar):
        # Readable but unlabelled — better to hand back c0…cN than to
        # pretend the dataset is gone.
        logger.warning("no column sidecar beside %s; using stored names", path)
        return df
    with open(sidecar, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    index_col = meta.get("index_column")
    if index_col and index_col in df.columns:
        df = df.set_index(index_col)
        df.index.name = None

    names = [_decode_name(c) for c in meta.get("columns", [])]
    if len(names) == df.shape[1]:
        df.columns = names
    else:
        logger.warning("sidecar for %s lists %d columns but the file has %d",
                       path, len(names), df.shape[1])
    return df


def delete_frame(path: str) -> None:
    for p in (path, path + _SIDECAR_SUFFIX):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
