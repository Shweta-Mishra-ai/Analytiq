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
import threading
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


# One lock per destination path, created on demand. A single global
# lock would serialise every dataset write in the process; a lock per
# path only serialises writers who would otherwise collide.
_PATH_LOCKS: dict = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _path_lock(path: str) -> threading.Lock:
    key = os.path.abspath(path)
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = _PATH_LOCKS[key] = threading.Lock()
        return lock


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

    # A frame is a parquet file AND a sidecar naming its columns, and
    # the two only mean anything as a pair. Two concurrent writes to one
    # path used to share a single "<path>.tmp": measured at 4 writers x
    # 12 writes, 39 of 96 failed with
    #
    #     FileNotFoundError: ... 'race.parquet.tmp' -> 'race.parquet'
    #
    # because one writer replaced the temp file out from under another.
    # The half nobody would have noticed is worse: the sidecar is
    # written after the replace, so writer A's parquet could end up
    # beside writer B's column list, and every column in the frame would
    # then be read back under the wrong name.
    #
    # Two changes. A temp name unique to this writer, so nobody removes
    # anyone else's; and a lock per path, so the parquet and its sidecar
    # land together or not at all.
    tmp = "{}.{}.{}.tmp".format(path, os.getpid(), threading.get_ident())
    sidecar = path + _SIDECAR_SUFFIX
    sidecar_tmp = tmp + _SIDECAR_SUFFIX

    with _path_lock(path):
        try:
            stored.to_parquet(tmp, engine="pyarrow", index=False)
            with open(sidecar_tmp, "w", encoding="utf-8") as fh:
                json.dump({
                    "columns": [_encode_name(c) for c in df.columns],
                    "coerced_to_text": coerced,
                    "index_column": index_name,
                }, fh)
            os.replace(tmp, path)
            os.replace(sidecar_tmp, sidecar)
        finally:
            for leftover in (tmp, sidecar_tmp):
                if os.path.exists(leftover):
                    try:
                        os.remove(leftover)
                    except OSError:
                        logger.debug("could not remove %s", leftover,
                                     exc_info=True)
    if coerced:
        logger.info("stored %d mixed-type column(s) as text: %s",
                    len(coerced), ", ".join(coerced[:5]))
    return coerced


def read_frame(path: str) -> pd.DataFrame | None:
    """Read back what `write_frame` wrote. Returns None if absent."""
    if not os.path.exists(path):
        return None

    # Both halves under the same lock the writer takes. Making the WRITE
    # atomic was not enough: the reader opened the parquet and the
    # sidecar as two separate operations, so a writer could swap both
    # between them and hand back one frame's data under another frame's
    # column names. Measured after the write fix and before this one:
    # 86 mismatched reads in a few seconds of contention — a 20,000-row
    # frame whose columns were named for the 10-row frame.
    #
    # Readers on the same path serialise with each other, which is the
    # cost of the guarantee. Different datasets are different paths and
    # do not wait on one another; a hot dataset is served from the
    # in-memory cache above this layer and never reaches here at all.
    sidecar = path + _SIDECAR_SUFFIX
    with _path_lock(path):
        df = pd.read_parquet(path, engine="pyarrow")
        if not os.path.exists(sidecar):
            # Readable but unlabelled — better to hand back c0…cN than
            # to pretend the dataset is gone.
            logger.warning("no column sidecar beside %s; using stored names",
                           path)
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

