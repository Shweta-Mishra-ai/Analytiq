"""
services/serialize.py — convert engine outputs (dataclasses, numpy,
pandas, plotly) into JSON-safe structures for API responses.
"""
from __future__ import annotations
import logging

import dataclasses
import math
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    if _depth > 12:
        return str(obj)
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x, _depth + 1) for x in obj.tolist()]
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict(), _depth + 1)
    if isinstance(obj, pd.DataFrame):
        return df_records(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v, _depth + 1)
                for k, v in dataclasses.asdict(obj).items()}
    # NamedTuple before the generic tuple branch below: it IS a tuple, so
    # without this it serialises to a positional array and every field name
    # is lost. Clients then have to index by position — which silently
    # broke the industry-benchmark UI (it read .low/.high off what arrived
    # as [10, 15, "%", "..."]).
    if isinstance(obj, tuple) and hasattr(obj, "_asdict"):
        return {k: to_jsonable(v, _depth + 1) for k, v in obj._asdict().items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x, _depth + 1) for x in obj]
    # plotly figure
    if hasattr(obj, "to_plotly_json"):
        import plotly.io as pio
        import json
        return json.loads(pio.to_json(obj))
    if hasattr(obj, "__dict__"):
        return {k: to_jsonable(v, _depth + 1)
                for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def df_records(df: pd.DataFrame, limit: int = 1000) -> dict:
    """DataFrame → {columns, records, total_rows} with NaN→None."""
    head = df.head(limit).copy()
    for col in head.columns:
        if pd.api.types.is_datetime64_any_dtype(head[col]):
            head[col] = head[col].astype(str)
    head = head.replace([np.inf, -np.inf], np.nan)
    records = head.where(pd.notna(head), None).to_dict(orient="records")
    return {
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "records": to_jsonable(records),
        "total_rows": len(df),
        "truncated": len(df) > limit,
    }
