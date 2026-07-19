"""
services/dataset_store.py — server-side replacement for the old
Streamlit session_manager.

Each uploaded dataset gets a UUID. We keep:
  - raw df      (as uploaded, never mutated)
  - active df   (after cleaning steps)
  - per-dataset analysis caches keyed by a content hash of the active df,
    so caches invalidate automatically when the data changes.

DataFrames are pickled to disk (preserves dtypes exactly) with a small
in-memory cache in front.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from app.config import config


@dataclass
class DatasetMeta:
    dataset_id: str
    filename: str
    size_mb: float
    uploaded_at: float
    rows: int
    cols: int
    sheet_names: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class DatasetStore:
    """Thread-safe store for uploaded datasets and analysis caches."""

    _MEM_LIMIT = 8  # datasets kept in RAM; older ones reload from disk

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(config.data_dir, "datasets")
        os.makedirs(self.base_dir, exist_ok=True)
        self._lock = threading.RLock()
        self._mem: Dict[str, Dict[str, Any]] = {}      # id -> {raw, active, meta}
        self._caches: Dict[str, Dict[str, Any]] = {}   # id -> {key -> (hash, obj)}

    # ── paths ────────────────────────────────────────────
    def _dir(self, ds_id: str) -> str:
        return os.path.join(self.base_dir, ds_id)

    def _path(self, ds_id: str, name: str) -> str:
        return os.path.join(self._dir(ds_id), name)

    # ── lifecycle ────────────────────────────────────────
    def create(self, df_raw: pd.DataFrame, filename: str, size_mb: float,
               sheet_names: Optional[list] = None,
               warnings: Optional[list] = None) -> DatasetMeta:
        ds_id = uuid.uuid4().hex[:12]
        meta = DatasetMeta(
            dataset_id=ds_id,
            filename=filename,
            size_mb=round(size_mb, 2),
            uploaded_at=time.time(),
            rows=len(df_raw),
            cols=df_raw.shape[1],
            sheet_names=sheet_names or [],
            warnings=warnings or [],
        )
        with self._lock:
            os.makedirs(self._dir(ds_id), exist_ok=True)
            self._save_df(ds_id, "raw.pkl", df_raw)
            self._save_df(ds_id, "active.pkl", df_raw)
            with open(self._path(ds_id, "meta.pkl"), "wb") as f:
                pickle.dump(meta, f)
            self._touch_mem(ds_id, raw=df_raw, active=df_raw.copy(), meta=meta)
        return meta

    def list_meta(self) -> list[DatasetMeta]:
        out = []
        for ds_id in sorted(os.listdir(self.base_dir)):
            meta = self.get_meta(ds_id)
            if meta:
                out.append(meta)
        out.sort(key=lambda m: m.uploaded_at, reverse=True)
        return out

    def get_meta(self, ds_id: str) -> Optional[DatasetMeta]:
        with self._lock:
            if ds_id in self._mem:
                return self._mem[ds_id]["meta"]
        p = self._path(ds_id, "meta.pkl")
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    def delete(self, ds_id: str) -> bool:
        import shutil
        with self._lock:
            self._mem.pop(ds_id, None)
            self._caches.pop(ds_id, None)
            d = self._dir(ds_id)
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
                return True
        return False

    # ── dataframes ───────────────────────────────────────
    def get_df(self, ds_id: str) -> Optional[pd.DataFrame]:
        """Active (possibly cleaned) dataframe."""
        return self._load("active", ds_id)

    def get_raw_df(self, ds_id: str) -> Optional[pd.DataFrame]:
        return self._load("raw", ds_id)

    def update_active(self, ds_id: str, df: pd.DataFrame) -> None:
        with self._lock:
            self._save_df(ds_id, "active.pkl", df)
            if ds_id in self._mem:
                self._mem[ds_id]["active"] = df
            meta = self.get_meta(ds_id)
            if meta:
                meta.rows, meta.cols = len(df), df.shape[1]
                with open(self._path(ds_id, "meta.pkl"), "wb") as f:
                    pickle.dump(meta, f)
                if ds_id in self._mem:
                    self._mem[ds_id]["meta"] = meta

    def reset_active(self, ds_id: str) -> Optional[pd.DataFrame]:
        """Restore active df back to the raw upload."""
        raw = self.get_raw_df(ds_id)
        if raw is None:
            return None
        self.update_active(ds_id, raw.copy())
        return raw

    # ── analysis caches (hash-invalidated) ───────────────
    def cache_get(self, ds_id: str, key: str) -> Optional[Any]:
        df = self.get_df(ds_id)
        if df is None:
            return None
        h = self._hash_df(df)
        with self._lock:
            entry = self._caches.get(ds_id, {}).get(key)
            if entry and entry[0] == h:
                return entry[1]
        # disk fallback
        p = self._path(ds_id, f"cache_{key}.pkl")
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    stored_hash, obj = pickle.load(f)
                if stored_hash == h:
                    with self._lock:
                        self._caches.setdefault(ds_id, {})[key] = (h, obj)
                    return obj
            except Exception:
                pass
        return None

    def cache_set(self, ds_id: str, key: str, obj: Any) -> None:
        df = self.get_df(ds_id)
        if df is None:
            return
        h = self._hash_df(df)
        with self._lock:
            self._caches.setdefault(ds_id, {})[key] = (h, obj)
        try:
            with open(self._path(ds_id, f"cache_{key}.pkl"), "wb") as f:
                pickle.dump((h, obj), f)
        except Exception:
            pass  # cache persistence is best-effort

    # ── internals ────────────────────────────────────────
    def _save_df(self, ds_id: str, name: str, df: pd.DataFrame) -> None:
        with open(self._path(ds_id, name), "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load(self, which: str, ds_id: str) -> Optional[pd.DataFrame]:
        with self._lock:
            if ds_id in self._mem:
                return self._mem[ds_id][which]
        p = self._path(ds_id, f"{which}.pkl")
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            df = pickle.load(f)
        other = "raw" if which == "active" else "active"
        other_df = None
        op = self._path(ds_id, f"{other}.pkl")
        if os.path.exists(op):
            with open(op, "rb") as f:
                other_df = pickle.load(f)
        meta = self.get_meta(ds_id)
        with self._lock:
            self._touch_mem(ds_id, **{which: df, other: other_df, "meta": meta})
        return df

    def _touch_mem(self, ds_id: str, **entry) -> None:
        self._mem[ds_id] = entry
        while len(self._mem) > self._MEM_LIMIT:
            oldest = next(iter(self._mem))
            if oldest == ds_id:
                break
            self._mem.pop(oldest)

    @staticmethod
    def _hash_df(df: pd.DataFrame) -> str:
        sig = f"{df.shape}|{list(df.columns)}|{list(df.dtypes.astype(str))}"
        sample = df.head(100).to_json(default_handler=str)
        return hashlib.md5((sig + sample).encode()).hexdigest()


store = DatasetStore()
