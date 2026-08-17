"""
services/dataset_store.py — server-side replacement for the old
Streamlit session_manager.

Each uploaded dataset gets a UUID and is scoped to the owner (client
username) who uploaded it. We keep:
  - raw df      (as uploaded, never mutated)
  - active df   (after cleaning steps)
  - per-dataset analysis caches keyed by a content hash of the active df,
    so caches invalidate automatically when the data changes.

DataFrames are pickled to disk (preserves dtypes exactly) with a small
in-memory cache in front. Storage layout is base_dir/{owner}/{ds_id}/ —
physical separation per owner, not just a filtered query, so a bug in
one code path can't accidentally cross-serve another client's files.
Every method requires an explicit `owner` argument (no default) so a
new call site can't forget to scope it.
"""
from __future__ import annotations
import logging

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

logger = logging.getLogger(__name__)


@dataclass
class DatasetMeta:
    dataset_id: str
    filename: str
    size_mb: float
    uploaded_at: float
    rows: int
    cols: int
    owner: str = ""
    sheet_names: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class DatasetStore:
    """Thread-safe store for uploaded datasets and analysis caches."""

    _MEM_LIMIT = 8  # datasets kept in RAM; older ones reload from disk

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(config.data_dir, "datasets")
        os.makedirs(self.base_dir, exist_ok=True)
        self._lock = threading.RLock()
        self._mem: Dict[str, Dict[str, Any]] = {}      # "owner/id" -> {raw, active, meta}
        self._caches: Dict[str, Dict[str, Any]] = {}   # "owner/id" -> {key -> (hash, obj)}

    # ── paths ────────────────────────────────────────────
    @staticmethod
    def _safe(part: str) -> str:
        # owner/ds_id are always our own generated slugs or validated
        # usernames, but never trust path components blindly.
        if not part or "/" in part or "\\" in part or part in (".", ".."):
            raise ValueError(f"Invalid path segment: {part!r}")
        return part

    def _dir(self, owner: str, ds_id: str) -> str:
        return os.path.join(self.base_dir, self._safe(owner), self._safe(ds_id))

    def _path(self, owner: str, ds_id: str, name: str) -> str:
        return os.path.join(self._dir(owner, ds_id), name)

    def _mkey(self, owner: str, ds_id: str) -> str:
        return f"{owner}/{ds_id}"

    # ── lifecycle ────────────────────────────────────────
    def create(self, owner: str, df_raw: pd.DataFrame, filename: str, size_mb: float,
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
            owner=owner,
            sheet_names=sheet_names or [],
            warnings=warnings or [],
        )
        with self._lock:
            os.makedirs(self._dir(owner, ds_id), exist_ok=True)
            self._save_df(owner, ds_id, "raw.pkl", df_raw)
            self._save_df(owner, ds_id, "active.pkl", df_raw)
            with open(self._path(owner, ds_id, "meta.pkl"), "wb") as f:
                pickle.dump(meta, f)
            self._touch_mem(owner, ds_id, raw=df_raw, active=df_raw.copy(), meta=meta)
        return meta

    def storage_mb(self, owner: str) -> float:
        """How much disk this owner's datasets and caches occupy.

        Measured from the files rather than from the recorded upload
        sizes: the pickles, the cleaned copy and the cached reports are
        several times the size of the CSV that arrived, and it is the
        disk that fills up, not the CSV.
        """
        owner_dir = os.path.join(self.base_dir, self._safe(owner))
        total = 0
        for root, _dirs, files in os.walk(owner_dir):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    logger.debug("could not size %s", name, exc_info=True)
        return total / (1024 * 1024)

    def list_meta(self, owner: str) -> list[DatasetMeta]:
        owner_dir = os.path.join(self.base_dir, self._safe(owner))
        if not os.path.isdir(owner_dir):
            return []
        out = []
        for ds_id in sorted(os.listdir(owner_dir)):
            meta = self.get_meta(owner, ds_id)
            if meta:
                out.append(meta)
        out.sort(key=lambda m: m.uploaded_at, reverse=True)
        return out

    def list_all_meta(self) -> list[DatasetMeta]:
        """Every dataset across every owner. Internal use only (cleanup
        sweep) — never expose this to a client-facing route."""
        out = []
        if not os.path.isdir(self.base_dir):
            return out
        for owner in sorted(os.listdir(self.base_dir)):
            owner_dir = os.path.join(self.base_dir, owner)
            if not os.path.isdir(owner_dir):
                continue
            out.extend(self.list_meta(owner))
        return out

    def get_meta(self, owner: str, ds_id: str) -> Optional[DatasetMeta]:
        mkey = self._mkey(owner, ds_id)
        with self._lock:
            if mkey in self._mem:
                return self._mem[mkey]["meta"]
        p = self._path(owner, ds_id, "meta.pkl")
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            meta = pickle.load(f)
        if meta.owner and meta.owner != owner:
            return None  # defense in depth; should be unreachable via _dir()
        return meta

    def delete(self, owner: str, ds_id: str) -> bool:
        import shutil
        mkey = self._mkey(owner, ds_id)
        with self._lock:
            self._mem.pop(mkey, None)
            self._caches.pop(mkey, None)
            d = self._dir(owner, ds_id)
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
                return True
        return False

    # ── dataframes ───────────────────────────────────────
    def get_df(self, owner: str, ds_id: str) -> Optional[pd.DataFrame]:
        """Active (possibly cleaned) dataframe.

        Column names are made unique on the way out. The loader does this
        on upload, but a dataset pickled before that fix, or one written
        by a cleaning step that pivoted, would otherwise reach the
        engines with duplicates and take six of them down.
        """
        from app.services.dtypes import dedupe_columns

        return dedupe_columns(self._load("active", owner, ds_id))

    def get_raw_df(self, owner: str, ds_id: str) -> Optional[pd.DataFrame]:
        from app.services.dtypes import dedupe_columns

        return dedupe_columns(self._load("raw", owner, ds_id))

    def update_active(self, owner: str, ds_id: str, df: pd.DataFrame) -> None:
        mkey = self._mkey(owner, ds_id)
        with self._lock:
            self._save_df(owner, ds_id, "active.pkl", df)
            if mkey in self._mem:
                self._mem[mkey]["active"] = df
            meta = self.get_meta(owner, ds_id)
            if meta:
                meta.rows, meta.cols = len(df), df.shape[1]
                with open(self._path(owner, ds_id, "meta.pkl"), "wb") as f:
                    pickle.dump(meta, f)
                if mkey in self._mem:
                    self._mem[mkey]["meta"] = meta

    def reset_active(self, owner: str, ds_id: str) -> Optional[pd.DataFrame]:
        """Restore active df back to the raw upload."""
        raw = self.get_raw_df(owner, ds_id)
        if raw is None:
            return None
        self.update_active(owner, ds_id, raw.copy())
        return raw

    # ── analysis caches (hash-invalidated) ───────────────
    def cache_get(self, owner: str, ds_id: str, key: str) -> Optional[Any]:
        df = self.get_df(owner, ds_id)
        if df is None:
            return None
        h = self._hash_df(df)
        mkey = self._mkey(owner, ds_id)
        with self._lock:
            entry = self._caches.get(mkey, {}).get(key)
            if entry and entry[0] == h:
                return entry[1]
        # disk fallback
        p = self._path(owner, ds_id, f"cache_{key}.pkl")
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    stored_hash, obj = pickle.load(f)
                if stored_hash == h:
                    with self._lock:
                        self._caches.setdefault(mkey, {})[key] = (h, obj)
                    return obj
            except Exception:
                logger.debug("cache_get: suppressed exception", exc_info=True)
        return None

    def cache_set(self, owner: str, ds_id: str, key: str, obj: Any) -> None:
        df = self.get_df(owner, ds_id)
        if df is None:
            return
        h = self._hash_df(df)
        mkey = self._mkey(owner, ds_id)
        with self._lock:
            self._caches.setdefault(mkey, {})[key] = (h, obj)
        try:
            with open(self._path(owner, ds_id, f"cache_{key}.pkl"), "wb") as f:
                pickle.dump((h, obj), f)
        except Exception:
            # Persistence is best-effort — the in-memory copy above is
            # already set, so the request succeeds either way. It is not
            # best-effort enough to happen in silence: the usual cause is
            # a full disk, and a full disk announces itself as "the app
            # got slower" and nothing else until something that cannot
            # degrade fails too.
            logger.warning("could not persist the %s cache for %s/%s — "
                           "analysis will be recomputed next time; check "
                           "free disk space", key, owner, ds_id,
                           exc_info=True)

    # ── internals ────────────────────────────────────────
    def _save_df(self, owner: str, ds_id: str, name: str, df: pd.DataFrame) -> None:
        with open(self._path(owner, ds_id, name), "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load(self, which: str, owner: str, ds_id: str) -> Optional[pd.DataFrame]:
        mkey = self._mkey(owner, ds_id)
        with self._lock:
            if mkey in self._mem:
                return self._mem[mkey][which]
        p = self._path(owner, ds_id, f"{which}.pkl")
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            df = pickle.load(f)
        other = "raw" if which == "active" else "active"
        other_df = None
        op = self._path(owner, ds_id, f"{other}.pkl")
        if os.path.exists(op):
            with open(op, "rb") as f:
                other_df = pickle.load(f)
        meta = self.get_meta(owner, ds_id)
        with self._lock:
            self._touch_mem(owner, ds_id, **{which: df, other: other_df, "meta": meta})
        return df

    def _touch_mem(self, owner: str, ds_id: str, **entry) -> None:
        """Keep the most recently used datasets in memory.

        Re-assigning an existing dict key does not move it — Python keeps
        insertion order — so the dataset someone was actively working on
        stayed wherever it first landed and was the next one evicted,
        while a dataset nobody had touched in an hour survived. It only
        cost a reload from disk, but it meant the eviction policy was the
        opposite of the one the name claims.
        """
        mkey = self._mkey(owner, ds_id)
        self._mem.pop(mkey, None)
        self._mem[mkey] = entry
        while len(self._mem) > self._MEM_LIMIT:
            oldest = next(iter(self._mem))
            if oldest == mkey:
                break
            self._mem.pop(oldest)
            # The analysis caches follow the frame out of memory. They
            # were left behind: `_mem` was capped at eight datasets and
            # `_caches` was not capped at all, so every dataset the
            # process ever touched kept its profile, EDA, BI, story and
            # ML reports alive until restart — measured at 3.3 MB per
            # dataset for ML and EDA alone. Nothing is lost by dropping
            # them; `cache_get` reloads from the pickle beside the
            # dataset, and the hash check still decides whether it is
            # still valid.
            self._caches.pop(oldest, None)

    @staticmethod
    def _hash_df(df: pd.DataFrame) -> str:
        """A fingerprint of the whole frame, not of its first page.

        This used to hash the shape, the column names, the dtypes and
        `df.head(100)`. Every cleaning operation that changes values
        without changing the frame's shape — capping outliers, filling
        medians, stripping whitespace, correcting a typo — produced an
        identical fingerprint whenever the affected rows sat past the
        hundredth. On a 5,000-row file, capping outliers changed 2,990
        values and the hash did not move, so `cache_get` handed back the
        story, the ML report and the charts computed *before* the clean.
        The user cleaned their data and the report kept showing the old
        numbers, with nothing anywhere to say so.

        `hash_pandas_object` reads every value. It is O(n) rather than
        O(1), which is the correct trade: a cache that is fast and wrong
        is worse than no cache.
        """
        sig = "{}|{}|{}".format(df.shape, list(df.columns),
                                list(df.dtypes.astype(str)))
        try:
            values = pd.util.hash_pandas_object(df, index=True).values
            digest = hashlib.md5(values.tobytes())
        except Exception:
            # Object columns holding unhashable values (a list in a cell)
            # fall back to a full serialisation rather than to a sample —
            # slower, but it still sees every row.
            logger.debug("hash_pandas_object failed; serialising instead",
                         exc_info=True)
            digest = hashlib.md5(df.to_json(default_handler=str).encode())
        digest.update(sig.encode())
        return digest.hexdigest()


store = DatasetStore()
