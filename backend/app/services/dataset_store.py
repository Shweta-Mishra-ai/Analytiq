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
    # What each column is, as the user confirmed it ({} until they do).
    column_roles: dict = field(default_factory=dict)


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
            self._bump_revision(owner, ds_id)
        return meta

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
        """Active (possibly cleaned) dataframe."""
        return self._load("active", owner, ds_id)

    def get_raw_df(self, owner: str, ds_id: str) -> Optional[pd.DataFrame]:
        return self._load("raw", owner, ds_id)

    def update_active(self, owner: str, ds_id: str, df: pd.DataFrame) -> None:
        mkey = self._mkey(owner, ds_id)
        with self._lock:
            self._save_df(owner, ds_id, "active.pkl", df)
            if mkey in self._mem:
                self._mem[mkey]["active"] = self._with_roles(
                    df, self.get_meta(owner, ds_id))
            self._bump_revision(owner, ds_id)
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

    # ── column roles ─────────────────────────────────────
    @staticmethod
    def _with_roles(df: Optional[pd.DataFrame], meta) -> Optional[pd.DataFrame]:
        """The frame as analysed: confirmed roles applied on every load, so
        cleaning, undo and reset keep them."""
        roles = getattr(meta, "column_roles", None) or {}
        if df is None or not roles:
            return df
        from app.services.column_roles import apply
        return apply(df, roles, strict=False)[0]

    def set_column_roles(self, owner: str, ds_id: str, roles: dict) -> list:
        """Confirm what each column is. Raises column_roles.RoleError when the
        data cannot support a role; returns notes for the user otherwise."""
        from app.services.column_roles import apply
        with self._lock:
            meta = self.get_meta(owner, ds_id)
            if meta is None:
                raise KeyError(ds_id)
            # Start from the stored data, not the frame with the previous
            # roles applied: a column changed back from date to text must
            # not stay converted.
            with open(self._path(owner, ds_id, "active.pkl"), "rb") as f:
                stored = pickle.load(f)
            applied, notes = apply(stored, roles, strict=True)
            meta.column_roles = dict(applied.attrs.get("column_roles", {}))
            with open(self._path(owner, ds_id, "meta.pkl"), "wb") as f:
                pickle.dump(meta, f)
            mkey = self._mkey(owner, ds_id)
            if mkey in self._mem:
                self._mem[mkey]["meta"] = meta
                self._mem[mkey]["active"] = applied
            # Every analysis depends on the roles, so none of the cached
            # results still holds.
            self._bump_revision(owner, ds_id)
        return notes

    # ── analysis caches (revision-invalidated) ───────────
    #
    # An entry is valid for exactly one revision of the active frame; every
    # write to it (create, clean, reset) issues a new revision. This used to
    # compare a hash of the shape, dtypes and first 100 rows, so any edit
    # past row 100 that kept the shape — imputing a gap, capping an outlier —
    # served the analysis of the previous data as if it were current.
    # Hashing every row on every request would be correct but costs seconds
    # on a large file; a revision token is exact and free.
    #
    # Keys routinely contain user column names ("ml_Revenue/Unit"), so the
    # filename is a digest of the key, never the key itself: a "/" made the
    # write fail silently, and distinct parameter sets could collide.

    @staticmethod
    def _cache_file(key: str) -> str:
        return "cache_{}.pkl".format(hashlib.sha256(key.encode("utf-8")).hexdigest()[:32])

    def _bump_revision(self, owner: str, ds_id: str) -> str:
        rev = uuid.uuid4().hex
        with open(self._path(owner, ds_id, "revision"), "w") as f:
            f.write(rev)
        mkey = self._mkey(owner, ds_id)
        with self._lock:
            self._caches.pop(mkey, None)
            if mkey in self._mem:
                self._mem[mkey]["revision"] = rev
        return rev

    def _revision(self, owner: str, ds_id: str) -> Optional[str]:
        mkey = self._mkey(owner, ds_id)
        with self._lock:
            rev = self._mem.get(mkey, {}).get("revision")
        if rev:
            return rev
        if self.get_meta(owner, ds_id) is None:
            return None
        p = self._path(owner, ds_id, "revision")
        try:
            with open(p) as f:
                rev = f.read().strip()
        except FileNotFoundError:
            rev = ""
        if not rev:
            # A dataset stored before revisions existed: start one, which
            # invalidates its old hash-keyed caches once.
            rev = self._bump_revision(owner, ds_id)
        with self._lock:
            if mkey in self._mem:
                self._mem[mkey]["revision"] = rev
        return rev

    def cache_get(self, owner: str, ds_id: str, key: str) -> Optional[Any]:
        rev = self._revision(owner, ds_id)
        if rev is None:
            return None
        mkey = self._mkey(owner, ds_id)
        with self._lock:
            entry = self._caches.get(mkey, {}).get(key)
            if entry and entry[0] == rev:
                return entry[1]
        p = self._path(owner, ds_id, self._cache_file(key))
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    stored = pickle.load(f)
                if len(stored) == 3 and stored[0] == rev and stored[1] == key:
                    with self._lock:
                        self._caches.setdefault(mkey, {})[key] = (rev, stored[2])
                    return stored[2]
            except Exception:
                logger.debug("cache_get: suppressed exception", exc_info=True)
        return None

    def cache_set(self, owner: str, ds_id: str, key: str, obj: Any) -> None:
        rev = self._revision(owner, ds_id)
        if rev is None:
            return
        mkey = self._mkey(owner, ds_id)
        with self._lock:
            self._caches.setdefault(mkey, {})[key] = (rev, obj)
        try:
            with open(self._path(owner, ds_id, self._cache_file(key)), "wb") as f:
                pickle.dump((rev, key, obj), f)
        except Exception:
            # Best-effort, but no longer silent: a failure here means every
            # request recomputes, which is worth knowing about.
            logger.warning("cache_set: could not persist %r for %s/%s",
                           key, owner, ds_id, exc_info=True)

    # ── internals ────────────────────────────────────────
    def _save_df(self, owner: str, ds_id: str, name: str, df: pd.DataFrame) -> None:
        # Roles live in the metadata and are applied on load; the stored
        # frame stays free of them so a stale copy can never win.
        if "column_roles" in df.attrs:
            df = df.copy(deep=False)
            df.attrs = {k: v for k, v in df.attrs.items() if k != "column_roles"}
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
        active = df if which == "active" else other_df
        raw = other_df if which == "active" else df
        active = self._with_roles(active, meta)
        with self._lock:
            self._touch_mem(owner, ds_id, active=active, raw=raw, meta=meta)
        return active if which == "active" else raw

    def _touch_mem(self, owner: str, ds_id: str, **entry) -> None:
        mkey = self._mkey(owner, ds_id)
        self._mem[mkey] = entry
        while len(self._mem) > self._MEM_LIMIT:
            oldest = next(iter(self._mem))
            if oldest == mkey:
                break
            self._mem.pop(oldest)


store = DatasetStore()
