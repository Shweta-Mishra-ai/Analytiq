"""
services/dataset_store.py — server-side replacement for the old
Streamlit session_manager.

Each uploaded dataset gets a UUID and is scoped to the owner (client
username) who uploaded it. We keep:
  - raw df      (as uploaded, never mutated)
  - active df   (after cleaning steps)
  - per-dataset analysis caches keyed by a content hash of the active df,
    so caches invalidate automatically when the data changes.

Storage layout is base_dir/{owner}/{ds_id}/ — physical separation per
owner, not just a filtered query, so a bug in one code path can't
accidentally cross-serve another client's files. Every method requires an
explicit `owner` argument (no default) so a new call site can't forget to
scope it.

Frames are parquet and metadata is JSON (see frame_io), because
unpickling runs whatever the file says to run and the storage directory
should not be a way to execute code. Analysis caches are the one
remaining pickle — they hold fitted sklearn models and engine dataclasses
that have no data-only form — so they are HMAC-signed with a key that
never leaves the server, and an entry that does not verify is discarded
unread rather than unpickled.
"""
from __future__ import annotations
import logging

import hashlib
import hmac
import json
import os
import pickle
import secrets
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from app.config import config
from app.services import integrity
from app.services.frame_io import read_frame, write_frame

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


def _cache_key(base_dir: str) -> bytes:
    """The key that signs cache entries. Prefers the configured app
    secret so a multi-process deployment agrees on one; otherwise a
    per-install key generated once and kept owner-readable."""
    if config.app_secret:
        return hashlib.sha256(
            ("cache:" + config.app_secret).encode()).digest()
    path = os.path.join(base_dir, ".cache_key")
    try:
        if os.path.exists(path):
            with open(path, "rb") as fh:
                stored = fh.read().strip()
            if len(stored) >= 32:
                return stored
        generated = secrets.token_bytes(32)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(generated)
        return generated
    except Exception:
        # Read-only storage: sign with a process-lifetime key. Caches
        # then simply miss after a restart, which is safe.
        logger.warning("could not persist a cache signing key in %s; "
                       "caches will not survive a restart", base_dir)
        return secrets.token_bytes(32)


class DatasetStore:
    """Thread-safe store for uploaded datasets and analysis caches."""

    _MEM_LIMIT = 8  # datasets kept in RAM; older ones reload from disk

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(config.data_dir, "datasets")
        os.makedirs(self.base_dir, exist_ok=True)
        self._lock = threading.RLock()
        self._mem: Dict[str, Dict[str, Any]] = {}      # "owner/id" -> {raw, active, meta}
        self._caches: Dict[str, Dict[str, Any]] = {}   # "owner/id" -> {key -> (hash, obj)}
        self._sign_key = _cache_key(self.base_dir)

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
    def dataset_dir(self, owner: str, ds_id: str) -> str:
        """Where a dataset's files live. Public because the integrity
        layer writes its record and audit trail alongside them."""
        return self._dir(owner, ds_id)

    def create(self, owner: str, df_raw: pd.DataFrame, filename: str, size_mb: float,
               sheet_names: Optional[list] = None,
               warnings: Optional[list] = None,
               source_bytes: int = 0,
               source_sha256: str = "") -> DatasetMeta:
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
            warnings=list(warnings or []),
        )
        with self._lock:
            os.makedirs(self._dir(owner, ds_id), exist_ok=True)
            coerced = self._save_df(owner, ds_id, "raw", df_raw)
            self._save_df(owner, ds_id, "active", df_raw)
            if coerced:
                # The user should hear this from the upload response, not
                # discover it when a column stops summing.
                meta.warnings.append(
                    f"{len(coerced)} column(s) mix numbers and text and are "
                    f"stored as text: {', '.join(coerced[:5])}"
                    + ("…" if len(coerced) > 5 else ""))
            self._write_meta(owner, ds_id, meta)
            self._touch_mem(owner, ds_id, raw=df_raw, active=df_raw.copy(), meta=meta)
            # Taken here, at the moment of receipt, and never recomputed:
            # a digest written later would only prove the data matches
            # itself.
            integrity.record_ingest(
                self._dir(owner, ds_id), ds_id, df_raw, filename,
                source_bytes=source_bytes, source_sha256=source_sha256,
                actor=owner)
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
        p = self._path(owner, ds_id, "meta.json")
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            logger.warning("unreadable metadata for %s/%s", owner, ds_id,
                           exc_info=True)
            return None
        fields = {f.name for f in DatasetMeta.__dataclass_fields__.values()}
        meta = DatasetMeta(**{k: v for k, v in raw.items() if k in fields})
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

    def update_active(self, owner: str, ds_id: str, df: pd.DataFrame,
                      event: str = "transform",
                      detail: Optional[dict] = None) -> None:
        """Every change to the working copy goes through here, which is
        what lets the integrity check distinguish a recorded change from
        an unaccounted one. Callers name the event (`clean`, `reset`) so
        the audit trail says what happened, not just that something did.
        """
        mkey = self._mkey(owner, ds_id)
        with self._lock:
            self._save_df(owner, ds_id, "active", df)
            if mkey in self._mem:
                self._mem[mkey]["active"] = df
            meta = self.get_meta(owner, ds_id)
            if meta:
                meta.rows, meta.cols = len(df), df.shape[1]
                self._write_meta(owner, ds_id, meta)
                if mkey in self._mem:
                    self._mem[mkey]["meta"] = meta
            integrity.record_change(self._dir(owner, ds_id), df,
                                    event=event, actor=owner, detail=detail)

    def reset_active(self, owner: str, ds_id: str) -> Optional[pd.DataFrame]:
        """Restore active df back to the raw upload."""
        raw = self.get_raw_df(owner, ds_id)
        if raw is None:
            return None
        self.update_active(owner, ds_id, raw.copy(), event="reset",
                           detail={"restored_to": "the original upload"})
        return raw

    def record_event(self, owner: str, ds_id: str, event: str,
                     detail: Optional[dict] = None) -> None:
        """Note in the audit trail that something used this dataset —
        a report, an export — without changing it."""
        integrity.record_event(self._dir(owner, ds_id), event,
                               self.get_df(owner, ds_id), actor=owner,
                               detail=detail)

    def integrity(self, owner: str, ds_id: str) -> Optional[dict]:
        """Recomputed on every call, deliberately: an integrity verdict
        that is cached is a verdict about the past."""
        if self.get_meta(owner, ds_id) is None:
            return None
        return integrity.summary(self._dir(owner, ds_id),
                                 self.get_raw_df(owner, ds_id),
                                 self.get_df(owner, ds_id))

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
        p = self._path(owner, ds_id, f"cache_{self._safe_key(key)}.bin")
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    blob = f.read()
                payload = self._unseal(blob)
                if payload is None:
                    return None
                stored_hash, obj = pickle.loads(payload)
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
            blob = self._seal(pickle.dumps((h, obj),
                                           protocol=pickle.HIGHEST_PROTOCOL))
            with open(self._path(owner, ds_id,
                                 f"cache_{self._safe_key(key)}.bin"), "wb") as f:
                f.write(blob)
        except Exception:
            logger.debug("cache_set: persistence failed", exc_info=True)

    # ── cache signing ────────────────────────────────────
    def _seal(self, payload: bytes) -> bytes:
        return hmac.new(self._sign_key, payload, hashlib.sha256).digest() + payload

    def _unseal(self, blob: bytes) -> Optional[bytes]:
        """Return the payload only if this server wrote it. A file
        someone else put here is dropped, never unpickled."""
        if len(blob) <= 32:
            return None
        tag, payload = blob[:32], blob[32:]
        expected = hmac.new(self._sign_key, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected):
            logger.warning("discarding a cache entry that this server did "
                           "not sign")
            return None
        return payload

    # ── internals ────────────────────────────────────────
    @staticmethod
    def _safe_key(key: str) -> str:
        """Cache keys carry user-chosen text (``ml_{target}``), so they
        are never pasted into a path as-is."""
        cleaned = "".join(c if (c.isalnum() or c in "-_") else "_"
                          for c in str(key))[:60]
        return f"{cleaned}_{hashlib.sha256(str(key).encode()).hexdigest()[:8]}"

    def _write_meta(self, owner: str, ds_id: str, meta: DatasetMeta) -> None:
        p = self._path(owner, ds_id, "meta.json")
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f)
        os.replace(tmp, p)

    def _save_df(self, owner: str, ds_id: str, which: str,
                 df: pd.DataFrame) -> list[str]:
        return write_frame(self._path(owner, ds_id, f"{which}.parquet"), df)

    def _load(self, which: str, owner: str, ds_id: str) -> Optional[pd.DataFrame]:
        mkey = self._mkey(owner, ds_id)
        with self._lock:
            if mkey in self._mem:
                return self._mem[mkey][which]
        df = read_frame(self._path(owner, ds_id, f"{which}.parquet"))
        if df is None:
            return None
        other = "raw" if which == "active" else "active"
        other_df = read_frame(self._path(owner, ds_id, f"{other}.parquet"))
        meta = self.get_meta(owner, ds_id)
        with self._lock:
            self._touch_mem(owner, ds_id, **{which: df, other: other_df, "meta": meta})
        return df

    def _touch_mem(self, owner: str, ds_id: str, **entry) -> None:
        mkey = self._mkey(owner, ds_id)
        self._mem[mkey] = entry
        while len(self._mem) > self._MEM_LIMIT:
            oldest = next(iter(self._mem))
            if oldest == mkey:
                break
            self._mem.pop(oldest)

    @staticmethod
    def _hash_df(df: pd.DataFrame) -> str:
        sig = f"{df.shape}|{list(df.columns)}|{list(df.dtypes.astype(str))}"
        sample = df.head(100).to_json(default_handler=str)
        return hashlib.md5((sig + sample).encode()).hexdigest()


store = DatasetStore()
