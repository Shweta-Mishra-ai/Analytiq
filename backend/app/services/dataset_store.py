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

    def __init__(self, base_dir: Optional[str] = None, blobs=None):
        self.base_dir = base_dir or os.path.join(config.data_dir, "datasets")
        os.makedirs(self.base_dir, exist_ok=True)
        # Local disk stays the working store — parquet wants a path, and
        # so does the integrity layer — but it is a cache in front of the
        # blob store rather than the only copy.
        #
        # It had been the only copy, and the container's disk is not a
        # place to keep a client's data: Render happens to mount a volume,
        # railway.json declares none at all, and on either platform a
        # second instance has a second disk. A dataset was present on one
        # request and missing on the next.
        if blobs is not None:
            self.blobs = blobs
        elif base_dir is not None:
            self.blobs = None          # an explicit directory means a test
        else:
            from app.services.blobstore import build_store
            store = build_store("datasets")
            self.blobs = store if getattr(store, "durable", False) else None
        self._lock = threading.RLock()
        self._mem: Dict[str, Dict[str, Any]] = {}      # "owner/id" -> {raw, active, meta}
        self._caches: Dict[str, Dict[str, Any]] = {}   # "owner/id" -> {key -> (hash, obj)}
        # "owner/id" -> (the exact frame object, its content hash). Hashing
        # 40k rows costs ~30ms, which is fine once per change and wasteful
        # on every cache lookup, so the result is held against the frame
        # it describes and recomputed the moment a different object arrives.
        self._hashes: Dict[str, Any] = {}
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

    # ── durable copies ───────────────────────────────────
    # The frames, what they mean, and the record of what was done to
    # them. Not cache_*.bin: those are a fitted model or an engine
    # result, they regenerate from the frame in seconds, and shipping
    # them to object storage on every analysis would cost more than
    # recomputing them ever does.
    DURABLE = ("meta.json", "raw.parquet", "active.parquet",
               "raw.parquet.schema.json", "active.parquet.schema.json",
               "integrity.json", "audit.jsonl")

    def _blob_key(self, owner: str, ds_id: str, name: str) -> str:
        return "{}/{}/{}".format(self._safe(owner), self._safe(ds_id), name)

    def _back_up(self, owner: str, ds_id: str, *names: str) -> None:
        """Copy what was just written to the durable store."""
        if self.blobs is None:
            return
        for name in names:
            path = self._path(owner, ds_id, name)
            if not os.path.exists(path):
                continue
            try:
                with open(path, "rb") as fh:
                    self.blobs.put(self._blob_key(owner, ds_id, name),
                                   fh.read())
            except Exception:
                # A dataset the client can still use beats no dataset at
                # all, so a failure to back up is logged, not raised.
                logger.warning("could not back up %s for %s/%s", name,
                               owner, ds_id, exc_info=True)

    def _restore(self, owner: str, ds_id: str, name: str) -> bool:
        """Bring one file back from the durable store, if it is there.

        This is what makes a cold container — a redeploy, a restart, or
        simply a different instance of the same service — able to answer
        for a dataset it has never seen.
        """
        if self.blobs is None:
            return False
        path = self._path(owner, ds_id, name)
        if os.path.exists(path):
            return True
        try:
            data = self.blobs.get(self._blob_key(owner, ds_id, name))
        except Exception:
            logger.warning("could not read %s for %s/%s from durable store",
                           name, owner, ds_id, exc_info=True)
            return False
        if data is None:
            return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        from app.services.blobstore import atomic_write
        atomic_write(path, data)
        logger.info("restored %s for %s/%s from durable storage",
                    name, owner, ds_id)
        return True

    def _restore_all(self, owner: str, ds_id: str) -> None:
        for name in self.DURABLE:
            self._restore(owner, ds_id, name)

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
        # Both sides. Listing only the local directory is what made a
        # restarted container answer "you have no datasets" to an
        # account whose uploads were sitting safely in object storage.
        ids = set()
        owner_dir = os.path.join(self.base_dir, self._safe(owner))
        if os.path.isdir(owner_dir):
            ids.update(os.listdir(owner_dir))
        ids.update(self._durable_ids(owner))

        out = []
        for ds_id in sorted(ids):
            try:
                meta = self.get_meta(owner, ds_id)
            except ValueError:
                continue          # not one of ours; _safe rejected it
            if meta:
                out.append(meta)
        out.sort(key=lambda m: m.uploaded_at, reverse=True)
        return out

    def _durable_ids(self, owner: str) -> set:
        """Dataset ids this owner has in the durable store."""
        if self.blobs is None:
            return set()
        try:
            prefix = self._safe(owner)
            keys = self.blobs.list_prefix(prefix)
        except Exception:
            logger.warning("could not list durable datasets for %s", owner,
                           exc_info=True)
            return set()
        found = set()
        for key in keys:
            parts = str(key).split("/")
            if len(parts) >= 3 and parts[0] == prefix:
                found.add(parts[1])
        return found

    def _durable_owners(self) -> set:
        if self.blobs is None:
            return set()
        try:
            keys = self.blobs.list_prefix("")
        except Exception:
            logger.warning("could not list durable dataset owners",
                           exc_info=True)
            return set()
        return {str(k).split("/")[0] for k in keys if "/" in str(k)}

    def list_all_meta(self) -> list[DatasetMeta]:
        """Every dataset across every owner. Internal use only (cleanup
        sweep) — never expose this to a client-facing route."""
        owners = set(self._durable_owners())
        if os.path.isdir(self.base_dir):
            owners.update(
                d for d in os.listdir(self.base_dir)
                if os.path.isdir(os.path.join(self.base_dir, d)))
        out = []
        for owner in sorted(owners):
            try:
                out.extend(self.list_meta(owner))
            except ValueError:
                continue
        return out

    def get_meta(self, owner: str, ds_id: str) -> Optional[DatasetMeta]:
        mkey = self._mkey(owner, ds_id)
        with self._lock:
            if mkey in self._mem:
                return self._mem[mkey]["meta"]
        p = self._path(owner, ds_id, "meta.json")
        if not os.path.exists(p) and not self._restore(owner, ds_id,
                                                       "meta.json"):
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
            had_memory = self._mem.pop(mkey, None) is not None
            self._caches.pop(mkey, None)
            self._hashes.pop(mkey, None)
            d = self._dir(owner, ds_id)
            existed = os.path.isdir(d)
            if existed:
                shutil.rmtree(d, ignore_errors=True)
            # And from the durable store, or the next cold container
            # restores a dataset the client deleted.
            if self.blobs is not None:
                try:
                    if self.blobs.delete_prefix(
                            "{}/{}".format(self._safe(owner),
                                           self._safe(ds_id))):
                        existed = True
                except Exception:
                    logger.warning("could not remove %s/%s from durable "
                                   "storage", owner, ds_id, exc_info=True)
        # The in-memory entry is dropped either way, so reporting False
        # when only the directory was missing told the caller "not
        # found" about a dataset it had just deleted — the API answered
        # 404 and the row vanished from the list at the same time.
        return bool(existed or had_memory)

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
        # The integrity layer writes straight into the dataset directory,
        # so the store has to carry its files across too — a restored
        # dataset whose audit trail did not come with it can no longer
        # say what was done to it, which is the whole point of having one.
        self._back_up(owner, ds_id, "integrity.json", "audit.jsonl")

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
        self._back_up(owner, ds_id, "integrity.json", "audit.jsonl")

    def integrity(self, owner: str, ds_id: str) -> Optional[dict]:
        """Recomputed on every call, deliberately: an integrity verdict
        that is cached is a verdict about the past."""
        if self.get_meta(owner, ds_id) is None:
            return None
        self._restore(owner, ds_id, "integrity.json")
        self._restore(owner, ds_id, "audit.jsonl")
        return integrity.summary(self._dir(owner, ds_id),
                                 self.get_raw_df(owner, ds_id),
                                 self.get_df(owner, ds_id))

    # ── analysis caches (hash-invalidated) ───────────────
    def cache_get(self, owner: str, ds_id: str, key: str) -> Optional[Any]:
        df = self.get_df(owner, ds_id)
        if df is None:
            return None
        mkey = self._mkey(owner, ds_id)
        h = self._frame_hash(mkey, df)
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
        mkey = self._mkey(owner, ds_id)
        h = self._frame_hash(mkey, df)
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
    def _frame_hash(self, mkey: str, df: pd.DataFrame) -> str:
        """Content hash of `df`, remembered against that exact object.

        `update_active` replaces the frame rather than mutating it, so an
        identity check is enough to know the memo is stale, and it can
        never report a changed frame as unchanged."""
        with self._lock:
            entry = self._hashes.get(mkey)
            if entry is not None and entry[0] is df:
                return entry[1]
        h = self._content_hash(df)
        with self._lock:
            self._hashes[mkey] = (df, h)
        return h

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
        self._back_up(owner, ds_id, "meta.json")

    def _save_df(self, owner: str, ds_id: str, which: str,
                 df: pd.DataFrame) -> list[str]:
        written = write_frame(self._path(owner, ds_id, f"{which}.parquet"), df)
        self._back_up(owner, ds_id, "{}.parquet".format(which),
                      "{}.parquet.schema.json".format(which))
        return written

    def _load(self, which: str, owner: str, ds_id: str) -> Optional[pd.DataFrame]:
        mkey = self._mkey(owner, ds_id)
        with self._lock:
            if mkey in self._mem:
                return self._mem[mkey][which]
        # A cold container has the dataset in the durable store and
        # nothing on disk. Fetch it rather than answering "no such
        # dataset" about something the client uploaded an hour ago.
        self._restore(owner, ds_id, f"{which}.parquet")
        self._restore(owner, ds_id, f"{which}.parquet.schema.json")
        df = read_frame(self._path(owner, ds_id, f"{which}.parquet"))
        if df is None:
            return None
        other = "raw" if which == "active" else "active"
        self._restore(owner, ds_id, f"{other}.parquet")
        self._restore(owner, ds_id, f"{other}.parquet.schema.json")
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
            # The hash memo holds a strong reference to the frame it
            # describes, so leaving it behind here would keep in memory
            # exactly the dataset this eviction is trying to release.
            self._hashes.pop(oldest, None)

    @staticmethod
    def _content_hash(df: pd.DataFrame) -> str:
        """Every cell, not a sample of them.

        This used to hash shape, dtypes and the first 100 rows. Cleaning
        that only touched later rows — trimming whitespace, capping
        outliers, imputing a gap that starts at row 500 — left that
        signature identical, so `cache_get` returned the analysis of the
        *uncleaned* frame. The EDA page then said a column held two
        distinct values while the profile page, which recomputes, said
        one. `frame_digest` reads the whole frame and has the fallback
        for object columns holding unhashable cells."""
        return integrity.frame_digest(df)


store = DatasetStore()
