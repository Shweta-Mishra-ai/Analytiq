"""
services/artifacts.py — a finished report is a thing, not a download.

Until now a report existed only for the duration of the HTTP response
that built it. That single fact is behind three separate limitations:

  * a 3-second build had to be held on the request thread, because there
    was nowhere to put the result;
  * a client could not be sent a link, only a file, so delivering a
    report meant a human attaching it to an email; and
  * nothing could be produced on a schedule, because a schedule needs
    somewhere to leave what it produced.

Giving the output an identity and a home fixes all three at once, which
is why this is one module rather than three.

Design, and what each choice refuses:

**Owned, like everything else.** An artifact belongs to the account that
made it and lives under that account's directory. Reading one is scoped
by owner exactly as a dataset is, so the isolation story does not
acquire a second, weaker path.

**Bytes in a blob store, metadata in JSON.** A report is just bytes
plus a few strings — a pickle of an object graph would be a
deserialisation bug waiting to be found. Where those bytes physically
live is not this module's business: it asks services/blobstore, which
is a local directory by default and an S3-compatible bucket when one is
configured. That indirection is the whole reason a report can now
outlive the container that built it, which is what share links and
scheduled delivery were always promising.

**They expire.** A report is a photograph of a dataset at a moment, and
an old one is misleading rather than useful. They are swept on a
retention window, and the sweep is part of this module rather than a
cron job somebody has to remember to configure.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from app.config import config
from app.services.blobstore import (BlobStore, LocalBlobStore,
                                    build_store)

logger = logging.getLogger(__name__)

# How long a generated report stays retrievable. Long enough that a
# scheduled Monday report is still there on Friday; short enough that
# nobody circulates a link to a picture of last quarter.
DEFAULT_RETENTION_DAYS = 30

MEDIA_TYPES = {
    "pdf":  "application/pdf",
    "pptx": ("application/vnd.openxmlformats-officedocument"
             ".presentationml.presentation"),
    "csv":  "text/csv",
    "xlsx": ("application/vnd.openxmlformats-officedocument"
             ".spreadsheetml.sheet"),
}


@dataclass
class Artifact:
    artifact_id: str
    owner: str
    dataset_id: str
    kind: str                 # "report" | "health-report" | "deck" | ...
    fmt: str                  # "pdf" | "pptx" | ...
    filename: str
    size_bytes: int
    created_at: float
    title: str = ""
    # The job that produced it, so a failure can be traced back from the
    # thing the client is looking at to the run that made it.
    job_id: str = ""
    expires_at: float = 0.0
    meta: Dict = field(default_factory=dict)

    @property
    def media_type(self) -> str:
        return MEDIA_TYPES.get(self.fmt, "application/octet-stream")

    @property
    def expired(self) -> bool:
        return bool(self.expires_at) and time.time() > self.expires_at


class ArtifactStore:
    """Generated files, owned and expiring."""

    def __init__(self, base_dir: Optional[str] = None,
                 blobs: Optional["BlobStore"] = None):
        # base_dir is still accepted so a test can point one store at a
        # temporary directory without knowing about blob stores.
        if blobs is not None:
            self.blobs = blobs
        elif base_dir is not None:
            self.blobs = LocalBlobStore(base_dir)
        else:
            self.blobs = build_store("artifacts")
        self.base_dir = getattr(self.blobs, "base_dir", "")
        self._lock = threading.RLock()

    # ── paths ────────────────────────────────────────────
    def _prefix(self, owner: str, artifact_id: str = "") -> str:
        parts = [_safe(owner)]
        if artifact_id:
            parts.append(_safe(artifact_id))
        return "/".join(parts)

    def _meta_key(self, owner: str, artifact_id: str) -> str:
        return self._prefix(owner, artifact_id) + "/meta.json"

    def _blob_key(self, owner: str, artifact_id: str) -> str:
        return self._prefix(owner, artifact_id) + "/blob"

    # ── writing ──────────────────────────────────────────
    def put(self, owner: str, dataset_id: str, kind: str, fmt: str,
            data: bytes, filename: str, title: str = "",
            job_id: str = "", meta: Optional[Dict] = None,
            retention_days: int = DEFAULT_RETENTION_DAYS) -> Artifact:
        artifact_id = uuid.uuid4().hex[:16]
        now = time.time()
        art = Artifact(
            artifact_id=artifact_id, owner=owner, dataset_id=dataset_id,
            kind=kind, fmt=fmt, filename=filename, size_bytes=len(data),
            created_at=now, title=title, job_id=job_id,
            expires_at=now + retention_days * 86400, meta=dict(meta or {}),
        )
        with self._lock:
            # Bytes first, metadata second. Nothing is visible to a
            # reader until both have landed, because get() is what
            # decides an artifact exists — the reverse order would
            # publish a record whose file is not there yet.
            self.blobs.put(self._blob_key(owner, artifact_id), data)
            self.blobs.put(self._meta_key(owner, artifact_id),
                           json.dumps(asdict(art)).encode("utf-8"))
        logger.info("artifact %s stored (%s, %.1f KB) for %s",
                    artifact_id, kind, len(data) / 1024, owner)
        return art

    # ── reading ──────────────────────────────────────────
    def get(self, owner: str, artifact_id: str) -> Optional[Artifact]:
        raw = self.blobs.get(self._meta_key(owner, artifact_id))
        if raw is None:
            return None
        try:
            art = Artifact(**json.loads(raw.decode("utf-8")))
        except Exception:
            logger.warning("unreadable artifact metadata for %s/%s", owner,
                           artifact_id, exc_info=True)
            return None
        return None if art.expired else art

    def read(self, owner: str, artifact_id: str) -> Optional[bytes]:
        if self.get(owner, artifact_id) is None:
            return None
        data = self.blobs.get(self._blob_key(owner, artifact_id))
        if data is None:
            logger.warning("artifact %s has metadata but no bytes", artifact_id)
        return data

    def list(self, owner: str, dataset_id: str = "",
             limit: int = 50) -> List[Artifact]:
        out: List[Artifact] = []
        for artifact_id in self._ids_for(owner):
            art = self.get(owner, artifact_id)
            if art is None:
                continue
            if dataset_id and art.dataset_id != dataset_id:
                continue
            out.append(art)
        out.sort(key=lambda a: a.created_at, reverse=True)
        return out[:limit]

    def update(self, art: Artifact) -> None:
        """Re-write an artifact's metadata in place.

        Sharing attaches and revokes a token on the record. It used to
        do that by writing the store's own metadata path itself, which
        meant renaming that path broke sharing silently — the artifact
        store is the only thing that should know where its metadata is.
        """
        with self._lock:
            self.blobs.put(self._meta_key(art.owner, art.artifact_id),
                           json.dumps(asdict(art)).encode("utf-8"))

    # ── housekeeping ─────────────────────────────────────
    def delete(self, owner: str, artifact_id: str) -> bool:
        with self._lock:
            return self.blobs.delete_prefix(self._prefix(owner, artifact_id))

    def sweep(self) -> int:
        """Remove what has expired. Returns how many went.

        Part of this module rather than an operational chore: a
        retention window nobody runs is a promise the product does not
        keep, and the sweep is cheap enough to run on a timer.
        """
        removed = 0
        for owner in self._owners():
            for artifact_id in self._ids_for(owner):
                raw = self.blobs.get(self._meta_key(owner, artifact_id))
                try:
                    expires = json.loads(raw.decode("utf-8")).get(
                        "expires_at", 0)
                except Exception:
                    # Unreadable metadata is itself a reason to remove
                    # it: nothing can serve it and it will never expire
                    # on its own.
                    #
                    # Except while it is being written. put() writes the
                    # blob, then the metadata, so a sweep landing in
                    # that window sees exactly this and would delete the
                    # report someone is at that moment waiting for.
                    # Anything written within the grace period is left
                    # alone; if it really is broken, the next sweep
                    # takes it.
                    if self._recent(owner, artifact_id):
                        continue
                    logger.debug("sweeping unreadable artifact %s/%s",
                                 owner, artifact_id, exc_info=True)
                    expires = 1
                if expires and time.time() > expires:
                    self.blobs.delete_prefix(self._prefix(owner, artifact_id))
                    removed += 1
        if removed:
            logger.info("swept %d expired artifact(s)", removed)
        return removed

    # ── walking the store ────────────────────────────────
    def _owners(self) -> List[str]:
        return sorted({key.split("/")[0]
                       for key in self.blobs.list_prefix("")
                       if "/" in key})

    def _ids_for(self, owner: str) -> List[str]:
        prefix = self._prefix(owner)
        ids = set()
        for key in self.blobs.list_prefix(prefix):
            parts = key.split("/")
            if len(parts) >= 3:
                ids.add(parts[1])
        return sorted(ids)

    def _recent(self, owner: str, artifact_id: str) -> bool:
        """Is something still writing this artifact?"""
        when = self.blobs.modified_at(self._blob_key(owner, artifact_id))
        if when is None:
            return False
        return time.time() - when < WRITE_GRACE_SECONDS


def _safe(component: str) -> str:
    """A single path component that cannot climb out of the store.

    Owner names and ids come from tokens and uuid4, so this is defence
    against a future caller rather than a current hole — which is the
    only time it is cheap to add.
    """
    cleaned = "".join(c for c in str(component)
                      if c.isalnum() or c in "-_.")
    return cleaned.strip(".") or "unknown"


# How long a half-written artifact directory is given before the sweep
# is willing to call it broken. Longer than any write takes, shorter
# than anyone would notice.
WRITE_GRACE_SECONDS = 300.0


store = ArtifactStore()
