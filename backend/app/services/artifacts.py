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

**Bytes on disk, metadata in JSON.** Same shape as the dataset store,
for the same reason: a pickle of an object graph is a deserialisation
bug waiting to be found, and a report is just bytes plus a few strings.

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

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(config.data_dir, "artifacts")
        os.makedirs(self.base_dir, exist_ok=True)
        self._lock = threading.RLock()

    # ── paths ────────────────────────────────────────────
    def _dir(self, owner: str, artifact_id: str) -> str:
        return os.path.join(self.base_dir, _safe(owner), _safe(artifact_id))

    def _meta_path(self, owner: str, artifact_id: str) -> str:
        return os.path.join(self._dir(owner, artifact_id), "meta.json")

    def _blob_path(self, owner: str, artifact_id: str) -> str:
        return os.path.join(self._dir(owner, artifact_id), "blob")

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
        directory = self._dir(owner, artifact_id)
        os.makedirs(directory, exist_ok=True)
        with self._lock:
            _atomic_write(self._blob_path(owner, artifact_id), data)
            _atomic_write(self._meta_path(owner, artifact_id),
                          json.dumps(asdict(art)).encode("utf-8"))
        logger.info("artifact %s stored (%s, %.1f KB) for %s",
                    artifact_id, kind, len(data) / 1024, owner)
        return art

    # ── reading ──────────────────────────────────────────
    def get(self, owner: str, artifact_id: str) -> Optional[Artifact]:
        path = self._meta_path(owner, artifact_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                art = Artifact(**json.load(fh))
        except Exception:
            logger.warning("unreadable artifact metadata at %s", path,
                           exc_info=True)
            return None
        return None if art.expired else art

    def read(self, owner: str, artifact_id: str) -> Optional[bytes]:
        if self.get(owner, artifact_id) is None:
            return None
        path = self._blob_path(owner, artifact_id)
        if not os.path.exists(path):
            logger.warning("artifact %s has metadata but no bytes", artifact_id)
            return None
        with open(path, "rb") as fh:
            return fh.read()

    def list(self, owner: str, dataset_id: str = "",
             limit: int = 50) -> List[Artifact]:
        root = os.path.join(self.base_dir, _safe(owner))
        if not os.path.isdir(root):
            return []
        out: List[Artifact] = []
        for artifact_id in os.listdir(root):
            art = self.get(owner, artifact_id)
            if art is None:
                continue
            if dataset_id and art.dataset_id != dataset_id:
                continue
            out.append(art)
        out.sort(key=lambda a: a.created_at, reverse=True)
        return out[:limit]

    # ── housekeeping ─────────────────────────────────────
    def delete(self, owner: str, artifact_id: str) -> bool:
        import shutil
        directory = self._dir(owner, artifact_id)
        with self._lock:
            if os.path.isdir(directory):
                shutil.rmtree(directory, ignore_errors=True)
                return True
        return False

    def sweep(self) -> int:
        """Remove what has expired. Returns how many went.

        Part of this module rather than an operational chore: a
        retention window nobody runs is a promise the product does not
        keep, and the sweep is cheap enough to run on a timer.
        """
        removed = 0
        if not os.path.isdir(self.base_dir):
            return 0
        for owner in os.listdir(self.base_dir):
            owner_dir = os.path.join(self.base_dir, owner)
            if not os.path.isdir(owner_dir):
                continue
            for artifact_id in os.listdir(owner_dir):
                directory = os.path.join(owner_dir, artifact_id)
                meta = os.path.join(directory, "meta.json")
                try:
                    with open(meta, "r", encoding="utf-8") as fh:
                        expires = json.load(fh).get("expires_at", 0)
                except Exception:
                    # Unreadable metadata is itself a reason to remove
                    # it: nothing can serve it and it will never expire
                    # on its own.
                    #
                    # Except while it is being written. put() creates
                    # the directory, writes the blob, then writes the
                    # metadata, so a sweep landing in that window sees
                    # exactly this and would delete the report someone
                    # is at that moment waiting for. A directory younger
                    # than the grace period is left alone; if it really
                    # is broken, the next sweep takes it.
                    if _recent(directory):
                        continue
                    logger.debug("sweeping unreadable artifact at %s", meta,
                                 exc_info=True)
                    expires = 1
                if expires and time.time() > expires:
                    import shutil
                    shutil.rmtree(directory, ignore_errors=True)
                    removed += 1
        if removed:
            logger.info("swept %d expired artifact(s)", removed)
        return removed


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


def _recent(directory: str) -> bool:
    """Was this directory made moments ago — i.e. is something still
    writing into it?"""
    try:
        return time.time() - os.path.getmtime(directory) < WRITE_GRACE_SECONDS
    except OSError:
        return False


def _atomic_write(path: str, data: bytes) -> None:
    # The pid as well as the thread id: two workers of the same server
    # share this directory and can hold the same thread id, and two
    # writers landing on one temp name would interleave their bytes.
    tmp = "{}.{}.{}.tmp".format(path, os.getpid(), threading.get_ident())
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


store = ArtifactStore()
