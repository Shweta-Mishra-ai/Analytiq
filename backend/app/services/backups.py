"""
services/backups.py — snapshots of everything the server stores.

A snapshot is one .tar.gz of DATA_DIR: accounts, the signing key and every
client's data. They are taken on a schedule and kept beside the data, which
covers a deleted dataset, a bad release or a damaged file. They do NOT cover
losing the disk itself: for that, download one from the admin screen (or
fetch /api/admin/backup on a schedule) and keep it somewhere else.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tarfile
import threading
import time
from typing import List, Optional

from app.config import config

logger = logging.getLogger(__name__)

# Left out of a snapshot: earlier snapshots, and report downloads whose
# links expire within the hour.
_SKIP = ("data/backups", "data/mcp_reports")
_lock = threading.Lock()


def backup_dir() -> str:
    path = os.path.join(config.data_dir, "backups")
    os.makedirs(path, exist_ok=True)
    return path


def _keep(ti):
    return None if any(ti.name == s or ti.name.startswith(s + "/") for s in _SKIP) else ti


def write_archive(path: str) -> None:
    """The whole of DATA_DIR as a .tar.gz at `path`."""
    with tarfile.open(path, "w:gz", compresslevel=3) as tar:
        tar.add(config.data_dir, arcname="data", filter=_keep)


def list_backups() -> List[dict]:
    out = []
    folder = backup_dir()
    for name in os.listdir(folder):
        if name.startswith("analytiq-backup-") and name.endswith(".tar.gz"):
            st = os.stat(os.path.join(folder, name))
            out.append({"name": name, "created_at": st.st_mtime,
                        "size_mb": round(st.st_size / 2**20, 2)})
    return sorted(out, key=lambda b: -b["created_at"])


def _data_bytes() -> int:
    total = 0
    skip = (os.path.join(config.data_dir, "backups"),
            os.path.join(config.data_dir, "mcp_reports"))
    for root, dirs, files in os.walk(config.data_dir):
        dirs[:] = [d for d in dirs if os.path.join(root, d) not in skip]
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def take_backup(keep: Optional[int] = None) -> Optional[dict]:
    """Write a snapshot and drop the oldest beyond `keep`. Returns its
    entry, or None when it was skipped for lack of disk space: a snapshot
    that fills the disk would stop every upload."""
    keep = config.backup_keep if keep is None else keep
    with _lock:
        folder = backup_dir()
        need = _data_bytes()
        free = shutil.disk_usage(folder).free
        if free - need < config.min_free_disk_mb * 2**20:
            logger.warning("backup skipped: %d MB of data, %d MB free on the disk",
                           need // 2**20, free // 2**20)
            return None
        name = "analytiq-backup-{}.tar.gz".format(time.strftime("%Y%m%d-%H%M%S"))
        tmp = os.path.join(folder, name + ".part")
        try:
            write_archive(tmp)
            os.replace(tmp, os.path.join(folder, name))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        for old in list_backups()[max(keep, 1):]:
            try:
                os.unlink(os.path.join(folder, old["name"]))
            except OSError:
                logger.warning("could not remove old backup %s", old["name"])
        logger.info("backup written: %s", name)
        return next((b for b in list_backups() if b["name"] == name), None)


def backup_path(name: str) -> Optional[str]:
    """The file of a listed snapshot; None for anything else, so a name
    from a request cannot reach outside the folder."""
    return (os.path.join(backup_dir(), name)
            if any(b["name"] == name for b in list_backups()) else None)


async def backup_loop() -> None:
    """Background task: a snapshot whenever the newest is older than
    BACKUP_EVERY_HOURS. Checked hourly, so a restart does not reset the
    clock and does not take a snapshot on every start."""
    while True:
        every = config.backup_every_hours
        if every > 0:
            try:
                existing = list_backups()
                if not existing or time.time() - existing[0]["created_at"] >= every * 3600:
                    await asyncio.to_thread(take_backup)
            except Exception:
                logger.exception("scheduled backup failed")
        await asyncio.sleep(3600)
