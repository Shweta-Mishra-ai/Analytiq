"""
services/cleanup.py — storage lifecycle for uploaded datasets and RAG
knowledge bases.

Without this, `DatasetStore`/`RagStore` write to disk forever with no
expiry — fine for a demo, but a real deployment needs old data to age
out automatically. This sweeps anything past `DATA_TTL_DAYS` old.

Runs on a background loop from app startup (see main.py's lifespan)
and can also be triggered on demand via POST /api/admin/cleanup.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from app.config import config
from app.services.dataset_store import store as dataset_store

logger = logging.getLogger(__name__)


@dataclass
class CleanupResult:
    datasets_deleted: list[str] = field(default_factory=list)
    kbs_deleted: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_deleted(self) -> int:
        return len(self.datasets_deleted) + len(self.kbs_deleted)


def sweep_expired(ttl_days: int | None = None) -> CleanupResult:
    """Delete datasets and RAG knowledge bases older than the TTL.
    Synchronous and safe to call directly (e.g. from a test or an
    admin endpoint) as well as from the background loop."""
    ttl = ttl_days if ttl_days is not None else config.data_ttl_days
    cutoff = time.time() - ttl * 86400
    result = CleanupResult()

    if ttl <= 0:
        return result  # 0 or negative disables expiry entirely

    for meta in dataset_store.list_all_meta():
        if meta.uploaded_at < cutoff:
            try:
                if dataset_store.delete(meta.owner, meta.dataset_id):
                    result.datasets_deleted.append(meta.dataset_id)
            except Exception as e:
                result.errors.append(f"dataset {meta.dataset_id}: {e}")

    try:
        from app.rag.vector_store import RagStore
        rag_store = RagStore()
        for kb in rag_store.list_all():
            if kb.get("created_at", 0) < cutoff:
                try:
                    if rag_store.delete(kb["owner"], kb["kb_id"]):
                        result.kbs_deleted.append(kb["kb_id"])
                except Exception as e:
                    result.errors.append(f"kb {kb.get('kb_id')}: {e}")
    except ImportError:
        pass  # RAG deps not installed in this environment

    if result.total_deleted:
        logger.info(
            f"Cleanup: removed {len(result.datasets_deleted)} dataset(s), "
            f"{len(result.kbs_deleted)} knowledge base(s) older than "
            f"{ttl} day(s)")
    if result.errors:
        logger.warning(f"Cleanup errors: {result.errors}")
    return result


async def cleanup_loop(interval_hours: float | None = None) -> None:
    """Background task: sweep on startup, then every `interval_hours`.
    Cancelled cleanly on app shutdown (see main.py lifespan)."""
    interval = interval_hours if interval_hours is not None \
        else config.cleanup_interval_hours
    while True:
        try:
            sweep_expired()
        except Exception:
            logger.exception("Scheduled cleanup sweep failed")
        await asyncio.sleep(max(interval, 0.1) * 3600)
