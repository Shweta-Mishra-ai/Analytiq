"""
services/llm_cache.py — don't pay twice for the same paragraph.

Regenerating a report re-ran every LLM call: the same executive summary
for the same dataset, the same narrative for the same chart, billed
again and adding thirty seconds to a rebuild that changed nothing.

Keyed on the prompt itself rather than on a dataset id, so a cache entry
is only reused when the request is genuinely identical — a cleaned
dataset produces a different prompt and correctly misses. Entries live on
disk beside the datasets, so a restart does not throw them away, and they
expire so a stale narrative cannot outlive the data it describes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# A narrative older than this is regenerated. Long enough that a working
# session reuses it, short enough that it cannot describe last month's data.
DEFAULT_TTL_SEC = 7 * 24 * 3600
MAX_ENTRIES = 2000


def _key(system: str, user: str, task: str, model: str) -> str:
    digest = hashlib.sha256()
    for part in (system, user, task, model):
        digest.update(str(part).encode("utf-8", "replace"))
        digest.update(b"\x00")
    return digest.hexdigest()


class LLMCache:
    def __init__(self, base_dir: Optional[str] = None,
                 ttl: int = DEFAULT_TTL_SEC):
        self.ttl = ttl
        if base_dir is None:
            from app.config import config
            base_dir = os.path.join(config.data_dir, "llm_cache")
        self.dir = base_dir
        self.hits = 0
        self.misses = 0
        try:
            os.makedirs(self.dir, exist_ok=True)
        except Exception:
            logger.warning("could not create the LLM cache directory at %s — "
                           "running without a cache", self.dir, exc_info=True)
            self.dir = ""

    def _path(self, key: str) -> str:
        return os.path.join(self.dir, key[:2], key + ".json")

    def get(self, system: str, user: str, task: str,
            model: str) -> Optional[str]:
        if not self.dir:
            return None
        path = self._path(_key(system, user, task, model))
        try:
            if not os.path.exists(path):
                self.misses += 1
                return None
            if time.time() - os.path.getmtime(path) > self.ttl:
                os.remove(path)
                self.misses += 1
                return None
            with open(path, "r", encoding="utf-8") as fh:
                self.hits += 1
                return json.load(fh).get("text")
        except Exception:
            logger.debug("cache read failed for %s", path, exc_info=True)
            self.misses += 1
            return None

    def put(self, system: str, user: str, task: str, model: str,
            text: str) -> None:
        if not self.dir or not text:
            return
        path = self._path(_key(system, user, task, model))
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"task": task, "model": model, "text": text}, fh)
            os.replace(tmp, path)
        except Exception:
            logger.debug("cache write failed for %s", path, exc_info=True)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits, "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "enabled": bool(self.dir),
        }

    def clear(self) -> int:
        """Drop every entry. Returns how many were removed."""
        if not self.dir:
            return 0
        removed = 0
        for root, _dirs, files in os.walk(self.dir):
            for name in files:
                try:
                    os.remove(os.path.join(root, name))
                    removed += 1
                except Exception:
                    logger.debug("could not remove %s", name, exc_info=True)
        return removed


llm_cache = LLMCache()
