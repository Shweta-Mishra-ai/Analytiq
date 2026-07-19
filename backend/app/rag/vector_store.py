"""
rag/vector_store.py — small, dependency-free vector store.

Embeddings come from Gemini text-embedding-004 when a key is configured.
Without a key we fall back to a deterministic local hashing embedder —
retrieval quality is lower but the whole pipeline stays testable offline.

Persistence: one pickle per knowledge base under DATA_DIR/rag/.
Scale target is thousands of chunks per KB, where brute-force cosine
similarity with numpy is faster than spinning up a vector DB.
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import re
import threading
import time
import uuid
from typing import List, Optional

import numpy as np

from app.config import config

logger = logging.getLogger(__name__)

_EMBED_BATCH = 90
_LOCAL_DIM = 384


# ── embedding backends ───────────────────────────────────

def _gemini_embed(texts: List[str], task: str) -> Optional[np.ndarray]:
    if not config.gemini_api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=config.gemini_api_key)
        vecs: list = []
        for i in range(0, len(texts), _EMBED_BATCH):
            batch = texts[i:i + _EMBED_BATCH]
            resp = genai.embed_content(
                model=f"models/{config.gemini_embed_model}",
                content=batch, task_type=task)
            vecs.extend(resp["embedding"])
        return np.array(vecs, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Gemini embeddings failed, using local fallback: {e}")
        return None


def _local_embed(texts: List[str]) -> np.ndarray:
    """Deterministic hashed bag-of-words embedding (offline fallback)."""
    out = np.zeros((len(texts), _LOCAL_DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            out[i, h % _LOCAL_DIM] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def embed(texts: List[str], task: str = "retrieval_document") -> tuple[np.ndarray, str]:
    vecs = _gemini_embed(texts, task)
    if vecs is not None:
        return vecs, "gemini"
    return _local_embed(texts), "local"


# ── store ────────────────────────────────────────────────

class KnowledgeBase:
    def __init__(self, kb_id: str, name: str, path: str):
        self.kb_id = kb_id
        self.name = name
        self.path = path
        self.created_at = time.time()
        self.chunks: List[dict] = []      # {id,text,source,locator,kind}
        self.vectors: Optional[np.ndarray] = None
        self.embedder: str = ""
        self.files: List[dict] = []       # {filename,kind,chunks,added_at}

    # persistence ---------------------------------------------------------
    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> "KnowledgeBase":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ops -----------------------------------------------------------------
    def add_chunks(self, chunks: List[dict], filename: str, kind: str):
        texts = [c["text"] for c in chunks]
        vecs, backend = embed(texts, "retrieval_document")
        if self.vectors is not None and self.embedder and backend != self.embedder:
            # re-embed everything with the new backend to keep space consistent
            all_texts = [c["text"] for c in self.chunks] + texts
            all_vecs, backend = embed(all_texts, "retrieval_document")
            self.vectors = all_vecs[:len(self.chunks)]
            vecs = all_vecs[len(self.chunks):]
        for c in chunks:
            c["id"] = uuid.uuid4().hex[:8]
        self.chunks.extend(chunks)
        self.vectors = vecs if self.vectors is None else np.vstack([self.vectors, vecs])
        self.embedder = backend
        self.files.append({"filename": filename, "kind": kind,
                           "chunks": len(chunks), "added_at": time.time()})
        self.save()

    def search(self, query: str, k: int = 6) -> List[dict]:
        if self.vectors is None or not len(self.chunks):
            return []
        if self.embedder == "gemini":
            qv, backend = embed([query], "retrieval_query")
            if backend != "gemini":   # key vanished mid-session
                return []
        else:
            qv = _local_embed([query])
        qv = qv[0]
        mat = self.vectors
        sims = (mat @ qv) / (
            (np.linalg.norm(mat, axis=1) * np.linalg.norm(qv)) + 1e-9)
        idx = np.argsort(-sims)[:k]
        return [{**self.chunks[i], "score": float(sims[i])} for i in idx]


class RagStore:
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(config.data_dir, "rag")
        os.makedirs(self.base_dir, exist_ok=True)
        self._lock = threading.RLock()
        self._cache: dict[str, KnowledgeBase] = {}

    def _path(self, kb_id: str) -> str:
        return os.path.join(self.base_dir, f"{kb_id}.pkl")

    def create(self, name: str) -> KnowledgeBase:
        kb_id = uuid.uuid4().hex[:12]
        kb = KnowledgeBase(kb_id, name, self._path(kb_id))
        with self._lock:
            kb.save()
            self._cache[kb_id] = kb
        return kb

    def get(self, kb_id: str) -> Optional[KnowledgeBase]:
        with self._lock:
            if kb_id in self._cache:
                return self._cache[kb_id]
            p = self._path(kb_id)
            if not os.path.exists(p):
                return None
            kb = KnowledgeBase.load(p)
            kb.path = p
            self._cache[kb_id] = kb
            return kb

    def list(self) -> List[dict]:
        out = []
        for fn in os.listdir(self.base_dir):
            if fn.endswith(".pkl"):
                kb = self.get(fn[:-4])
                if kb:
                    out.append({"kb_id": kb.kb_id, "name": kb.name,
                                "files": len(kb.files),
                                "chunks": len(kb.chunks),
                                "created_at": kb.created_at})
        out.sort(key=lambda x: x["created_at"], reverse=True)
        return out

    def delete(self, kb_id: str) -> bool:
        with self._lock:
            self._cache.pop(kb_id, None)
            p = self._path(kb_id)
            if os.path.exists(p):
                os.unlink(p)
                return True
        return False


rag_store = RagStore()
