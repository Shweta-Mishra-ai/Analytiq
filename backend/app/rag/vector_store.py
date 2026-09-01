"""
rag/vector_store.py — small, dependency-free vector store.

Embeddings come from Gemini's embedding model when a key is configured.
Without a key we fall back to a deterministic local hashing embedder —
retrieval quality is lower but the whole pipeline stays testable offline.

Persistence: a JSON manifest plus a .npy of embeddings per
knowledge base under DATA_DIR/rag/.
Scale target is thousands of chunks per KB, where brute-force cosine
similarity with numpy is faster than spinning up a vector DB.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from typing import List, Optional

import numpy as np

from app.config import config
from app.rag.retrieval import (BM25Index, CrossEncoderReranker,
                               hybrid_search)

logger = logging.getLogger(__name__)

# Small, fast, and good enough that offline retrieval stops
# being keyword matching. 384 dimensions, ~90MB.
LOCAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_EMBED_BATCH = 90
_LOCAL_DIM = 384


# ── embedding backends ───────────────────────────────────

def _gemini_embed(texts: List[str], task: str) -> Optional[np.ndarray]:
    if not config.gemini_api_key:
        return None
    from app.ai import gemini_client
    vecs: list = []
    for i in range(0, len(texts), _EMBED_BATCH):
        batch = texts[i:i + _EMBED_BATCH]
        result = gemini_client.embed(batch, task=task)
        if result is None:
            return None  # gemini_client already logged the failure
        vecs.extend(result)
    return np.array(vecs, dtype=np.float32)


# Function words carry no topic. Left in, they dominate the similarity:
# "what was warehouse throughput in the third quarter" matched a passage
# about probation periods at 0.16 — entirely on "the", "in" and "was" —
# which is enough to clear any floor low enough to admit real matches.
# Removing them takes that same off-topic query to 0.0 while a genuine
# question about leave rises from 0.32 to 0.42.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for
from by with without within into over under is are was were be been being
am do does did done have has had having it its as we our ours you your they
them their he she him her his hers i me my mine not no nor so such about up
down out off only own same too very can will just should would could may
might must shall there here when where which who whom whose what why how all
any both each few more most other some only per via
""".split())


def _tokens(text: str) -> List[str]:
    """Topic-bearing tokens, with plurals folded onto their singular.

    "expense claim deadline" should match a passage about "Expense claims"
    — without the fold it does not, because the hashed dimensions for
    "claim" and "claims" are unrelated.
    """
    out = []
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if len(tok) <= 2 or tok in _STOPWORDS:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


def _local_embed(texts: List[str]) -> np.ndarray:
    """Deterministic hashed bag-of-words embedding (offline fallback)."""
    out = np.zeros((len(texts), _LOCAL_DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            out[i, h % _LOCAL_DIM] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


class _SentenceEmbedder:
    """all-MiniLM-L6-v2, loaded once, when it is available.

    This is what makes offline retrieval semantic rather than lexical.
    The hashed fallback below cannot match "annual leave" to "vacation
    entitlement" — the two share no token, so their hashed dimensions are
    unrelated and the cosine is zero. A sentence encoder puts them next
    to each other, which is the whole point of embedding text.

    Optional on purpose: the model is a download and a torch dependency,
    and an install without it still works, on BM25 and the hashed vector.
    """

    _model = None
    _tried = False

    @classmethod
    def get(cls):
        if cls._tried:
            return cls._model
        cls._tried = True
        try:
            from sentence_transformers import SentenceTransformer
            cls._model = SentenceTransformer(LOCAL_MODEL_NAME)
            logger.info("local sentence embedder loaded: %s", LOCAL_MODEL_NAME)
        except Exception as exc:
            logger.info("local sentence embedder unavailable (%s) — falling "
                        "back to hashed term vectors, which match on shared "
                        "words only", exc)
            cls._model = None
        return cls._model

    @classmethod
    def encode(cls, texts: List[str]) -> Optional[np.ndarray]:
        model = cls.get()
        if model is None:
            return None
        try:
            return np.asarray(
                model.encode(list(texts), normalize_embeddings=True,
                             show_progress_bar=False),
                dtype=np.float32)
        except Exception:
            logger.warning("sentence embedding failed", exc_info=True)
            return None


def embed(texts: List[str], task: str = "retrieval_document") -> tuple[np.ndarray, str]:
    """Vectors for a batch of texts, and which backend produced them.

    Order is deliberate: a configured API embedder is the strongest, a
    local sentence model is the strongest thing that needs no key, and
    the hashed vector is the floor that guarantees retrieval works at
    all. The name comes back because the relevance floor differs per
    backend — their cosines are not on the same scale.
    """
    vecs = _gemini_embed(texts, task)
    if vecs is not None:
        return vecs, "gemini"
    vecs = _SentenceEmbedder.encode(texts)
    if vecs is not None:
        return vecs, "sentence"
    return _local_embed(texts), "local"


# ── relevance ────────────────────────────────────────────

# Absolute cosine floor below which a chunk is not about the query at
# all. The two embedders live on different scales — Gemini's vectors put
# unrelated business text around 0.3-0.5, while the local hashing
# embedder scores unrelated text far lower because it only matches on
# shared tokens — so one number cannot serve both.
MIN_SCORE = {"gemini": 0.45, "sentence": 0.28, "local": 0.12}
# A chunk scoring far below the best match is padding: it was returned
# only because k slots had to be filled.
RELATIVE_FLOOR = 0.55


def _filter_by_relevance(hits: List[dict], embedder: str) -> List[dict]:
    """Drop chunks that are not about the query.

    An empty result is a correct and useful answer — it is what lets the
    caller say "your documents do not cover this" instead of composing a
    confident paragraph out of the nearest unrelated text.
    """
    if not hits:
        return []
    floor = MIN_SCORE.get(embedder or "local", MIN_SCORE["local"])
    top = hits[0]["score"]
    if top < floor:
        return []
    return [h for h in hits if h["score"] >= max(floor, top * RELATIVE_FLOOR)]


# ── store ────────────────────────────────────────────────

class KnowledgeBase:
    def __init__(self, kb_id: str, name: str, path: str):
        self.kb_id = kb_id
        self.name = name
        self.path = path
        self.owner: str = ""
        self.created_at = time.time()
        self.chunks: List[dict] = []      # {id,text,source,locator,kind}
        self.vectors: Optional[np.ndarray] = None
        self.embedder: str = ""
        self.files: List[dict] = []       # {filename,kind,chunks,added_at}

    # persistence ---------------------------------------------------------
    # A knowledge base is text plus an array of floats, so it is stored as
    # JSON plus a .npy — not as a pickle of this object. Unpickling runs
    # whatever the file says to run, and a knowledge base is built from
    # user-uploaded documents in a directory the server later re-reads.
    @property
    def _vec_path(self) -> str:
        return self.path[:-5] + ".vec.npy" if self.path.endswith(".json") \
            else self.path + ".vec.npy"

    def save(self):
        payload = {
            "kb_id": self.kb_id,
            "name": self.name,
            "owner": self.owner,
            "created_at": self.created_at,
            "chunks": self.chunks,
            "embedder": self.embedder,
            "files": self.files,
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, self.path)
        if self.vectors is None:
            try:
                os.remove(self._vec_path)
            except FileNotFoundError:
                pass
        else:
            vtmp = self._vec_path + ".tmp"
            with open(vtmp, "wb") as f:
                np.save(f, np.asarray(self.vectors), allow_pickle=False)
            os.replace(vtmp, self._vec_path)

    @staticmethod
    def load(path: str) -> "KnowledgeBase":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        kb = KnowledgeBase(payload["kb_id"], payload["name"], path)
        kb.owner = payload.get("owner", "")
        kb.created_at = float(payload.get("created_at", time.time()))
        kb.chunks = payload.get("chunks", [])
        kb.embedder = payload.get("embedder", "")
        kb.files = payload.get("files", [])
        if os.path.exists(kb._vec_path):
            # allow_pickle stays off: a .npy is only ever an array here,
            # and the flag is the whole point of not using a pickle.
            with open(kb._vec_path, "rb") as f:
                kb.vectors = np.load(f, allow_pickle=False)
        return kb

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
        """The k best-matching chunks that are actually about the query.

        Returning the k least-irrelevant chunks regardless of score was
        the single thing that let this answer questions the documents do
        not cover: ask a KB of HR policies about churn and it handed the
        model six unrelated paragraphs, which the model then answered
        from. Nothing in the pipeline afterwards can recover from that —
        the citations even look right, because they point at the
        paragraphs that were supplied.
        """
        if not len(self.chunks):
            return []

        # Dense half — only when the query can be embedded the same way
        # the chunks were. A query vector from a different backend is not
        # comparable to the stored ones.
        qv = None
        if self.vectors is not None:
            if self.embedder == "gemini":
                vec, backend = embed([query], "retrieval_query")
                qv = vec[0] if backend == "gemini" else None
            elif self.embedder == "sentence":
                vec = _SentenceEmbedder.encode([query])
                qv = vec[0] if vec is not None else None
            else:
                qv = _local_embed([query])[0]

        mat = self.vectors
        if mat is not None and qv is not None:
            # Normalise once so the fusion sees a cosine, whichever
            # backend produced the vectors.
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
            qv = qv / (np.linalg.norm(qv) or 1.0)
        else:
            mat = None

        # Sparse half — exact terms a dense model has no way to hold: a
        # policy number, a surname, an SKU.
        bm25 = self._bm25_index()

        fused = hybrid_search(query, mat, qv, bm25,
                              k=max(k * 3, 12), candidates=30)
        if not fused:
            return []

        # Reranking reads each candidate against the question rather than
        # comparing two vectors built in isolation. Too slow for a
        # corpus, affordable on a shortlist, and it is what moves the
        # right passage from rank six to rank one.
        order = [i for i, _ in fused]
        if CrossEncoderReranker.available():
            texts = [self.chunks[i].get("text", "") for i in order]
            ranked = CrossEncoderReranker.rerank(query, texts, top_k=k)
            order = [order[pos] for pos, _score in ranked]

        scores = dict(fused)
        hits = []
        for i in order[:k]:
            chunk = self.chunks[i]
            # The cosine, where there is one, is what the relevance floor
            # is calibrated against — a fused RRF score has no absolute
            # meaning.
            cosine = float(mat[i] @ qv) if mat is not None and qv is not None \
                else float(scores.get(i, 0.0))
            hits.append({**chunk, "score": cosine,
                         "fusion_score": float(scores.get(i, 0.0))})
        return _filter_by_relevance(hits, self.embedder)

    def _bm25_index(self):
        """Built on first use and rebuilt when the chunks change."""
        signature = len(self.chunks)
        cached = getattr(self, "_bm25_cache", None)
        if cached is not None and cached[0] == signature:
            return cached[1]
        try:
            index = BM25Index([c.get("text", "") for c in self.chunks])
        except Exception:
            logger.warning("could not build the keyword index", exc_info=True)
            return None
        self._bm25_cache = (signature, index)
        return index


class RagStore:
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(config.data_dir, "rag")
        os.makedirs(self.base_dir, exist_ok=True)
        self._lock = threading.RLock()
        self._cache: dict[str, KnowledgeBase] = {}  # cache key: "owner/kb_id"

    @staticmethod
    def _safe(part: str) -> str:
        if not part or "/" in part or "\\" in part or part in (".", ".."):
            raise ValueError(f"Invalid path segment: {part!r}")
        return part

    def _owner_dir(self, owner: str) -> str:
        d = os.path.join(self.base_dir, self._safe(owner))
        os.makedirs(d, exist_ok=True)
        return d

    def _path(self, owner: str, kb_id: str) -> str:
        return os.path.join(self._owner_dir(owner), f"{self._safe(kb_id)}.json")

    def _ckey(self, owner: str, kb_id: str) -> str:
        return f"{owner}/{kb_id}"

    def create(self, owner: str, name: str) -> KnowledgeBase:
        kb_id = uuid.uuid4().hex[:12]
        kb = KnowledgeBase(kb_id, name, self._path(owner, kb_id))
        kb.owner = owner
        with self._lock:
            kb.save()
            self._cache[self._ckey(owner, kb_id)] = kb
        return kb

    def get(self, owner: str, kb_id: str) -> Optional[KnowledgeBase]:
        ckey = self._ckey(owner, kb_id)
        with self._lock:
            if ckey in self._cache:
                return self._cache[ckey]
            p = self._path(owner, kb_id)
            if not os.path.exists(p):
                return None
            kb = KnowledgeBase.load(p)
            kb.path = p
            self._cache[ckey] = kb
            return kb

    def list(self, owner: str) -> List[dict]:
        out = []
        owner_dir = os.path.join(self.base_dir, self._safe(owner))
        if not os.path.isdir(owner_dir):
            return out
        for fn in os.listdir(owner_dir):
            if fn.endswith(".json"):
                kb = self.get(owner, fn[:-5])
                if kb:
                    out.append({"kb_id": kb.kb_id, "name": kb.name,
                                "files": len(kb.files),
                                "chunks": len(kb.chunks),
                                "created_at": kb.created_at})
        out.sort(key=lambda x: x["created_at"], reverse=True)
        return out

    def list_all(self) -> List[dict]:
        """Every KB across every owner. Internal use only (cleanup sweep)."""
        out = []
        if not os.path.isdir(self.base_dir):
            return out
        for owner in sorted(os.listdir(self.base_dir)):
            if os.path.isdir(os.path.join(self.base_dir, owner)):
                for kb in self.list(owner):
                    out.append({**kb, "owner": owner})
        return out

    def delete(self, owner: str, kb_id: str) -> bool:
        with self._lock:
            self._cache.pop(self._ckey(owner, kb_id), None)
            p = self._path(owner, kb_id)
            if os.path.exists(p):
                os.unlink(p)
                # The embeddings live in a sibling file; deleting only the
                # manifest would leave the user's document vectors behind.
                try:
                    os.remove(p[:-5] + ".vec.npy")
                except FileNotFoundError:
                    pass
                return True
        return False


rag_store = RagStore()
