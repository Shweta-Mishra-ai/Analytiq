"""
rag/retrieval.py — finding the passage that answers the question.

The previous retriever was a single cosine search over a hashed
bag-of-words vector. That is keyword matching wearing the clothes of
semantic search: "how much annual leave do I get" never matched a
passage about "vacation entitlement", because the two share no token and
the hash dimensions for `leave` and `vacation` are unrelated. Worse, the
hashed vector counts raw term frequency, so a passage that says "the"
forty times competes with one that says "vacation" twice.

Three pieces fix it, and they fix different failures:

  * **Dense retrieval** finds passages that mean the same thing in
    different words. It misses exact strings — a policy number, a
    surname, an SKU — because those carry little meaning to embed.
  * **BM25** finds exactly those. It is the classic sparse ranking
    function: term frequency saturating rather than growing linearly,
    inverse document frequency so a rare word counts for more than a
    common one, and length normalisation so a long passage does not win
    by containing everything.
  * **Reciprocal rank fusion** combines the two without needing their
    scores to be on comparable scales — which they are not, and which is
    why naive score-averaging of a cosine and a BM25 score produces
    nonsense. RRF uses only the rank each retriever assigned.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Okapi BM25's conventional parameters. k1 controls how fast term
# frequency saturates; b how strongly length is normalised.
BM25_K1 = 1.5
BM25_B = 0.75

# RRF's damping constant. 60 is the value from the original paper and is
# not sensitive: it stops the top rank from dominating so completely that
# the second retriever cannot contribute.
RRF_K = 60

_WORD = re.compile(r"[a-z0-9]+")

_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that",
    "this", "these", "those", "of", "in", "on", "at", "to", "for", "with",
    "without", "from", "by", "as", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "he", "she", "they", "we", "you", "i", "do",
    "does", "did", "have", "has", "had", "will", "would", "can", "could",
    "should", "may", "might", "must", "not", "no", "so", "such", "there",
    "here", "what", "which", "who", "whom", "when", "where", "how", "why",
}


def tokenize(text: str) -> List[str]:
    """Topic-bearing tokens, with plurals folded onto the singular.

    "expense claims" and "expense claim" have to reach the same term, or
    an exact-match retriever misses the one thing it is meant to be good
    at.
    """
    out = []
    for tok in _WORD.findall(str(text).lower()):
        if len(tok) <= 2 or tok in _STOP:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


class BM25Index:
    """Okapi BM25 over a fixed set of documents.

    Small enough to keep in memory and rebuild on ingest; a knowledge
    base here is thousands of chunks, not millions.
    """

    def __init__(self, documents: Sequence[str]):
        self.docs: List[List[str]] = [tokenize(d) for d in documents]
        self.n = len(self.docs)
        self.lengths = np.array([len(d) for d in self.docs], dtype=np.float32)
        self.avg_len = float(self.lengths.mean()) if self.n else 0.0

        # term -> {doc index: count}
        self.postings: Dict[str, Dict[int, int]] = {}
        for i, tokens in enumerate(self.docs):
            for term, count in Counter(tokens).items():
                self.postings.setdefault(term, {})[i] = count

        # Robertson/Sparck-Jones idf with the +0.5 smoothing, floored at
        # a small positive value so a term appearing in most documents
        # contributes nothing rather than a negative score.
        self.idf: Dict[str, float] = {}
        for term, posting in self.postings.items():
            df = len(posting)
            self.idf[term] = max(
                math.log((self.n - df + 0.5) / (df + 0.5) + 1.0), 1e-6)

    def search(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        """(document index, score), best first."""
        if not self.n:
            return []
        scores = np.zeros(self.n, dtype=np.float32)
        for term in tokenize(query):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            for doc_id, freq in posting.items():
                length_norm = (1 - BM25_B
                               + BM25_B * self.lengths[doc_id] / (self.avg_len or 1))
                scores[doc_id] += idf * (freq * (BM25_K1 + 1)) / (
                    freq + BM25_K1 * length_norm)
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


def reciprocal_rank_fusion(rankings: Iterable[Sequence[int]],
                           weights: Optional[Sequence[float]] = None,
                           k: int = RRF_K) -> List[Tuple[int, float]]:
    """Merge several ranked lists into one.

    Each list contributes 1/(k + rank) for every document it ranks, so a
    document ranked well by both retrievers beats one ranked first by
    either alone. Only the ranks are used, which is the point: a cosine
    similarity of 0.71 and a BM25 score of 14.2 are not comparable
    quantities, and averaging them means nothing.
    """
    lists = [list(r) for r in rankings]
    if weights is None:
        weights = [1.0] * len(lists)
    fused: Dict[int, float] = {}
    for ranking, weight in zip(lists, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def dense_search(vectors: Optional[np.ndarray], query_vector: Optional[np.ndarray],
                 k: int = 20) -> List[Tuple[int, float]]:
    """Cosine search over normalised vectors."""
    if vectors is None or query_vector is None or len(vectors) == 0:
        return []
    try:
        sims = vectors @ np.asarray(query_vector).reshape(-1)
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]
    except Exception:
        logger.warning("dense search failed", exc_info=True)
        return []


def hybrid_search(query: str, vectors, query_vector, bm25: Optional[BM25Index],
                  k: int = 6, candidates: int = 30,
                  dense_weight: float = 1.0,
                  sparse_weight: float = 1.0) -> List[Tuple[int, float]]:
    """Dense and sparse retrieval, fused.

    Either half may be missing — no embedder configured, or a knowledge
    base with no text yet — and the fusion degrades to whichever is
    present rather than returning nothing.
    """
    dense = dense_search(vectors, query_vector, k=candidates)
    sparse = bm25.search(query, k=candidates) if bm25 is not None else []

    if dense and not sparse:
        return dense[:k]
    if sparse and not dense:
        return sparse[:k]
    if not dense and not sparse:
        return []

    fused = reciprocal_rank_fusion(
        [[i for i, _ in dense], [i for i, _ in sparse]],
        weights=[dense_weight, sparse_weight])
    return fused[:k]


class CrossEncoderReranker:
    """Reorders a shortlist by reading each passage against the query.

    Retrieval scores a passage without ever looking at it alongside the
    question — the two are embedded separately and compared. A cross
    encoder reads both together, which is much more accurate and far too
    slow to run over a whole corpus. Run over the twenty candidates a
    hybrid search returns, it is affordable and it is what moves the
    right passage from rank six to rank one.

    Optional: when the model is not installed or cannot be loaded, the
    hybrid order is returned unchanged and retrieval still works.
    """

    _model = None
    _tried = False

    @classmethod
    def _load(cls):
        if cls._tried:
            return cls._model
        cls._tried = True
        try:
            from sentence_transformers import CrossEncoder
            from app.config import config
            # Configurable via RERANKER_MODEL. Not a generative model, so
            # it has no place in the task registry — but it does change
            # which passages a client is shown, so it has no place as a
            # literal in a call site either.
            name = config.reranker_model
            cls._model = CrossEncoder(name, max_length=512)
            logger.info("cross-encoder reranker loaded: %s", name)
        except Exception as exc:
            logger.info("cross-encoder unavailable (%s) — retrieval will use "
                        "the hybrid order without reranking", exc)
            cls._model = None
        return cls._model

    @classmethod
    def available(cls) -> bool:
        return cls._load() is not None

    @classmethod
    def rerank(cls, query: str, passages: Sequence[str],
               top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        model = cls._load()
        if model is None or not passages:
            return [(i, 0.0) for i in range(len(passages))][:top_k]
        try:
            scores = model.predict([(query, p) for p in passages])
            order = np.argsort(-np.asarray(scores))
            out = [(int(i), float(scores[i])) for i in order]
            return out[:top_k] if top_k else out
        except Exception:
            logger.warning("reranking failed — keeping the retrieved order",
                           exc_info=True)
            return [(i, 0.0) for i in range(len(passages))][:top_k]
