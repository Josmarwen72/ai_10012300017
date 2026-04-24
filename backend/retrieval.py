"""
Custom retrieval: dense top-k + BM25 hybrid + optional feedback re-ranking (Part B).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .bm25 import BM25Index, tokenize
from .chunking import Chunk
from .config import HYBRID_ALPHA, TOP_K
from .feedback_store import load_feedback_weights
from .vector_store import FaissVectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    dense_score: float
    bm25_score: float
    hybrid_score: float
    rank: int


def minmax_norm(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return np.ones_like(scores)
    return (scores - lo) / (hi - lo)


def retrieve_hybrid(
    store: FaissVectorStore,
    bm25: BM25Index,
    query: str,
    query_vec: np.ndarray,
    k: int = TOP_K,
    alpha: float = HYBRID_ALPHA,
    use_feedback: bool = True,
) -> list[RetrievedChunk]:
    """Dense top-k ∪ BM25 top-k, fuse scores, optional domain feedback boost."""
    q_tokens = tokenize(query)
    n = len(store.chunks)

    dense_k = min(k * 4, n)
    sims, ids = store.search(query_vec, dense_k)
    cand: set[int] = set(int(i) for i in ids if i >= 0)

    bm25_scores_full = np.array([bm25.score(q_tokens, i) for i in range(n)], dtype=np.float64)
    top_bm25_idx = np.argsort(-bm25_scores_full)[:dense_k]
    cand.update(int(i) for i in top_bm25_idx)

    cand_list = sorted(cand)
    if not cand_list:
        return []

    dense_map = {int(i): float(s) for i, s in zip(ids, sims)}
    d_vec = np.array([dense_map.get(i, 0.0) for i in cand_list], dtype=np.float64)
    b_vec = np.array([bm25_scores_full[i] for i in cand_list], dtype=np.float64)
    d_n = minmax_norm(d_vec)
    b_n = minmax_norm(b_vec)
    hybrid = alpha * d_n + (1 - alpha) * b_n

    if use_feedback:
        wmap = load_feedback_weights()
        for j, doc_i in enumerate(cand_list):
            sid = store.chunks[doc_i].source_id
            w = wmap.get(sid, 0.0)
            hybrid[j] = hybrid[j] * (1.0 + w)

    order = np.argsort(-hybrid)[:k]
    out: list[RetrievedChunk] = []
    for rank, j in enumerate(order):
        doc_i = cand_list[int(j)]
        out.append(
            RetrievedChunk(
                chunk=store.chunks[doc_i],
                dense_score=float(d_vec[j]),
                bm25_score=float(b_vec[j]),
                hybrid_score=float(hybrid[j]),
                rank=rank,
            )
        )
    return out


def retrieve_dense_only(
    store: FaissVectorStore,
    query_vec: np.ndarray,
    k: int = TOP_K,
) -> list[RetrievedChunk]:
    sims, ids = store.search(query_vec, k)
    out: list[RetrievedChunk] = []
    for rank, (s, i) in enumerate(zip(sims, ids)):
        if i < 0:
            continue
        c = store.chunks[int(i)]
        out.append(
            RetrievedChunk(
                chunk=c,
                dense_score=float(s),
                bm25_score=0.0,
                hybrid_score=float(s),
                rank=rank,
            )
        )
    return out


def build_bm25_index(chunks: Sequence[Chunk]) -> BM25Index:
    docs = [tokenize(c.text) for c in chunks]
    return BM25Index(docs)
