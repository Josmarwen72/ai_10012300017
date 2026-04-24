"""End-to-end RAG pipeline with stage logging (Part D)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from .config import INDEX_DIR, PIPELINE_LOGS_PATH, ensure_dirs
from .embeddings import embed_query
from .llm_client import chat_completion_sync
from .prompt_builder import PromptBundle, build_prompt
from .retrieval import (
    BM25Index,
    RetrievedChunk,
    build_bm25_index,
    retrieve_dense_only,
    retrieve_hybrid,
)
from .vector_store import FaissVectorStore

Mode = Literal["rag_hybrid", "rag_dense", "llm_only"]


def _serialize_retrieved(rows: list[RetrievedChunk]) -> list[dict[str, Any]]:
    out = []
    for rc in rows:
        out.append(
            {
                "source_id": rc.chunk.source_id,
                "meta": rc.chunk.meta,
                "dense_score": rc.dense_score,
                "bm25_score": rc.bm25_score,
                "hybrid_score": rc.hybrid_score,
                "rank": rc.rank,
                "text_preview": rc.chunk.text[:400]
                + ("…" if len(rc.chunk.text) > 400 else ""),
            }
        )
    return out


def run_pipeline(
    query: str,
    store: FaissVectorStore,
    bm25: BM25Index | None,
    *,
    mode: Mode = "rag_hybrid",
    prompt_profile: str = "strict",
    use_feedback: bool = True,
) -> dict[str, Any]:
    ensure_dirs()
    t0 = time.perf_counter()
    stages: list[dict[str, Any]] = []

    qvec = None
    if mode != "llm_only":
        qvec = embed_query(query)
        stages.append(
            {
                "stage": "embed_query",
                "ms": round((time.perf_counter() - t0) * 1000, 2),
                "detail": {"dim": int(qvec.shape[0])},
            }
        )
    else:
        stages.append({"stage": "embed_query", "ms": 0, "detail": {"skipped": True}})

    t1 = time.perf_counter()
    retrieved: list[RetrievedChunk] = []
    if mode == "llm_only":
        stages.append({"stage": "retrieve", "ms": 0, "detail": {"skipped": True}})
    elif mode == "rag_dense":
        assert qvec is not None
        retrieved = retrieve_dense_only(store, qvec)
        stages.append(
            {
                "stage": "retrieve_dense",
                "ms": round((time.perf_counter() - t1) * 1000, 2),
                "detail": {"k": len(retrieved)},
            }
        )
    else:
        assert bm25 is not None and qvec is not None
        retrieved = retrieve_hybrid(
            store, bm25, query, qvec, use_feedback=use_feedback
        )
        stages.append(
            {
                "stage": "retrieve_hybrid",
                "ms": round((time.perf_counter() - t1) * 1000, 2),
                "detail": {"k": len(retrieved), "feedback": use_feedback},
            }
        )

    t2 = time.perf_counter()
    if mode == "llm_only":
        bundle = PromptBundle(
            system=(
                "You are a helpful assistant for Academic City. "
                "Answer from general knowledge; no document context is provided."
            ),
            user=f"QUESTION:\n{query.strip()}",
            context_block="",
            truncated=False,
        )
    else:
        bundle = build_prompt(query, retrieved, profile=prompt_profile)
    stages.append(
        {
            "stage": "prompt_build",
            "ms": round((time.perf_counter() - t2) * 1000, 2),
            "detail": {
                "profile": prompt_profile,
                "context_truncated": bundle.truncated,
                "context_chars": len(bundle.context_block),
            },
        }
    )

    t3 = time.perf_counter()
    llm_out = chat_completion_sync(bundle)
    stages.append(
        {
            "stage": "llm",
            "ms": round((time.perf_counter() - t3) * 1000, 2),
            "detail": {"model": llm_out.get("model"), "error": llm_out.get("error")},
        }
    )

    full_prompt_for_ui = f"SYSTEM:\n{bundle.system}\n\nUSER:\n{bundle.user}"
    result = {
        "query": query,
        "mode": mode,
        "prompt_profile": prompt_profile,
        "retrieved": _serialize_retrieved(retrieved),
        "full_prompt": full_prompt_for_ui,
        "answer": llm_out.get("content", ""),
        "llm_error": llm_out.get("error"),
        "stages": stages,
        "total_ms": round((time.perf_counter() - t0) * 1000, 2),
    }

    rec = {
        "ts": time.time(),
        "query": query,
        "mode": mode,
        "prompt_profile": prompt_profile,
        "stages": stages,
        "retrieved_ids": [r["source_id"] for r in result["retrieved"]],
    }
    with PIPELINE_LOGS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return result


def load_store_and_bm25() -> tuple[FaissVectorStore, BM25Index]:
    store = FaissVectorStore.load(INDEX_DIR)
    bm25 = build_bm25_index(store.chunks)
    return store, bm25
