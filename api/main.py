"""FastAPI service: chat, retrieval diagnostics, manual logs, feedback."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import INDEX_DIR, MANUAL_LOGS_PATH, PIPELINE_LOGS_PATH, ensure_dirs
from backend.embeddings import embed_query
from backend.feedback_store import append_feedback
from backend.pipeline import load_store_and_bm25, run_pipeline
from backend.retrieval import BM25Index
from backend.retrieval import retrieve_dense_only, retrieve_hybrid
from backend.vector_store import FaissVectorStore

app = FastAPI(title="Academic City RAG", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    mode: Literal["rag_hybrid", "rag_dense", "llm_only"] = "rag_hybrid"
    prompt_profile: Literal["strict", "concise", "verbose"] = "strict"
    use_feedback: bool = True


class FeedbackRequest(BaseModel):
    source_id: str
    label: Literal["up", "down"]


class ManualLogRequest(BaseModel):
    entry: str = Field(..., min_length=1, max_length=20000)


_store = None
_bm25 = None


def get_index() -> tuple[FaissVectorStore, BM25Index]:
    global _store, _bm25
    if not INDEX_DIR.joinpath("index.faiss").is_file():
        raise HTTPException(
            status_code=503,
            detail="Index not built. Run: python scripts/build_index.py",
        )
    if _store is None:
        _store, _bm25 = load_store_and_bm25()
    assert _store is not None and _bm25 is not None
    return _store, _bm25


@app.on_event("startup")
def _startup() -> None:
    ensure_dirs()


@app.get("/api/health")
def health() -> dict[str, Any]:
    ok = INDEX_DIR.joinpath("index.faiss").is_file()
    return {"ok": True, "index_ready": ok}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    store, bm25 = get_index()
    bm25_use = bm25 if req.mode == "rag_hybrid" else None
    return run_pipeline(
        req.query,
        store,
        bm25_use,
        mode=req.mode,
        prompt_profile=req.prompt_profile,
        use_feedback=req.use_feedback,
    )


@app.get("/api/retrieve_compare")
def retrieve_compare(q: str = Query(..., min_length=1)) -> dict[str, Any]:
    """Dense vs hybrid side-by-side (Part B failure-case evidence)."""
    store, bm25 = get_index()
    qv = embed_query(q)
    dense = retrieve_dense_only(store, qv, k=8)
    hyb = retrieve_hybrid(store, bm25, q, qv, k=8, use_feedback=False)
    return {
        "query": q,
        "dense_top": [r.chunk.source_id for r in dense],
        "hybrid_top": [r.chunk.source_id for r in hyb],
        "dense_scores": [r.dense_score for r in dense],
        "hybrid_scores": [r.hybrid_score for r in hyb],
    }


@app.post("/api/compare_rag_vs_llm")
def compare_rag_vs_llm(req: ChatRequest) -> dict[str, Any]:
    store, bm25 = get_index()
    rag = run_pipeline(
        req.query,
        store,
        bm25,
        mode="rag_hybrid",
        prompt_profile=req.prompt_profile,
        use_feedback=req.use_feedback,
    )
    base = run_pipeline(
        req.query,
        store,
        bm25,
        mode="llm_only",
        prompt_profile=req.prompt_profile,
        use_feedback=False,
    )
    return {"rag": rag, "llm_only": base}


@app.post("/api/feedback")
def feedback(req: FeedbackRequest) -> dict[str, str]:
    append_feedback(req.source_id, req.label)
    return {"status": "ok"}


@app.post("/api/logs/manual")
def manual_log_append(req: ManualLogRequest) -> dict[str, str]:
    ensure_dirs()
    import time

    rec = {"ts": time.time(), "entry": req.entry}
    with MANUAL_LOGS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"status": "ok"}


@app.get("/api/logs/manual")
def manual_log_list(limit: int = 200) -> list[dict[str, Any]]:
    if not MANUAL_LOGS_PATH.is_file():
        return []
    lines = MANUAL_LOGS_PATH.read_text(encoding="utf-8").splitlines()[-limit:]
    out: list[dict[str, Any]] = []
    for ln in lines:
        if ln.strip():
            out.append(json.loads(ln))
    return out


@app.get("/api/logs/pipeline")
def pipeline_logs(limit: int = 50) -> list[dict[str, Any]]:
    if not PIPELINE_LOGS_PATH.is_file():
        return []
    lines = PIPELINE_LOGS_PATH.read_text(encoding="utf-8").splitlines()[-limit:]
    return [json.loads(ln) for ln in lines if ln.strip()]

