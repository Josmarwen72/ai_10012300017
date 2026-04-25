from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

from flask import Flask, redirect, render_template, request, url_for, jsonify

_FLASK_DIR = Path(__file__).resolve().parent
ROOT = _FLASK_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import MANUAL_LOGS_PATH, ensure_dirs

# Try to import ML dependencies, use fallback if not available
try:
    from backend.feedback_store import append_feedback
    from backend.pipeline import load_store_and_bm25, run_pipeline
    ML_AVAILABLE = True
except ImportError as e:
    print(f"ML dependencies not available: {e}")
    ML_AVAILABLE = False

# Fix the DATA_DIR to point to the correct data directory
DATA_DIR = _FLASK_DIR.parent / "data"
INDEX_DIR = DATA_DIR / "index"

Mode = Literal["rag_hybrid", "rag_dense", "llm_only"]

app = Flask(
    __name__,
    template_folder=str(_FLASK_DIR / "templates"),
)
app.secret_key = "acity_rag_feedback_secret_key_2026"

_store: tuple[Any, Any] | None = None

# Dynamic log storage system
_experiment_logs: list[dict[str, Any]] = []
_query_history: list[dict[str, Any]] = []
_next_log_id: int = 1
_next_query_id: int = 1


def get_index():
    global _store
    if not INDEX_DIR.joinpath("index.faiss").is_file():
        return None, None
    
    # Load the index directly using the correct INDEX_DIR
    from backend.vector_store import FaissVectorStore
    from backend.bm25 import BM25Index, tokenize
    
    store = FaissVectorStore.load(INDEX_DIR)
    bm25 = BM25Index([tokenize(c.text) for c in store.chunks])
    _store = (store, bm25)
    
    return _store


def _mode_options(current: str) -> list[dict[str, str]]:
    modes: list[tuple[str, str]] = [
        ("rag_hybrid", "RAG hybrid"),
        ("rag_dense", "RAG dense"),
        ("llm_only", "LLM only"),
    ]
    return [
        {"value": v, "label": lab, "selected": "selected" if v == current else ""}
        for v, lab in modes
    ]


def _profile_options(current: str) -> list[dict[str, str]]:
    profiles: list[tuple[str, str]] = [
        ("strict", "strict"),
        ("concise", "concise"),
        ("verbose", "verbose"),
    ]
    return [
        {"value": v, "label": lab, "selected": "selected" if v == current else ""}
        for v, lab in profiles
    ]


def _enrich_result(result: dict[str, Any]) -> dict[str, Any]:
    """Pre-format numbers so the template stays plain HTML-friendly."""
    rows = []
    for c in result.get("retrieved", []):
        rows.append(
            {
                "source_id": c["source_id"],
                "text_preview": c["text_preview"],
                "scores_line": (
                    f"d={c['dense_score']:.3f} | b={c['bm25_score']:.3f} | "
                    f"h={c['hybrid_score']:.3f}"
                ),
            }
        )
    out = dict(result)
    out["retrieved_rows"] = rows
    return out


def _load_manual_logs(limit: int = 40) -> list[dict[str, Any]]:
    """Load logs from in-memory storage (dynamic)"""
    return list(reversed(_experiment_logs[-limit:]))


def _add_experiment_log(query: str, mode: str, status: str, response: str = "", observation: str = "") -> int:
    """Add a new experiment log entry"""
    global _next_log_id, _experiment_logs
    log_entry = {
        "id": _next_log_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "query": query,
        "mode": mode,
        "status": status,
        "response": response,
        "observation": observation,
    }
    _experiment_logs.append(log_entry)
    _next_log_id += 1
    return log_entry["id"]


def _delete_experiment_log(log_id: int) -> bool:
    """Delete an experiment log by ID"""
    global _experiment_logs
    original_length = len(_experiment_logs)
    _experiment_logs = [log for log in _experiment_logs if log.get("id") != log_id]
    return len(_experiment_logs) < original_length


def _add_query_history(query: str, mode: str) -> int:
    """Add a query to history"""
    global _next_query_id, _query_history
    query_entry = {
        "id": _next_query_id,
        "query": query,
        "mode": mode,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    _query_history.append(query_entry)
    _next_query_id += 1
    # Keep only last 20 queries
    if len(_query_history) > 20:
        _query_history = _query_history[-20:]
    return query_entry["id"]


def _delete_query_history(query_id: int) -> bool:
    """Delete a query from history"""
    global _query_history
    original_length = len(_query_history)
    _query_history = [q for q in _query_history if q.get("id") != query_id]
    return len(_query_history) < original_length


def _clear_all_results() -> None:
    """Clear all stored query results and comparisons"""
    global _experiment_logs, _query_history
    _experiment_logs = []
    _query_history = []


def _get_query_history(limit: int = 10) -> list[dict[str, Any]]:
    """Get recent query history"""
    return list(reversed(_query_history[-limit:]))


@app.route("/", methods=["GET", "POST"])
def home() -> str:
    ensure_dirs()
    result: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    
    # Clear results if there's no query history (fresh start)
    if len(_query_history) == 0:
        result = None
        comparison = None
    error = ""
    query = ""
    mode: Mode = "rag_hybrid"
    prompt_profile = "strict"
    use_feedback = True

    store, bm25 = get_index()
    index_ready = store is not None

    if request.method == "POST":
        action = (request.form.get("action") or "run").strip()
        query = (request.form.get("query") or "").strip()
        mode = (request.form.get("mode") or "rag_hybrid").strip()  # type: ignore[assignment]
        prompt_profile = (request.form.get("prompt_profile") or "strict").strip()
        use_feedback = request.form.get("use_feedback") == "on"

        if not ML_AVAILABLE:
            error = "ML dependencies not available. Deploy with full ML stack for RAG functionality."
        elif not index_ready:
            error = "Index not built. Run: python scripts/build_index.py"
        elif not query:
            error = "Please type a query."
        else:
            if action == "compare":
                # Part E requirement: evidence-based RAG vs pure LLM comparison.
                rag_raw = run_pipeline(
                    query,
                    store,
                    bm25,
                    mode="rag_hybrid",
                    prompt_profile=prompt_profile,
                    use_feedback=use_feedback,
                )
                llm_raw = run_pipeline(
                    query,
                    store,
                    bm25,
                    mode="llm_only",
                    prompt_profile=prompt_profile,
                    use_feedback=False,
                )
                result = _enrich_result(rag_raw)
                comparison = {
                    "rag_answer": rag_raw.get("answer", ""),
                    "rag_model": rag_raw.get("stages", [{}])[-1].get("detail", {}).get("model", ""),
                    "llm_answer": llm_raw.get("answer", ""),
                    "llm_model": llm_raw.get("stages", [{}])[-1].get("detail", {}).get("model", ""),
                    "llm_error": llm_raw.get("llm_error"),
                }
            else:
                bm25_use = bm25 if mode == "rag_hybrid" else None
                raw = run_pipeline(
                    query,
                    store,
                    bm25_use,
                    mode=mode,
                    prompt_profile=prompt_profile,
                    use_feedback=use_feedback,
                )
                result = _enrich_result(raw)
                
                # Add to query history
                if query and not error:
                    _add_query_history(query, mode)
                
                # Create experiment log entry
                if result and not error:
                    status = "Success"
                    if result.get("llm_error"):
                        status = "Failed"
                    elif not result.get("answer"):
                        status = "Partial"
                    
                    _add_experiment_log(
                        query=query,
                        mode=mode,
                        status=status,
                        response=result.get("answer", ""),
                        observation=""
                    )

    # Get and clear feedback message from session
    from flask import session
    feedback_message = session.pop('feedback_message', None)
    
    return render_template(
        "index.html",
        index_ready=index_ready,
        result=result,
        error=error,
        query=query,
        comparison=comparison,
        mode_options=_mode_options(mode),
        profile_options=_profile_options(prompt_profile),
        use_feedback=use_feedback,
        logs=_load_manual_logs(),
        query_history=_get_query_history(),
        feedback_message=feedback_message,
    )


@app.post("/feedback")
def feedback() -> Any:
    source_id = (request.form.get("source_id") or "").strip()
    label = (request.form.get("label") or "").strip()
    if source_id and label in {"up", "down"}:
        if ML_AVAILABLE:
            append_feedback(source_id, label)
            # Store success message in session for display
            from flask import session
            session['feedback_message'] = f"Feedback recorded: {label} for {source_id}"
        else:
            from flask import session
            session['feedback_message'] = "ML dependencies not available - feedback not recorded"
    return redirect(url_for("home"))


@app.post("/manual-log")
def manual_log() -> Any:
    entry = (request.form.get("entry") or "").strip()
    if entry:
        ensure_dirs()
        with MANUAL_LOGS_PATH.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps({"ts": time.time(), "entry": entry}, ensure_ascii=False) + "\n"
            )
    return redirect(url_for("home"))


@app.post("/api/delete-log/<int:log_id>")
def delete_log(log_id: int) -> Any:
    """API endpoint to delete experiment log"""
    success = _delete_experiment_log(log_id)
    return jsonify({"success": success, "log_id": log_id})


@app.post("/api/delete-query/<int:query_id>")
def delete_query(query_id: int) -> Any:
    """API endpoint to delete query from history"""
    success = _delete_query_history(query_id)
    return jsonify({"success": success, "query_id": query_id})


@app.post("/api/new-experiment-log")
def new_experiment_log() -> Any:
    """API endpoint to create new experiment log"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"})
    
    query = data.get("query", "").strip()
    mode = data.get("mode", "rag_hybrid")
    status = data.get("status", "Success")
    response = data.get("response", "")
    observation = data.get("observation", "")
    
    if not query:
        return jsonify({"success": False, "error": "Query is required"})
    
    log_id = _add_experiment_log(query, mode, status, response, observation)
    return jsonify({"success": True, "log_id": log_id})


@app.get("/api/logs")
def get_logs() -> Any:
    """API endpoint to get current experiment logs"""
    return jsonify({"logs": _load_manual_logs()})


@app.get("/api/query-history")
def get_query_history_api() -> Any:
    """API endpoint to get current query history"""
    return jsonify({"query_history": _get_query_history()})


@app.post("/api/clear-all")
def clear_all_data() -> Any:
    """API endpoint to clear all query history and experiment logs"""
    _clear_all_results()
    # Clear feedback message from session when clearing all data
    from flask import session
    session.pop('feedback_message', None)
    return jsonify({"success": True, "message": "All data cleared successfully"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8501, debug=True)
