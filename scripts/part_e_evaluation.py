"""Part E evaluation runner: adversarial tests + RAG vs pure LLM evidence.

This script creates evidence files under data/evaluation/ that you can cite in your report.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.pipeline import load_store_and_bm25, run_pipeline


def _hallucination_rate(answer: str, retrieved: list[dict]) -> float:
    """Simple evidence metric: numeric tokens in answer not seen in retrieved previews."""
    import re

    nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", answer or "")
    if not nums:
        return 0.0
    ctx = " ".join(r.get("text_preview", "") for r in retrieved).lower()
    miss = sum(1 for n in nums if n.lower() not in ctx)
    return miss / len(nums)


def main() -> None:
    out_dir = ROOT / "data" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = [
        {
            "name": "adversarial_ambiguous",
            "query": "How did Mahama perform?",
            "category": "ambiguous",
        },
        {
            "name": "adversarial_misleading",
            "query": "What was the NPP vote in Volta Region in 2020?",
            "category": "misleading_or_incomplete",
        },
    ]

    store, bm25 = load_store_and_bm25()
    report: dict[str, object] = {
        "generated_at": time.time(),
        "items": [],
    }

    for q in queries:
        rag = run_pipeline(
            q["query"],
            store,
            bm25,
            mode="rag_hybrid",
            prompt_profile="strict",
            use_feedback=True,
        )
        llm = run_pipeline(
            q["query"],
            store,
            bm25,
            mode="llm_only",
            prompt_profile="strict",
            use_feedback=False,
        )
        rag2 = run_pipeline(
            q["query"],
            store,
            bm25,
            mode="rag_hybrid",
            prompt_profile="strict",
            use_feedback=True,
        )

        rag_consistent = (rag.get("answer", "").strip() == rag2.get("answer", "").strip())
        item = {
            "name": q["name"],
            "category": q["category"],
            "query": q["query"],
            "rag": {
                "answer": rag.get("answer"),
                "llm_error": rag.get("llm_error"),
                "hallucination_rate_est": _hallucination_rate(
                    rag.get("answer", ""), rag.get("retrieved", [])
                ),
                "retrieved_ids": [r.get("source_id") for r in rag.get("retrieved", [])],
            },
            "llm_only": {
                "answer": llm.get("answer"),
                "llm_error": llm.get("llm_error"),
                "hallucination_rate_est": _hallucination_rate(
                    llm.get("answer", ""), llm.get("retrieved", [])
                ),
            },
            "consistency": {
                "rag_run1_equals_run2": rag_consistent,
            },
        }
        report["items"].append(item)

    out_json = out_dir / "part_e_report.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote evaluation report: {out_json}")


if __name__ == "__main__":
    main()

