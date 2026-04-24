"""
Innovation (Part G): human feedback adjusts chunk weights for hybrid retrieval.

Thumbs-up/down stored in JSONL; running sum maps source_id -> weight delta.
"""

from __future__ import annotations

import json

from .config import FEEDBACK_PATH, ensure_dirs


def append_feedback(source_id: str, label: str) -> None:
    ensure_dirs()
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {"source_id": source_id, "label": label}
    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_feedback_weights() -> dict[str, float]:
    if not FEEDBACK_PATH.is_file():
        return {}
    weights: dict[str, float] = {}
    with FEEDBACK_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec["source_id"]
            label = rec.get("label", "")
            delta = 0.15 if label == "up" else -0.2 if label == "down" else 0.0
            weights[sid] = weights.get(sid, 0.0) + delta
    return weights
