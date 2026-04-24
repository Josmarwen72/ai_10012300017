"""Prompt templates, hallucination guardrails, context budgeting (Part C)."""

from __future__ import annotations

from dataclasses import dataclass

from .config import MAX_CONTEXT_CHARS
from .retrieval import RetrievedChunk


@dataclass
class PromptBundle:
    system: str
    user: str
    context_block: str
    truncated: bool


PROFILES: dict[str, dict[str, str]] = {
    "strict": {
        "system": (
            "You are Academic City's research assistant. Answer ONLY using the CONTEXT. "
            "If the answer is not contained in the CONTEXT, say you do not have that "
            "information in the indexed documents. Cite chunk hints (region/year or "
            "source_id) when stating numbers. Do not invent statistics."
        ),
        "suffix": "Remember: if CONTEXT is silent, refuse to speculate.",
    },
    "concise": {
        "system": (
            "You are Academic City's assistant. Use CONTEXT when relevant. Keep answers "
            "short (3-6 sentences). If CONTEXT conflicts with general knowledge, prefer CONTEXT."
        ),
        "suffix": "",
    },
    "verbose": {
        "system": (
            "You are Academic City's assistant. Explain reasoning step-by-step using CONTEXT. "
            "Quote short phrases from CONTEXT where helpful. If information is missing, list "
            "what is missing before any general discussion."
        ),
        "suffix": "End with a brief 'Confidence' line: high/medium/low based on CONTEXT coverage.",
    },
}


def assemble_context(
    retrieved: list[RetrievedChunk], max_chars: int = MAX_CONTEXT_CHARS
) -> tuple[str, bool]:
    """Rank is already by hybrid score; pack chunks until char budget."""
    parts: list[str] = []
    used = 0
    truncated = False
    for rc in retrieved:
        header = f"[{rc.chunk.source_id}] (score={rc.hybrid_score:.4f})\n"
        block = header + rc.chunk.text.strip() + "\n\n"
        if used + len(block) > max_chars:
            truncated = True
            remain = max_chars - used - len(header) - 10
            if remain > 80:
                parts.append(header + rc.chunk.text[:remain].strip() + "…\n\n")
                used = max_chars
            break
        parts.append(block)
        used += len(block)
    return "".join(parts).strip(), truncated


def build_prompt(
    query: str,
    retrieved: list[RetrievedChunk],
    profile: str = "strict",
) -> PromptBundle:
    prof = PROFILES.get(profile, PROFILES["strict"])
    ctx, truncated = assemble_context(retrieved)
    context_block = ctx or "(no context retrieved)"
    user = (
        f"QUESTION:\n{query.strip()}\n\n"
        f"CONTEXT:\n{context_block}\n"
        f"{prof['suffix']}\n"
    )
    return PromptBundle(
        system=prof["system"],
        user=user,
        context_block=context_block,
        truncated=truncated,
    )
