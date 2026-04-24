"""Local LLM fallback using Transformers (CPU-friendly).

Used only when OPENAI_API_KEY is not set so the app still generates answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import LOCAL_LLM_MAX_NEW_TOKENS, LOCAL_LLM_MODEL
from .prompt_builder import PromptBundle


@dataclass
class _LocalModel:
    name: str
    tok: Any
    model: Any


_cached: _LocalModel | None = None


def _get_local() -> _LocalModel:
    global _cached
    if _cached is not None:
        return _cached

    tok = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_LLM_MODEL)
    model.eval()
    _cached = _LocalModel(name=LOCAL_LLM_MODEL, tok=tok, model=model)
    return _cached


def local_generate(bundle: PromptBundle) -> dict[str, Any]:
    """Generate with a seq2seq instruction model (e.g., FLAN-T5).

    We convert chat-style bundle into a single instruction prompt.
    """
    lm = _get_local()
    instruction = (
        "You are Academic City's assistant.\n\n"
        "Follow the SYSTEM rules and answer the QUESTION.\n"
        "Use the CONTEXT if provided. If CONTEXT is missing or does not contain the answer, say so.\n\n"
        f"SYSTEM:\n{bundle.system}\n\n"
        f"{bundle.user}\n\n"
        "ANSWER:"
    )

    inputs = lm.tok(
        instruction,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    with torch.no_grad():
        out_ids = lm.model.generate(
            **inputs,
            max_new_tokens=LOCAL_LLM_MAX_NEW_TOKENS,
            do_sample=False,
        )
    text = lm.tok.decode(out_ids[0], skip_special_tokens=True).strip()
    return {"error": None, "model": f"local:{lm.name}", "content": text}

