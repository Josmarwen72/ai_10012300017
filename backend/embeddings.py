"""Embedding pipeline using Sentence-Transformers (manual wiring, no LangChain)."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBED_MODEL_NAME

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    m = get_model()
    vecs = m.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def embed_query(q: str) -> np.ndarray:
    return embed_texts([q])[0]
