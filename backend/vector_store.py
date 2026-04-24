"""FAISS vector store — manual persistence (vectors + sidecar metadata)."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .chunking import Chunk


class FaissVectorStore:
    """Inner-product index with L2-normalized vectors (= cosine similarity)."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = query_vec.astype(np.float32).reshape(1, -1)
        sims, ids = self.index.search(q, k)
        return sims[0], ids[0]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        meta = [{"text": c.text, "source_id": c.source_id, "meta": c.meta} for c in self.chunks]
        (directory / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        (directory / "dim.pkl").write_bytes(pickle.dumps(self.dim))

    @classmethod
    def load(cls, directory: Path) -> "FaissVectorStore":
        dim = pickle.loads((directory / "dim.pkl").read_bytes())
        store = cls(dim)
        store.index = faiss.read_index(str(directory / "index.faiss"))
        raw = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        store.chunks = [Chunk(text=r["text"], source_id=r["source_id"], meta=r["meta"]) for r in raw]
        return store
