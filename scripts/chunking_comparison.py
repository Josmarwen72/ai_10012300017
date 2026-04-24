"""
Comparative chunking analysis (Part A): dense retrieval stability vs chunk size on PDF.

Run after placing a budget PDF in data/. Prints mean top-1 dense score over sample queries.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.chunking import chunk_pdf_alternate_small, chunk_pdf_sliding
from backend.cleaning import clean_pdf_text
from backend.embeddings import embed_texts, embed_query
from backend.vector_store import FaissVectorStore


def index_chunks(chunks):
    texts = [c.text for c in chunks]
    vecs = embed_texts(texts)
    store = FaissVectorStore(vecs.shape[1])
    store.add(vecs, chunks)
    return store


def main() -> None:
    from pypdf import PdfReader

    from backend.config import DATA_DIR

    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    pdf = pdfs[0] if pdfs else None
    if not pdf or not pdf.is_file():
        print("No PDF in data/ — skip or add PDF for PDF chunking comparison.")
        return
    reader = PdfReader(str(pdf))
    full = "\n".join((p.extract_text() or "") for p in reader.pages)
    full = clean_pdf_text(full)
    label = pdf.stem
    large = chunk_pdf_sliding(full, label)
    small = chunk_pdf_alternate_small(full, label)
    sl = index_chunks(large)
    ss = index_chunks(small)
    queries = [
        "What are fiscal targets for Ghana?",
        "revenue measures budget",
        "debt sustainability",
        "capital expenditure",
    ]
    print(f"Large chunks: {len(large)} | Small chunks: {len(small)}")
    for q in queries:
        qv = embed_query(q)
        s1, _ = sl.search(qv, 1)
        s2, _ = ss.search(qv, 1)
        print(f"Q: {q[:50]}... | top1 large={float(s1[0]):.4f} small={float(s2[0]):.4f}")


if __name__ == "__main__":
    main()
