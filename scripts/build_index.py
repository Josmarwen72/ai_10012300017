"""Build FAISS index from CSV + optional PDF."""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.chunking import iter_all_chunks
from backend.config import DATA_DIR, INDEX_DIR, ensure_dirs
from backend.embeddings import embed_texts
from backend.vector_store import FaissVectorStore

# Same file as: https://github.com/GodwinDansoAcity/acitydataset/blob/main/Ghana_Election_Result.csv
CSV_URL = (
    "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/"
    "Ghana_Election_Result.csv"
)

# Official MOFEP budget PDF (same document as the GitHub-linked course copy)
BUDGET_PDF_URL = (
    "https://mofep.gov.gh/sites/default/files/budget-statements/"
    "2025-Budget-Statement-and-Economic-Policy_v4.pdf"
)
BUDGET_PDF_FILENAME = "2025-Budget-Statement-and-Economic-Policy_v4.pdf"


def main() -> None:
    ensure_dirs()
    csv_path = DATA_DIR / "Ghana_Election_Result.csv"
    if not csv_path.is_file():
        uploads = ROOT.parent / "uploads" / "Ghana_Election_Result-0.csv"
        if uploads.is_file():
            shutil.copy(uploads, csv_path)
        else:
            print(f"Downloading CSV from GitHub -> {csv_path}")
            urllib.request.urlretrieve(CSV_URL, csv_path)

    pdf_candidates = sorted(
        set(DATA_DIR.glob("*.pdf")) | set(DATA_DIR.glob("**/*.pdf"))
    )
    pdf_path: Path | None = pdf_candidates[0] if pdf_candidates else None
    if pdf_path is None:
        dest_pdf = DATA_DIR / BUDGET_PDF_FILENAME
        print(f"Downloading budget PDF from MOFEP -> {dest_pdf}")
        try:
            urllib.request.urlretrieve(BUDGET_PDF_URL, dest_pdf)
            pdf_path = dest_pdf
        except OSError as e:
            raise SystemExit(
                "Could not download the 2025 Budget PDF. Place a copy in data/ manually "
                f"(e.g. from {BUDGET_PDF_URL} or your local file). Error: {e}"
            ) from e

    chunks = iter_all_chunks(csv_path, pdf_path)
    if not chunks:
        raise SystemExit("No chunks produced")

    texts = [c.text for c in chunks]
    vecs = embed_texts(texts)
    dim = vecs.shape[1]
    store = FaissVectorStore(dim)
    store.add(vecs, chunks)
    store.save(INDEX_DIR)
    print(f"Indexed {len(chunks)} chunks -> {INDEX_DIR}")
    print(f"PDF used: {pdf_path}")


if __name__ == "__main__":
    main()
