"""
Chunking strategies (Part A) — implemented without retrieval frameworks.

Design:
- CSV: region-year aggregated chunks keep candidate rows together (semantic unit for Q&A).
- PDF: sliding character windows with overlap preserve sentence boundaries where possible.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
from itertools import groupby
from pathlib import Path
from typing import Any

from . import cleaning
from .config import CSV_CHUNK_MAX_ROWS, PDF_CHUNK_CHARS, PDF_CHUNK_OVERLAP


@dataclass
class Chunk:
    text: str
    source_id: str
    meta: dict[str, Any]


def _strip_csv_preamble(raw_csv: str) -> str:
    lines = raw_csv.splitlines()
    for i, line in enumerate(lines):
        s = line.lstrip("\ufeff").strip()
        if s.startswith("Year,") or s.startswith('"Year",'):
            return "\n".join(lines[i:])
    return raw_csv


def load_csv_text_rows(raw_csv: str) -> list[dict[str, str]]:
    raw_csv = _strip_csv_preamble(raw_csv)
    reader = csv.DictReader(StringIO(raw_csv))
    rows: list[dict[str, str]] = []
    for r in reader:
        cleaned = cleaning.clean_csv_row(r)
        if cleaned:
            rows.append(cleaned)  # type: ignore[arg-type]
    return rows


def chunk_csv_by_region_year(
    rows: list[dict[str, str]], max_rows_per_chunk: int = CSV_CHUNK_MAX_ROWS
) -> list[Chunk]:
    """Group rows with same (Year, New Region); split long lists into sub-chunks."""
    rows_sorted = sorted(
        rows,
        key=lambda x: (int(x["Year"]), x.get("New Region", ""), x.get("Party", "")),
    )
    chunks: list[Chunk] = []
    chunk_idx = 0
    for (year, region), grp in groupby(
        rows_sorted, key=lambda x: (x["Year"], x["New Region"])
    ):
        block = list(grp)
        for offset in range(0, len(block), max_rows_per_chunk):
            sub = block[offset : offset + max_rows_per_chunk]
            lines = [
                f"Ghana {b['Year']} presidential election results: {b['Candidate']} from {b['Party']} party "
                f"received {b['Votes']} votes ({b['Votes(%)']} percentage) in {b['New Region']} region. "
                f"Election candidate {b['Candidate']} performance in {b['New Region']} with vote count {b['Votes']}."
                for b in sub
            ]
            text = "Ghana presidential election data:\n" + "\n".join(lines)
            cid = f"csv:{year}:{region}:{chunk_idx}"
            chunks.append(
                Chunk(
                    text=text,
                    source_id=cid,
                    meta={"type": "election_csv", "year": year, "region": region},
                )
            )
            chunk_idx += 1
    return chunks


def chunk_pdf_sliding(text: str, source_label: str) -> list[Chunk]:
    """Character-based chunks with overlap; split preferentially on paragraph breaks."""
    t = cleaning.clean_pdf_text(text)
    chunks: list[Chunk] = []
    start = 0
    n = len(t)
    idx = 0
    while start < n:
        end = min(start + PDF_CHUNK_CHARS, n)
        window = t[start:end]
        if end < n:
            cut = window.rfind("\n\n")
            if cut > PDF_CHUNK_CHARS // 3:
                window = window[:cut]
                end = start + cut
        chunk_text = window.strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    text=f"Budget / policy excerpt ({source_label}):\n{chunk_text}",
                    source_id=f"pdf:{source_label}:{idx}",
                    meta={"type": "budget_pdf", "pdf": source_label},
                )
            )
            idx += 1
        if end >= n:
            break
        start = max(end - PDF_CHUNK_OVERLAP, start + 1)
    return chunks


def chunk_pdf_alternate_small(text: str, source_label: str) -> list[Chunk]:
    """Smaller chunks for comparative analysis (Part A deliverable)."""
    small = 400
    overlap = 60
    t = cleaning.clean_pdf_text(text)
    chunks: list[Chunk] = []
    start = 0
    n = len(t)
    idx = 0
    while start < n:
        end = min(start + small, n)
        window = t[start:end].strip()
        if window:
            chunks.append(
                Chunk(
                    text=f"Budget / policy excerpt ({source_label}):\n{window}",
                    source_id=f"pdf_small:{source_label}:{idx}",
                    meta={"type": "budget_pdf", "pdf": source_label, "variant": "small"},
                )
            )
            idx += 1
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def iter_all_chunks(csv_path: Path, pdf_path: Path | None) -> list[Chunk]:
    raw = csv_path.read_text(encoding="utf-8", errors="replace")
    rows = load_csv_text_rows(raw)
    out = chunk_csv_by_region_year(rows)
    if pdf_path and pdf_path.is_file():
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError("Install pypdf to index PDFs") from e
        reader = PdfReader(str(pdf_path))
        parts: list[str] = []
        for p in reader.pages:
            parts.append(p.extract_text() or "")
        full = "\n".join(parts)
        label = pdf_path.stem
        out.extend(chunk_pdf_sliding(full, label))
    return out
