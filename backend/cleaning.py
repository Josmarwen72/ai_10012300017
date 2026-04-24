"""Data cleaning for CSV election rows and extracted PDF text (Part A)."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_whitespace(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\xa0", " ").replace("\u200b", "")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def clean_csv_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return cleaned row or None if invalid."""
    out: dict[str, str] = {}
    for k, v in row.items():
        key = normalize_whitespace(str(k))
        if v is None:
            return None
        out[key] = normalize_whitespace(str(v))
    required = {"Year", "New Region", "Candidate", "Party", "Votes", "Votes(%)"}
    if not required.issubset(set(out.keys())):
        return None
    year = out["Year"]
    if not year.isdigit():
        return None
    votes = out["Votes"].replace(",", "")
    if not votes.isdigit():
        return None
    pct = out["Votes(%)"]
    if not re.match(r"^\d+(\.\d+)?%?$", pct.replace("%", "") + ("" if "%" in pct else "")):
        return None
    if "%" not in pct:
        pct = pct + "%"
    out["Votes"] = votes
    out["Votes(%)"] = pct
    return out


def clean_pdf_text(text: str) -> str:
    t = normalize_whitespace(text)
    t = re.sub(r"-\s*\n", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t
