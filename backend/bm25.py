"""Manual BM25 scoring (hybrid search keyword leg — no retrieval framework)."""

from __future__ import annotations

import math
import re
from collections import Counter


_token_re = re.compile(r"[a-z0-9]+", re.I)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _token_re.findall(text)]


class BM25Index:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.N = len(docs)
        self.doc_lens = [len(d) for d in docs]
        self.avgdl = sum(self.doc_lens) / self.N if self.N else 0.0
        self.df: dict[str, int] = Counter()
        for d in docs:
            for t in set(d):
                self.df[t] += 1
        self.idf: dict[str, float] = {}
        for t, df in self.df.items():
            self.idf[t] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, q_tokens: list[str], doc_idx: int) -> float:
        d = self.docs[doc_idx]
        if not d:
            return 0.0
        freqs = Counter(d)
        dl = self.doc_lens[doc_idx]
        K = self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
        s = 0.0
        for t in q_tokens:
            if t not in freqs:
                continue
            f = freqs[t]
            idf = self.idf.get(t, 0.0)
            s += idf * (f * (self.k1 + 1)) / (f + K)
        return s
