"""FAISS store holding two parallel indexes: text passages and figures.

Figures are kept separate rather than mixed into one index for a practical
reason -- a question like "what does the architecture diagram show" should be
able to surface a figure even when five text passages score higher. One shared
index makes figures lose almost every time, because captions are short and text
passages are long.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings
from app.index.embedder import Embedder


@dataclass
class Hit:
    score: float
    payload: dict
    kind: str  # "text" | "figure"


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(
        c.lower() if c.isalnum() else " " for c in text
    ).split() if len(t) > 1]


class VectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.text_index: faiss.Index | None = None
        self.figure_index: faiss.Index | None = None
        self.text_meta: list[dict] = []
        self.figure_meta: list[dict] = []
        self.bm25: BM25Okapi | None = None

    # ------------------------------------------------------------------
    def build(self, chunks: list[dict], figures: list[dict]) -> None:
        if chunks:
            vectors = self.embedder.embed([c["text"] for c in chunks])
            self.text_index = faiss.IndexFlatIP(vectors.shape[1])
            self.text_index.add(vectors)
            self.text_meta = chunks
            self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])

        if figures:
            texts = [f.get("searchable_text") or f.get("caption", "") for f in figures]
            vectors = self.embedder.embed(texts)
            self.figure_index = faiss.IndexFlatIP(vectors.shape[1])
            self.figure_index.add(vectors)
            self.figure_meta = figures

    # ------------------------------------------------------------------
    def _dense(self, index: faiss.Index | None, meta: list[dict],
               query_vec: np.ndarray, k: int, kind: str) -> list[Hit]:
        if index is None or index.ntotal == 0:
            return []
        scores, ids = index.search(query_vec, min(k, index.ntotal))
        return [
            Hit(score=float(s), payload=meta[i], kind=kind)
            for s, i in zip(scores[0], ids[0]) if i != -1
        ]

    def search_text(self, query: str, k: int | None = None,
                    hybrid: bool | None = None) -> list[Hit]:
        """Dense search, optionally fused with BM25 by reciprocal rank fusion.

        RRF is used instead of score averaging because cosine scores and BM25
        scores are on incomparable scales; ranks are all that can be combined
        honestly without tuning a weight per corpus.
        """
        # `k or default` would silently turn an explicit 0 into the default.
        k = settings.top_k if k is None else k
        if k <= 0:
            return []
        hybrid = settings.use_hybrid if hybrid is None else hybrid
        qv = self.embedder.embed([query], is_query=True)
        dense = self._dense(self.text_index, self.text_meta, qv, k * 3, "text")

        if not hybrid or self.bm25 is None:
            return dense[:k]

        lexical_scores = self.bm25.get_scores(_tokenize(query))
        lexical_order = np.argsort(lexical_scores)[::-1][: k * 3]

        fused: dict[int, float] = {}
        rrf_k = settings.rrf_k
        for rank, hit in enumerate(dense):
            idx = hit.payload["_row"]  # row id stamped on every chunk at build
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(lexical_order):
            if lexical_scores[idx] <= 0:
                continue
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (rrf_k + rank + 1)

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Hit(score=s, payload=self.text_meta[i], kind="text") for i, s in ordered]

    def search_figures(self, query: str, k: int | None = None) -> list[Hit]:
        k = settings.top_k_figures if k is None else k
        if k <= 0 or self.figure_index is None:
            return []
        qv = self.embedder.embed([query], is_query=True)
        return self._dense(self.figure_index, self.figure_meta, qv, k, "figure")

    # ------------------------------------------------------------------
    def save(self, path: Path | None = None) -> None:
        path = Path(path or settings.index_dir)
        path.mkdir(parents=True, exist_ok=True)
        if self.text_index is not None:
            faiss.write_index(self.text_index, str(path / "text.faiss"))
        if self.figure_index is not None:
            faiss.write_index(self.figure_index, str(path / "figures.faiss"))
        (path / "text_meta.json").write_text(json.dumps(self.text_meta))
        (path / "figure_meta.json").write_text(json.dumps(self.figure_meta))
        if self.bm25 is not None:
            (path / "bm25.pkl").write_bytes(pickle.dumps(self.bm25))

    @classmethod
    def load(cls, embedder: Embedder, path: Path | None = None) -> "VectorStore":
        path = Path(path or settings.index_dir)
        store = cls(embedder)
        if (path / "text.faiss").exists():
            store.text_index = faiss.read_index(str(path / "text.faiss"))
        if (path / "figures.faiss").exists():
            store.figure_index = faiss.read_index(str(path / "figures.faiss"))
        if (path / "text_meta.json").exists():
            store.text_meta = json.loads((path / "text_meta.json").read_text())
        if (path / "figure_meta.json").exists():
            store.figure_meta = json.loads((path / "figure_meta.json").read_text())
        if (path / "bm25.pkl").exists():
            store.bm25 = pickle.loads((path / "bm25.pkl").read_bytes())
        return store

    @property
    def is_empty(self) -> bool:
        return self.text_index is None and self.figure_index is None
