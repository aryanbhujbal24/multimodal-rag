"""Embedding backends.

Gemini is the real one. The hashing embedder exists so the test suite, CI and a
laptop with no API key can still exercise the entire pipeline end to end --
retrieval quality is meaningless there, but every wire gets tested.
"""
from __future__ import annotations

import hashlib
import re
import time
from abc import ABC, abstractmethod

import numpy as np

from app.config import settings


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        ...

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """L2-normalise so FAISS inner product behaves as cosine similarity."""
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.clip(norms, 1e-9, None)


class HashingEmbedder(Embedder):
    """Deterministic bag-of-words hashing. No network, no model download."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0
        return self.normalize(out)


class GeminiEmbedder(Embedder):
    """text-embedding-004 via the google-genai SDK.

    Documents and queries are embedded with different task types, which is
    worth a real chunk of retrieval accuracy and costs nothing.
    """

    dim = 768

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        from google import genai  # imported lazily: optional dependency

        key = api_key or settings.google_api_key
        if not key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Put it in .env, or run with "
                "EMBEDDER=hashing to use the offline embedder."
            )
        self.client = genai.Client(api_key=key)
        self.model = model or settings.embed_model

    def embed(self, texts: list[str], *, is_query: bool = False,
              batch_size: int = 64) -> np.ndarray:
        from google.genai import types

        task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        vectors: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            for attempt in range(5):
                try:
                    resp = self.client.models.embed_content(
                        model=self.model,
                        contents=batch,
                        config=types.EmbedContentConfig(task_type=task),
                    )
                    vectors.extend(e.values for e in resp.embeddings)
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    # Free tier rate-limits aggressively; back off and retry.
                    time.sleep(2 ** attempt)
        return self.normalize(np.asarray(vectors, dtype="float32"))


def get_embedder() -> Embedder:
    if settings.embedder == "hashing":
        return HashingEmbedder()
    return GeminiEmbedder()
