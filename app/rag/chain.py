"""The assembled RAG pipeline: one object the API and the UI both call."""
from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.index.embedder import get_embedder
from app.index.store import VectorStore
from app.rag.generator import Answer, EchoGenerator, Generator
from app.rag.retriever import MultimodalRetriever


class RAGChain:
    def __init__(self, store: VectorStore | None = None, generator=None,
                 retriever: MultimodalRetriever | None = None) -> None:
        self.store = store or VectorStore.load(get_embedder())
        self.retriever = retriever or MultimodalRetriever(store=self.store)
        self.generator = generator or (
            EchoGenerator() if not settings.google_api_key else Generator()
        )

    def ask(self, question: str, *, k: int | None = None,
            k_figures: int | None = None) -> Answer:
        retriever = self.retriever
        if k is not None or k_figures is not None:
            retriever = self.retriever.with_overrides(
                k=k or self.retriever.k,
                k_figures=k_figures if k_figures is not None else self.retriever.k_figures,
            )
        text_docs, figure_docs = retriever.retrieve_split(question)
        return self.generator.answer(question, text_docs, figure_docs)

    @property
    def stats(self) -> dict:
        return {
            "text_chunks": len(self.store.text_meta),
            "figures": len(self.store.figure_meta),
            "hybrid": settings.use_hybrid,
            "top_k": settings.top_k,
            "chunk_size": settings.chunk_size,
            "generator": type(self.generator).__name__,
        }


@lru_cache(maxsize=1)
def get_chain() -> RAGChain:
    """Cached so the API loads FAISS once per process, not once per request."""
    return RAGChain()


def reset_chain() -> None:
    get_chain.cache_clear()
