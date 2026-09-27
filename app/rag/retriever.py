"""A LangChain retriever over the multimodal store.

Implementing BaseRetriever rather than using a stock vector-store retriever is
deliberate: the figure branch, the RRF fusion and the page metadata that
citations depend on all live in here, and none of them fit a drop-in wrapper.
"""
from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from app.config import settings
from app.index.store import VectorStore


class MultimodalRetriever(BaseRetriever):
    store: VectorStore
    k: int = Field(default_factory=lambda: settings.top_k)
    k_figures: int = Field(default_factory=lambda: settings.top_k_figures)
    include_figures: bool = True

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        docs: list[Document] = []

        for hit in self.store.search_text(query, k=self.k):
            p = hit.payload
            docs.append(
                Document(
                    page_content=p["text"],
                    metadata={
                        "kind": "text",
                        "source": p["source"],
                        "page": p["page"],
                        "pages": p.get("pages", [p["page"]]),
                        "section": p.get("section", ""),
                        "score": hit.score,
                        "citation": f'{p["source"]} p.{p["page"]}',
                    },
                )
            )

        if self.include_figures and self.k_figures > 0:
            for hit in self.store.search_figures(query, k=self.k_figures):
                p = hit.payload
                body = "\n".join(
                    x for x in (p.get("caption", ""), p.get("description", "")) if x
                ) or "(figure with no caption)"
                docs.append(
                    Document(
                        page_content=body,
                        metadata={
                            "kind": "figure",
                            "source": p["source"],
                            "page": p["page"],
                            "figure_id": p["figure_id"],
                            "image_path": p["image_path"],
                            "score": hit.score,
                            "citation": f'{p["source"]} p.{p["page"]} (figure)',
                        },
                    )
                )
        return docs

    def retrieve_split(self, query: str) -> tuple[list[Document], list[Document]]:
        """Same retrieval, split by modality -- the generator needs them apart
        because figures are sent to the model as images, not as text."""
        docs = self._get_relevant_documents(query)
        text = [d for d in docs if d.metadata["kind"] == "text"]
        figures = [d for d in docs if d.metadata["kind"] == "figure"]
        return text, figures

    def with_overrides(self, **kwargs: Any) -> "MultimodalRetriever":
        return MultimodalRetriever(
            store=self.store,
            k=kwargs.get("k", self.k),
            k_figures=kwargs.get("k_figures", self.k_figures),
            include_figures=kwargs.get("include_figures", self.include_figures),
        )
