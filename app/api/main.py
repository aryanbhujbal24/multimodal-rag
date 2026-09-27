"""FastAPI service."""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.ingest.pipeline import ingest_corpus
from app.rag.chain import get_chain, reset_chain

app = FastAPI(
    title="Multimodal RAG for Technical Documents",
    version="1.0.0",
    description="Grounded question answering over diagram-heavy PDFs, "
                "with page-level citations and figure retrieval.",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=20)
    k_figures: int | None = Field(default=None, ge=0, le=6)


class ContextOut(BaseModel):
    block: int
    kind: str
    source: str
    page: int
    citation: str
    score: float
    section: str = ""
    figure_id: str | None = None
    excerpt: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[str]
    contexts: list[ContextOut]
    figures: list[dict]
    grounded_shape: bool
    latency_ms: int


@app.get("/healthz")
def healthz() -> dict:
    try:
        return {"status": "ok", **get_chain().stats}
    except Exception as exc:
        # An unbuilt index is an expected state on a cold container, not a crash.
        return {"status": "degraded", "detail": str(exc)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    chain = get_chain()
    if chain.store.is_empty:
        raise HTTPException(status_code=409, detail="Index is empty. POST /ingest first.")

    started = time.perf_counter()
    try:
        result = chain.ask(req.question, k=req.k, k_figures=req.k_figures)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Generation failed: {exc}") from exc

    contexts = [
        ContextOut(
            block=i + 1,
            kind=d.metadata["kind"],
            source=d.metadata["source"],
            page=d.metadata["page"],
            citation=d.metadata["citation"],
            score=round(float(d.metadata.get("score", 0.0)), 4),
            section=d.metadata.get("section", ""),
            figure_id=d.metadata.get("figure_id"),
            excerpt=d.page_content[:400],
        )
        for i, d in enumerate(result.contexts)
    ]
    return QueryResponse(
        question=req.question,
        answer=result.text,
        citations=result.citations,
        contexts=contexts,
        figures=result.figures_used,
        grounded_shape=result.is_grounded_shape,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


@app.post("/ingest")
def ingest(background: BackgroundTasks, describe_figures: bool = True) -> dict:
    """Rebuild the index from data/corpus.

    Synchronous ingest of a large corpus outlives an HTTP timeout, so the work
    is queued and the caller polls /healthz for the updated counts.
    """
    def _run() -> None:
        ingest_corpus(describe=describe_figures)
        reset_chain()

    background.add_task(_run)
    return {"status": "started", "corpus_dir": str(settings.corpus_dir)}


@app.get("/figures/{figure_id}")
def figure(figure_id: str) -> FileResponse:
    if not figure_id.isalnum():  # path traversal guard
        raise HTTPException(status_code=400, detail="Invalid figure id.")
    path = Path(settings.figures_dir) / f"{figure_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Figure not found.")
    return FileResponse(path, media_type="image/png")
