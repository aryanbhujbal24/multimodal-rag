"""End-to-end ingest: PDFs in, searchable index out."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from app.config import settings
from app.index.embedder import get_embedder
from app.index.store import VectorStore
from app.ingest.chunker import chunk_document
from app.ingest.parser import parse_pdf


def describe_figures(figures: list[dict], enabled: bool = True) -> list[dict]:
    """Ask the VLM what each figure actually shows.

    A caption says "Figure 3: Overall architecture". The pixels say it is a
    two-branch network with a feature pyramid and a detection head. Indexing the
    second makes the figure findable by questions that never use the caption's
    words, which is the entire reason for doing vision work at index time
    instead of at query time.
    """
    if not enabled or not figures:
        return figures
    if settings.embedder == "hashing" or not settings.google_api_key:
        return figures  # offline mode: captions only

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.google_api_key)
    prompt = (
        "Describe this figure from a computer-vision research paper in 2-4 "
        "sentences. State what kind of figure it is (architecture diagram, "
        "qualitative results, plot, table), name the components or axes that "
        "are labelled, and state what it is demonstrating. Do not speculate "
        "beyond what is visible."
    )

    for fig in figures:
        path = Path(fig["image_path"])
        if not path.exists():
            continue
        try:
            resp = client.models.generate_content(
                model=settings.gen_model,
                contents=[
                    types.Part.from_bytes(data=path.read_bytes(), mime_type="image/png"),
                    prompt,
                ],
            )
            fig["description"] = (resp.text or "").strip()
        except Exception:
            # A failed description degrades retrieval for one figure; it should
            # never abort an ingest of 200 pages.
            fig["description"] = ""
        fig["searchable_text"] = "\n".join(
            p for p in (fig.get("caption", ""), fig.get("description", ""),
                        fig.get("section", "")) if p
        )
    return figures


def ingest_paths(paths: list[Path], *, describe: bool = True) -> dict:
    settings.ensure_dirs()
    all_chunks: list[dict] = []
    all_figures: list[dict] = []

    for path in paths:
        doc = parse_pdf(path)
        for chunk in chunk_document(doc):
            all_chunks.append(chunk.to_dict())
        for fig in doc.figures:
            record = asdict(fig)
            record["searchable_text"] = fig.searchable_text
            all_figures.append(record)

    # Row ids let reciprocal rank fusion address a chunk by position.
    for row, chunk in enumerate(all_chunks):
        chunk["_row"] = row
    for row, fig in enumerate(all_figures):
        fig["_row"] = row

    all_figures = describe_figures(all_figures, enabled=describe)

    store = VectorStore(get_embedder())
    store.build(all_chunks, all_figures)
    store.save()

    return {
        "documents": len(paths),
        "chunks": len(all_chunks),
        "figures": len(all_figures),
        "figures_described": sum(1 for f in all_figures if f.get("description")),
    }


def ingest_corpus(directory: Path | None = None, **kwargs) -> dict:
    directory = Path(directory or settings.corpus_dir)
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs in {directory}. Run: python scripts/fetch_corpus.py"
        )
    return ingest_paths(pdfs, **kwargs)
