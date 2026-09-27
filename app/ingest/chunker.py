"""Turn parsed blocks into retrievable chunks.

Two properties matter more than cleverness here. A chunk must never span a
section boundary, because mixing "Related Work" into "Method" is how a RAG
system answers with the wrong paper's approach. And every chunk must carry the
page it came from, because that is what makes a citation checkable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from app.config import settings
from app.ingest.parser import ParsedDoc, TextBlock


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int  # first page the chunk covers
    pages: list[int]  # every page, for chunks straddling a page break
    section: str
    kind: str = "text"  # "text" | "caption"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def citation(self) -> str:
        return f"{self.source} p.{self.page}"


def _flush(buf: list[TextBlock], source: str, idx: int) -> Chunk | None:
    if not buf:
        return None
    text = "\n".join(b.text for b in buf).strip()
    if len(text) < settings.min_chunk_chars:
        return None
    pages = sorted({b.page for b in buf})
    section = buf[0].section
    return Chunk(
        chunk_id=f"{source}::{idx}",
        text=(f"[{section}]\n{text}" if section else text),
        source=source,
        page=pages[0],
        pages=pages,
        section=section,
        kind="caption" if all(b.is_caption for b in buf) else "text",
    )


def _split_long(block: TextBlock) -> list[TextBlock]:
    """Split an oversized block on sentence boundaries where possible."""
    size, overlap = settings.chunk_size, settings.chunk_overlap
    if len(block.text) <= size:
        return [block]

    pieces: list[TextBlock] = []
    text, start = block.text, 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text.rfind(". ", start + size // 2, end)
            if window != -1:
                end = window + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(
                TextBlock(text=piece, page=block.page, bbox=block.bbox,
                          size=block.size, is_caption=block.is_caption,
                          section=block.section)
            )
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def chunk_document(doc: ParsedDoc) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf: list[TextBlock] = []
    buf_len = 0
    idx = 0

    def emit() -> None:
        nonlocal buf, buf_len, idx
        chunk = _flush(buf, doc.source, idx)
        if chunk is not None:
            chunks.append(chunk)
            idx += 1
        buf, buf_len = [], 0

    for block in doc.blocks:
        for piece in _split_long(block):
            crosses_section = buf and piece.section != buf[-1].section
            too_long = buf_len + len(piece.text) > settings.chunk_size
            if crosses_section or too_long:
                emit()
                # Overlap carries the tail of the previous chunk forward so a
                # fact split across a boundary stays retrievable from one side.
                if not crosses_section and chunks and settings.chunk_overlap > 0:
                    tail = chunks[-1].text[-settings.chunk_overlap:]
                    buf = [TextBlock(text=tail, page=piece.page, bbox=piece.bbox,
                                     size=piece.size, section=piece.section)]
                    buf_len = len(tail)
            buf.append(piece)
            buf_len += len(piece.text)

    emit()
    return chunks
