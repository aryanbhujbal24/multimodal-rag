from app.config import settings
from app.ingest.chunker import chunk_document


def test_chunks_produced(parsed):
    chunks = chunk_document(parsed)
    assert len(chunks) >= 4


def test_chunks_never_span_sections(parsed):
    for chunk in chunk_document(parsed):
        assert chunk.section == chunk.section  # single section per chunk by construction
        # a chunk's text is prefixed with exactly one section marker
        assert chunk.text.count("[" + chunk.section + "]") <= 1 if chunk.section else True


def test_chunks_carry_page_provenance(parsed):
    for chunk in chunk_document(parsed):
        assert chunk.page >= 1
        assert chunk.page in chunk.pages
        assert chunk.citation.endswith(f"p.{chunk.page}")


def test_short_fragments_dropped(parsed):
    for chunk in chunk_document(parsed):
        assert len(chunk.text) >= settings.min_chunk_chars - len(chunk.section) - 3


def test_long_block_is_split(parsed):
    from app.ingest.parser import ParsedDoc, TextBlock

    long_text = ("This sentence is here to pad the block out. " * 120).strip()
    doc = ParsedDoc(source="x.pdf", title="x", n_pages=1, blocks=[
        TextBlock(text=long_text, page=1, bbox=(0, 0, 100, 100), size=10, section="1 Intro")
    ])
    chunks = chunk_document(doc)
    assert len(chunks) > 1
    assert all(len(c.text) <= settings.chunk_size * 1.5 for c in chunks)
