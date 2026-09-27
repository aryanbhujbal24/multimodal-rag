import pytest

from app.index.embedder import HashingEmbedder
from app.index.store import VectorStore
from app.ingest.chunker import chunk_document
from app.rag.retriever import MultimodalRetriever


@pytest.fixture(scope="module")
def store(parsed):
    chunks = [c.to_dict() for c in chunk_document(parsed)]
    for row, c in enumerate(chunks):
        c["_row"] = row
    figures = []
    for row, f in enumerate(parsed.figures):
        figures.append({
            "figure_id": f.figure_id, "source": f.source, "page": f.page,
            "image_path": f.image_path, "caption": f.caption,
            "description": "", "searchable_text": f.searchable_text, "_row": row,
        })
    s = VectorStore(HashingEmbedder())
    s.build(chunks, figures)
    return s


def test_embeddings_are_normalised():
    v = HashingEmbedder().embed(["kalman filter tracking", "small object detection"])
    norms = (v ** 2).sum(axis=1) ** 0.5
    assert all(abs(n - 1.0) < 1e-5 for n in norms)


def test_text_search_returns_relevant_chunk(store):
    hits = store.search_text("how is velocity estimated", k=3)
    assert hits
    assert any("velocity" in h.payload["text"].lower() for h in hits)


def test_hybrid_and_dense_both_work(store):
    assert store.search_text("kalman filter", k=3, hybrid=False)
    assert store.search_text("kalman filter", k=3, hybrid=True)


def test_figure_search_finds_architecture_diagram(store):
    hits = store.search_figures("overall architecture diagram", k=1)
    assert hits
    assert "architecture" in hits[0].payload["caption"].lower()


def test_save_and_load_roundtrip(store, tmp_path):
    store.save(tmp_path)
    loaded = VectorStore.load(HashingEmbedder(), tmp_path)
    assert len(loaded.text_meta) == len(store.text_meta)
    assert len(loaded.figure_meta) == len(store.figure_meta)
    assert loaded.search_text("velocity", k=2)


def test_retriever_returns_langchain_documents_with_citations(store):
    retriever = MultimodalRetriever(store=store, k=3, k_figures=1)
    text_docs, figure_docs = retriever.retrieve_split("architecture of the detector")
    assert text_docs and figure_docs
    for doc in text_docs + figure_docs:
        assert doc.metadata["citation"]
        assert doc.metadata["page"] >= 1
    assert figure_docs[0].metadata["kind"] == "figure"
    assert figure_docs[0].metadata["image_path"]


def test_retriever_respects_k(store):
    retriever = MultimodalRetriever(store=store, k=2, k_figures=0)
    text_docs, figure_docs = retriever.retrieve_split("detector")
    assert len(text_docs) <= 2
    assert figure_docs == []
