from langchain_core.documents import Document

from app.index.embedder import HashingEmbedder
from app.index.store import VectorStore
from app.ingest.chunker import chunk_document
from app.rag.chain import RAGChain
from app.rag.generator import CITATION_RE, EchoGenerator


def _build_store(parsed):
    chunks = [c.to_dict() for c in chunk_document(parsed)]
    for row, c in enumerate(chunks):
        c["_row"] = row
    s = VectorStore(HashingEmbedder())
    s.build(chunks, [])
    return s


def test_citation_parsing():
    assert sorted({int(m) for m in CITATION_RE.findall("a [2] b [5][2] c")}) == [2, 5]


def test_chain_end_to_end_offline(parsed):
    chain = RAGChain(store=_build_store(parsed), generator=EchoGenerator())
    answer = chain.ask("how is velocity estimated")
    assert answer.text
    assert answer.cited_blocks == [1]
    assert answer.citations and "p." in answer.citations[0]
    assert answer.is_grounded_shape


def test_empty_context_is_refused():
    answer = EchoGenerator().answer("anything", [], [])
    assert "does not contain" in answer.text.lower()
    assert answer.citations == []


def test_grounded_shape_flags_uncited_assertion():
    from app.rag.generator import Answer

    doc = Document(page_content="x", metadata={"citation": "a p.1"})
    assert not Answer(text="YOLO runs at 45 FPS.", contexts=[doc]).is_grounded_shape


def test_retrieval_metrics_on_gold_set(parsed):
    from eval.evaluate import Question, evaluate_retrieval

    questions = [
        Question(question="how is velocity estimated", gold_pages=[3]),
        Question(question="what controls duplicate detections at tile borders",
                 gold_pages=[1]),
    ]
    result = evaluate_retrieval(_build_store(parsed), questions, k=5, k_figures=0)
    assert result["n"] == 2
    assert 0.0 <= result["hit_rate@k"] <= 1.0
    assert 0.0 <= result["mrr"] <= 1.0
    assert result["hit_rate@k"] > 0  # the hashing embedder should manage this much
