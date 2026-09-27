"""Evaluation harness.

Two things are measured, and they fail for different reasons, so they are kept
apart:

  Retrieval  -- did the right passage come back at all? Measured by hit-rate@k
                and MRR against hand-labelled gold pages. No model needed, so
                it is cheap enough to sweep over parameters.
  Generation -- given what came back, is the answer supported by it? Measured
                by an LLM judge for groundedness, plus a structural check that
                the answer cites anything at all.

A RAG system can score well on the second while being useless, if retrieval
never surfaced the right page. Reporting one number hides that.

    python -m eval.evaluate --questions eval/questions.jsonl
    python -m eval.evaluate --sweep
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.index.embedder import get_embedder  # noqa: E402
from app.index.store import VectorStore  # noqa: E402
from app.rag.chain import RAGChain  # noqa: E402
from app.rag.retriever import MultimodalRetriever  # noqa: E402


@dataclass
class Question:
    question: str
    gold_pages: list[int]
    gold_source: str = ""
    reference: str = ""
    needs_figure: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        return cls(
            question=d["question"],
            gold_pages=d.get("gold_pages", []),
            gold_source=d.get("gold_source", ""),
            reference=d.get("reference", ""),
            needs_figure=d.get("needs_figure", False),
        )


def load_questions(path: Path) -> list[Question]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [Question.from_dict(r) for r in rows]


def _is_hit(doc, q: Question) -> bool:
    meta = doc.metadata
    if q.gold_source and not meta["source"].startswith(q.gold_source):
        return False
    pages = set(meta.get("pages") or [meta["page"]])
    return bool(pages & set(q.gold_pages))


def evaluate_retrieval(store: VectorStore, questions: list[Question],
                       k: int, k_figures: int) -> dict:
    retriever = MultimodalRetriever(store=store, k=k, k_figures=k_figures)
    hits = 0
    reciprocal_ranks: list[float] = []
    figure_hits = 0
    figure_needed = 0

    for q in questions:
        text_docs, figure_docs = retriever.retrieve_split(q.question)
        rank = next((i + 1 for i, d in enumerate(text_docs) if _is_hit(d, q)), 0)
        if rank:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        if q.needs_figure:
            figure_needed += 1
            if any(_is_hit(d, q) for d in figure_docs):
                figure_hits += 1

    n = len(questions) or 1
    out = {
        "n": len(questions),
        "hit_rate@k": round(hits / n, 3),
        "mrr": round(sum(reciprocal_ranks) / n, 3),
    }
    if figure_needed:
        out["figure_hit_rate"] = round(figure_hits / figure_needed, 3)
        out["figure_questions"] = figure_needed
    return out


def _judge(client, model: str, prompt: str) -> dict:
    from google.genai import types

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0, response_mime_type="application/json"
        ),
    )
    try:
        return json.loads(resp.text)
    except Exception:
        return {}


def evaluate_generation(chain: RAGChain, questions: list[Question]) -> dict:
    """Groundedness and (where a reference is given) correctness."""
    from google import genai

    from app.rag.prompts import CORRECTNESS_JUDGE, GROUNDEDNESS_JUDGE

    if not settings.google_api_key:
        return {"skipped": "GOOGLE_API_KEY not set"}

    client = genai.Client(api_key=settings.google_api_key)
    grounded = cited = correct = graded_correct = 0
    latencies: list[float] = []

    for q in questions:
        started = time.perf_counter()
        answer = chain.ask(q.question)
        latencies.append((time.perf_counter() - started) * 1000)

        if answer.cited_blocks:
            cited += 1

        context = "\n\n".join(
            f"[{i + 1}] {d.page_content[:1200]}" for i, d in enumerate(answer.contexts)
        )
        verdict = _judge(client, settings.gen_model,
                         GROUNDEDNESS_JUDGE.format(context=context, answer=answer.text))
        if verdict.get("grounded"):
            grounded += 1

        if q.reference:
            graded_correct += 1
            verdict = _judge(
                client, settings.gen_model,
                CORRECTNESS_JUDGE.format(question=q.question, reference=q.reference,
                                         answer=answer.text),
            )
            if verdict.get("correct"):
                correct += 1
        time.sleep(1)  # free-tier rate limit

    n = len(questions) or 1
    out = {
        "groundedness": round(grounded / n, 3),
        "citation_rate": round(cited / n, 3),
        "median_latency_ms": int(sorted(latencies)[len(latencies) // 2]) if latencies else 0,
    }
    if graded_correct:
        out["answer_correctness"] = round(correct / graded_correct, 3)
    return out


def sweep(questions: list[Question], chunk_sizes: list[int], ks: list[int]) -> list[dict]:
    """Grid over chunk size and k.

    Chunk size changes the index, so each value costs a full re-embed; k does
    not, so it is swept inside the chunk loop. Ordering the loops the other way
    round would multiply the embedding bill by len(ks) for no benefit.
    """
    from app.ingest.pipeline import ingest_corpus

    rows: list[dict] = []
    original = settings.chunk_size
    try:
        for chunk_size in chunk_sizes:
            settings.chunk_size = chunk_size
            settings.chunk_overlap = max(50, chunk_size // 6)
            print(f"\n[sweep] re-indexing at chunk_size={chunk_size} ...")
            stats = ingest_corpus(describe=False)  # descriptions are k-invariant
            store = VectorStore.load(get_embedder())
            for k in ks:
                result = evaluate_retrieval(store, questions, k=k, k_figures=2)
                row = {"chunk_size": chunk_size, "k": k,
                       "chunks": stats["chunks"], **result}
                rows.append(row)
                print(f"  k={k:<3} hit@k={row['hit_rate@k']:<6} mrr={row['mrr']}")
    finally:
        settings.chunk_size = original
    return rows


def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return "(no results)"
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--k", type=int, default=settings.top_k)
    ap.add_argument("--sweep", action="store_true", help="grid over chunk_size and k")
    ap.add_argument("--generation", action="store_true",
                    help="also run the LLM-judge groundedness evaluation")
    ap.add_argument("--out", default="eval/results.md")
    args = ap.parse_args()

    questions = load_questions(Path(args.questions))
    print(f"Loaded {len(questions)} questions")

    report: list[str] = ["# Evaluation results", ""]

    if args.sweep:
        rows = sweep(questions, chunk_sizes=[500, 900, 1400], ks=[3, 5, 8])
        report += ["## Retrieval sweep", "", to_markdown(rows), ""]
    else:
        store = VectorStore.load(get_embedder())
        result = evaluate_retrieval(store, questions, k=args.k,
                                    k_figures=settings.top_k_figures)
        print(json.dumps(result, indent=2))
        report += ["## Retrieval", "", to_markdown([{"k": args.k, **result}]), ""]

    if args.generation:
        chain = RAGChain()
        gen = evaluate_generation(chain, questions)
        print(json.dumps(gen, indent=2))
        report += ["## Generation", "", to_markdown([gen]), ""]

    Path(args.out).write_text("\n".join(report))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
