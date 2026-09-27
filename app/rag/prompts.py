"""Prompt templates.

Kept in one file so a prompt change is a reviewable diff rather than something
buried in application logic.
"""
from __future__ import annotations

ANSWER_SYSTEM = """You answer questions about computer-vision research papers \
using only the context provided.

Rules:
- Use only the numbered context blocks. If they do not contain the answer, say \
so plainly and stop. Do not fall back on general knowledge.
- Cite every factual claim inline as [n], where n is the context block number. \
A sentence drawing on two blocks gets both, like [2][5].
- Some context blocks are figures, supplied as images with their captions. \
Refer to them the same way, by block number.
- Prefer the paper's own terminology over paraphrase.
- Be concise. Three to six sentences unless the question demands more."""


def build_context_block(index: int, doc) -> str:
    meta = doc.metadata
    header = f"[{index}] {meta['source']}, page {meta['page']}"
    if meta.get("section"):
        header += f", section: {meta['section']}"
    if meta["kind"] == "figure":
        header += " (FIGURE — image attached below)"
    return f"{header}\n{doc.page_content}"


ANSWER_USER = """Context blocks:

{context}

Question: {question}

Answer using only the context above, with inline [n] citations."""


# --- evaluation -------------------------------------------------------------

GROUNDEDNESS_JUDGE = """You are grading whether an answer is grounded in its \
retrieved context. Grounded means every factual claim in the answer is \
supported by the context. Fluency, style and completeness are irrelevant here; \
only support matters.

Context:
{context}

Answer:
{answer}

Reply with strict JSON and nothing else:
{{"grounded": true|false, "unsupported_claims": ["..."], "reason": "one sentence"}}"""


CORRECTNESS_JUDGE = """Compare a generated answer against a reference answer \
for the same question. They need not match in wording; judge whether the \
generated answer conveys the same substantive facts.

Question: {question}
Reference answer: {reference}
Generated answer: {answer}

Reply with strict JSON and nothing else:
{{"correct": true|false, "reason": "one sentence"}}"""
