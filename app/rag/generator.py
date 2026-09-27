"""Answer generation against retrieved text and figures."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from app.config import settings
from app.rag.prompts import ANSWER_SYSTEM, ANSWER_USER, build_context_block

CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    text: str
    contexts: list[Document] = field(default_factory=list)
    cited_blocks: list[int] = field(default_factory=list)
    figures_used: list[dict] = field(default_factory=list)

    @property
    def citations(self) -> list[str]:
        out: list[str] = []
        for n in self.cited_blocks:
            if 1 <= n <= len(self.contexts):
                cite = self.contexts[n - 1].metadata["citation"]
                if cite not in out:
                    out.append(cite)
        return out

    @property
    def is_grounded_shape(self) -> bool:
        """Cheap structural check: an answer asserting facts with no citation
        at all is a red flag worth surfacing before any LLM judge runs."""
        refused = "does not contain" in self.text.lower() or "not in the context" in self.text.lower()
        return bool(self.cited_blocks) or refused


class Generator:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or settings.gen_model
        self._key = api_key or settings.google_api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai

            if not self._key:
                raise RuntimeError("GOOGLE_API_KEY is not set.")
            self._client = genai.Client(api_key=self._key)
        return self._client

    def answer(self, question: str, text_docs: list[Document],
               figure_docs: list[Document]) -> Answer:
        from google.genai import types

        # Numbering is shared across both modalities so the model can cite a
        # figure and a passage in the same sentence.
        contexts = text_docs + figure_docs
        blocks = [build_context_block(i + 1, d) for i, d in enumerate(contexts)]
        user_text = ANSWER_USER.format(context="\n\n".join(blocks), question=question)

        parts: list = [user_text]
        figures_used: list[dict] = []
        for offset, doc in enumerate(figure_docs):
            path = Path(doc.metadata["image_path"])
            if not path.exists():
                continue
            block_no = len(text_docs) + offset + 1
            parts.append(f"Image for context block [{block_no}]:")
            parts.append(
                types.Part.from_bytes(data=path.read_bytes(), mime_type="image/png")
            )
            figures_used.append(
                {
                    "block": block_no,
                    "figure_id": doc.metadata["figure_id"],
                    "page": doc.metadata["page"],
                    "source": doc.metadata["source"],
                    "image_path": str(path),
                }
            )

        resp = self.client.models.generate_content(
            model=self.model,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=ANSWER_SYSTEM,
                temperature=0.1,  # extraction, not creative writing
            ),
        )
        text = (resp.text or "").strip()
        cited = sorted({int(m) for m in CITATION_RE.findall(text)})
        return Answer(text=text, contexts=contexts, cited_blocks=cited,
                      figures_used=figures_used)


class EchoGenerator:
    """Offline stand-in used by tests: returns the top context verbatim with a
    citation, so chain wiring and citation parsing can be tested without a key."""

    def answer(self, question: str, text_docs: list[Document],
               figure_docs: list[Document]) -> Answer:
        contexts = text_docs + figure_docs
        if not contexts:
            return Answer(text="The context does not contain an answer.", contexts=[])
        snippet = contexts[0].page_content[:300].replace("\n", " ")
        figures_used = [
            {
                "block": len(text_docs) + i + 1,
                "figure_id": d.metadata["figure_id"],
                "page": d.metadata["page"],
                "source": d.metadata["source"],
                "image_path": d.metadata["image_path"],
            }
            for i, d in enumerate(figure_docs)
        ]
        return Answer(text=f"{snippet} [1]", contexts=contexts, cited_blocks=[1],
                      figures_used=figures_used)
