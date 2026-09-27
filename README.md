# Multimodal RAG for Technical Documents

Grounded question answering over diagram-heavy research papers. Answers cite the
page they came from, and figures are retrieved as figures — cropped, described by
a vision-language model at index time, and passed back to the model as images
when they are relevant to the question.

Built and demonstrated on the detection and tracking literature: ByteTrack,
SAHI, SORT, DeepSORT, YOLO, FPN, FairMOT.

---

## Why this is not a plain RAG demo

A standard RAG pipeline extracts text and throws the pictures away. In a
computer-vision paper that discards the architecture diagrams, the qualitative
result grids and the ablation tables — often the only place a fact appears at
all. Three parts of this system exist to fix that.

**Figures are found by their captions, not by extracting images.** Pulling
embedded raster objects out of a PDF only finds figures that happen to be
bitmaps. Most paper figures are vector drawings, and a composite figure is
typically dozens of separate drawing operations plus text labels — there is no
single object to extract. Captions are the reliable anchor: every real figure
has one, it marks the figure's horizontal extent, and the whitespace between it
and the previous body text bounds it vertically. The pipeline hunts captions,
unions the vector and raster ink above each one, absorbs the text blocks lying
*inside* that hull (a diagram's own labels), then renders the region to PNG.
`app/ingest/parser.py:caption_anchored_regions`

**Figures are described before they are indexed.** A caption says "Figure 3:
Overall architecture". The pixels say it is a two-branch network with a feature
pyramid and a detection head. The VLM description is embedded alongside the
caption, so a figure is retrievable by what it *shows*, not only by what its
caption happens to word. Doing this at index time rather than query time means
one vision call per figure, forever, instead of one per query.

**Figures and text are indexed separately.** In a single index, short captions
lose to long passages on almost every query. Two indexes with separate top-k
means a diagram question can surface a diagram even when five text passages
score higher.

---

## Architecture

```
PDFs ──► parser ──────► text blocks (page, section, bbox)
           │            figures (crop + caption + section)
           │
           ├──► chunker ──────► section-bounded chunks, page provenance kept
           │                          │
           ├──► VLM describe ─────────┤
           │    (Gemini Flash)        │
           │                          ▼
           │                   embedder (text-embedding-004)
           │                          │
           │                  ┌───────┴────────┐
           │                  ▼                ▼
           │           FAISS text idx   FAISS figure idx
           │              + BM25
           │                  └───────┬────────┘
           │                          ▼
           │              MultimodalRetriever (LangChain)
           │                  dense + BM25 fused by RRF
           │                          ▼
           └────────────────► Generator (Gemini Flash)
                              text blocks + figure images
                                        ▼
                              answer with [n] citations
```

Served by FastAPI, with a Streamlit client and a Docker Compose stack.

### Design decisions worth defending

| Decision | Why |
|---|---|
| Chunks never cross a section boundary | Mixing Related Work into Method is how a RAG system answers with a different paper's approach |
| Reciprocal rank fusion, not score averaging | Cosine and BM25 scores are on incomparable scales; only ranks combine honestly without a per-corpus weight |
| Separate document and query embedding task types | `RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY` is free accuracy in `text-embedding-004` |
| Page kept on every chunk | A citation nobody can check is decoration |
| Render the page region, not the raw xref | Axis labels, arrows and vector overlays drawn on top of a bitmap belong to the figure |
| `IndexFlatIP` over normalised vectors | Exact search; at this corpus size an approximate index would trade recall for speed nobody needs |
| Offline hashing embedder | The whole pipeline, tests included, runs with no key and no network |

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # add your key from aistudio.google.com/apikey
make corpus                   # download 8 arXiv papers
make index                    # parse, describe figures, embed, build FAISS
make serve                    # API on :8000
make ui                       # Streamlit on :8501
```

Or the whole stack: `docker compose up --build`.

No API key? Everything still runs:

```bash
EMBEDDER=hashing make index-fast
EMBEDDER=hashing make serve
```

Retrieval quality is meaningless in that mode, but every component is exercised.

---

## Evaluation

```bash
make eval      # retrieval metrics + LLM-judge groundedness
make sweep     # grid over chunk_size × k
```

Retrieval and generation are measured separately, because they fail for
different reasons. A system can look perfectly grounded while being useless, if
retrieval never surfaced the right page.

- **hit-rate@k** — did any retrieved chunk come from a gold page
- **MRR** — how far down the list it was
- **figure hit-rate** — measured only on questions flagged `needs_figure`
- **groundedness** — LLM judge: is every claim supported by the retrieved context
- **citation rate** — structural check that the answer cites anything at all
- **answer correctness** — LLM judge against a reference answer

`eval/questions.jsonl` holds 15 labelled questions over the arXiv corpus.
**The gold page numbers in it are unverified** — open each paper, confirm the
page, and correct them before quoting any number from this harness.

The sweep loops chunk size on the outside and k on the inside, because chunk
size forces a re-embed and k does not.

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /query` | `{question, k?, k_figures?}` → answer, citations, contexts, figures, latency |
| `POST /ingest` | Rebuild the index from `data/corpus` (backgrounded) |
| `GET /figures/{id}` | PNG crop of a retrieved figure |
| `GET /healthz` | Index counts and active parameters |

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"How does ByteTrack use low-confidence detections?"}' | jq
```

---

## Tests

```bash
make test     # 27 tests, no network, no API key
```

Covers caption and heading regexes, cross-column caption theft, figure crop
geometry, section-bounded chunking, page provenance, embedding normalisation,
dense and hybrid search, index save/load, retriever contracts, citation parsing,
the refusal path, and API validation including path traversal on `/figures`.

---

## Known limits

- Two-column layouts with figures spanning both columns can crop to one column,
  because the caption's width is the horizontal anchor.
- Tables are captured as images, so their numbers are not searchable as text.
  Structured table extraction is the obvious next step.
- `IndexFlatIP` is exact and linear; past ~10⁵ chunks this needs IVF or HNSW.
- The LLM judge shares a model family with the generator, which biases
  groundedness upward. A different judge model would be a fairer check.
- Free-tier rate limits mean ingest of a large corpus is throttled by the
  figure-description pass, not by parsing or embedding.
