"""Central configuration.

Every tunable the evaluation harness sweeps lives here, so an experiment is
reproducible from the environment alone.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- model backend -------------------------------------------------
    google_api_key: str = ""
    # Gemini Flash handles text and vision in one model, so the figure path
    # needs no separately deployed VLM.
    gen_model: str = "gemini-2.0-flash"
    embed_model: str = "text-embedding-004"
    # "gemini" | "hashing". hashing is a deterministic offline embedder used by
    # the test suite so nothing in CI depends on network access or a key.
    embedder: str = "gemini"

    # --- chunking (swept by eval/evaluate.py) --------------------------
    chunk_size: int = 900  # characters, not tokens: cheap, stable, good enough
    chunk_overlap: int = 150
    min_chunk_chars: int = 120  # drops page numbers, running headers, footers

    # --- retrieval -----------------------------------------------------
    top_k: int = 5
    top_k_figures: int = 2
    use_hybrid: bool = True  # dense + BM25, fused with reciprocal rank fusion
    rrf_k: int = 60

    # --- figure extraction ---------------------------------------------
    min_figure_px: int = 120  # ignore logos, rules, bullet glyphs
    min_figure_area_ratio: float = 0.012  # fraction of the page
    caption_search_px: float = 90.0  # vertical window when hunting a caption

    # --- paths ----------------------------------------------------------
    corpus_dir: Path = ROOT / "data" / "corpus"
    figures_dir: Path = ROOT / "data" / "figures"
    index_dir: Path = ROOT / "data" / "index"

    def ensure_dirs(self) -> None:
        for d in (self.corpus_dir, self.figures_dir, self.index_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
