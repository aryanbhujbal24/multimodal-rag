import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def offline_mode(tmp_path_factory):
    """Force the offline embedder and isolate artefacts from the real index."""
    tmp = tmp_path_factory.mktemp("mmrag")
    settings.embedder = "hashing"
    settings.google_api_key = ""
    settings.figures_dir = tmp / "figures"
    settings.index_dir = tmp / "index"
    settings.ensure_dirs()
    return settings


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory):
    from scripts.make_sample_pdf import build

    return build(tmp_path_factory.mktemp("pdf") / "synthetic_paper.pdf")


@pytest.fixture(scope="session")
def parsed(sample_pdf):
    from app.ingest.parser import parse_pdf

    return parse_pdf(sample_pdf)
