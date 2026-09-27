"""Build the index from data/corpus.

    python -m scripts.build_index              # with VLM figure descriptions
    python -m scripts.build_index --no-describe  # captions only, no API calls
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.ingest.pipeline import ingest_corpus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-describe", action="store_true",
                    help="skip vision-language figure descriptions")
    args = ap.parse_args()

    started = time.perf_counter()
    stats = ingest_corpus(describe=not args.no_describe)
    stats["seconds"] = round(time.perf_counter() - started, 1)
    for key, value in stats.items():
        print(f"{key:>20}: {value}")


if __name__ == "__main__":
    main()
