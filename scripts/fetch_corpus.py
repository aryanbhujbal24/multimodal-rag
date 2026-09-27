"""Download the demo corpus: the detection and tracking papers behind the
DRDO work. All are open-access arXiv preprints.

    python scripts/fetch_corpus.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402

PAPERS = {
    "2110.06864": "bytetrack",          # ByteTrack: association by low-score boxes
    "2202.06934": "sahi",               # Slicing Aided Hyper Inference
    "1506.02640": "yolo_v1",            # You Only Look Once
    "2004.10934": "yolo_v4",            # YOLOv4
    "1602.00763": "sort",               # Simple Online and Realtime Tracking
    "1703.07402": "deepsort",           # SORT with a deep association metric
    "2004.01888": "fairmot",            # FairMOT
    "1612.03144": "fpn",                # Feature Pyramid Networks
}


def fetch(arxiv_id: str, name: str, out_dir: Path) -> bool:
    dest = out_dir / f"{name}.pdf"
    if dest.exists() and dest.stat().st_size > 50_000:
        print(f"  skip {name} (already present)")
        return True
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    req = Request(url, headers={"User-Agent": "multimodal-rag/1.0 (student project)"})
    try:
        with urlopen(req, timeout=60) as resp:
            data = resp.read()
    except Exception as exc:
        print(f"  FAILED {name}: {exc}")
        return False
    if not data.startswith(b"%PDF"):
        print(f"  FAILED {name}: response was not a PDF")
        return False
    dest.write_bytes(data)
    print(f"  saved {name} ({len(data) // 1024} KB)")
    return True


def main() -> None:
    settings.ensure_dirs()
    out = Path(settings.corpus_dir)
    print(f"Fetching {len(PAPERS)} papers into {out}")
    ok = 0
    for arxiv_id, name in PAPERS.items():
        if fetch(arxiv_id, name, out):
            ok += 1
        time.sleep(3)  # arXiv asks for a few seconds between requests
    print(f"\n{ok}/{len(PAPERS)} available. Next: python -m scripts.build_index")


if __name__ == "__main__":
    main()
