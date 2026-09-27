"""Layout-aware PDF parsing.

Two things come out of a PDF here: text blocks tagged with the section they
belong to, and figures paired with their captions. The second is the part that
makes the system multimodal -- a figure without its caption is nearly
unretrievable, because the pixels alone carry no searchable text.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from app.config import settings

# "Figure 3:", "Fig. 3.", "Table 2 -", "Algorithm 1" ...
CAPTION_RE = re.compile(
    r"^\s*(figure|fig\.?|table|tab\.?|algorithm|alg\.?)\s*([0-9]+[a-z]?)\s*[.:)\-\u2013]?\s+",
    re.IGNORECASE,
)

# Headings in papers: "3 Method", "3.1. Detection", "IV. EXPERIMENTS", "Abstract"
HEADING_RE = re.compile(
    r"^\s*("
    r"(?:[0-9]+(?:\.[0-9]+)*\.?)\s+[A-Z][^.!?]{2,60}"  # 3.1 Related Work
    r"|(?:[IVXL]+\.)\s+[A-Z][^.!?]{2,60}"  # IV. Experiments
    r"|abstract|introduction|related work|method(?:s|ology)?|experiments?"
    r"|results?|discussion|conclusions?|references|appendix"
    r")\s*$",
    re.IGNORECASE,
)


@dataclass
class TextBlock:
    text: str
    page: int  # 1-based, matches what a reader sees
    bbox: tuple[float, float, float, float]
    size: float  # dominant font size, used for heading detection
    is_caption: bool = False
    section: str = ""


@dataclass
class Figure:
    figure_id: str
    source: str
    page: int
    bbox: tuple[float, float, float, float]
    image_path: str
    caption: str = ""
    section: str = ""
    # Filled in later by the VLM during indexing; a visual description makes the
    # figure retrievable by what it *shows*, not only by what its caption says.
    description: str = ""

    @property
    def searchable_text(self) -> str:
        parts = [p for p in (self.caption, self.description, self.section) if p]
        return "\n".join(parts)


@dataclass
class ParsedDoc:
    source: str
    title: str
    blocks: list[TextBlock] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    n_pages: int = 0


def _dominant_size(block: dict) -> float:
    sizes: list[float] = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            sizes.append(span.get("size", 0.0))
    return max(sizes) if sizes else 0.0


def _block_text(block: dict) -> str:
    out: list[str] = []
    for line in block.get("lines", []):
        out.append("".join(span.get("text", "") for span in line.get("spans", [])))
    # Join hyphenated line breaks: "detec-\ntion" -> "detection"
    text = "\n".join(out)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _vertical_gap(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Distance between two bboxes vertically; 0 if they overlap."""
    if b[1] >= a[3]:
        return b[1] - a[3]
    if a[1] >= b[3]:
        return a[1] - b[3]
    return 0.0


def _horizontal_overlap(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    left, right = max(a[0], b[0]), min(a[2], b[2])
    if right <= left:
        return 0.0
    narrower = min(a[2] - a[0], b[2] - b[0]) or 1.0
    return (right - left) / narrower


def find_caption(fig_bbox: tuple[float, float, float, float],
                 blocks: list[TextBlock]) -> str:
    """Pick the caption belonging to a figure.

    Captions sit below the figure in most papers and above it for tables, so
    both directions are searched. Candidates must look like a caption, sit
    within a vertical window, and overlap the figure horizontally -- the last
    check is what stops a caption in the neighbouring column being stolen.
    """
    best: tuple[float, str] | None = None
    for b in blocks:
        if not b.is_caption:
            continue
        gap = _vertical_gap(fig_bbox, b.bbox)
        if gap > settings.caption_search_px:
            continue
        if _horizontal_overlap(fig_bbox, b.bbox) < 0.25:
            continue
        below = b.bbox[1] >= fig_bbox[3]
        # Prefer captions below: bias the score rather than filtering, so a
        # table's caption above it is still found when nothing sits below.
        score = gap if below else gap + 25.0
        if best is None or score < best[0]:
            best = (score, b.text)
    return best[1] if best else ""


def _keep_image(rect: pymupdf.Rect, page_area: float) -> bool:
    if rect.width < settings.min_figure_px or rect.height < settings.min_figure_px:
        return False
    if page_area <= 0:
        return False
    return (rect.width * rect.height) / page_area >= settings.min_figure_area_ratio


def caption_anchored_regions(
    page: pymupdf.Page, page_blocks: list[TextBlock]
) -> list[tuple[pymupdf.Rect, str]]:
    """Find figures by working backwards from their captions.

    Extracting embedded raster xrefs only finds figures that happen to be
    bitmaps. In research papers most diagrams are vector drawings, and a
    composite figure is often dozens of separate drawing operations plus text
    labels -- there is no single object to extract.

    Captions are the reliable anchor. Every real figure has one, it names the
    figure's horizontal extent, and the whitespace between the caption and the
    previous body text bounds it vertically. Rendering that region to PNG
    captures the figure exactly as a reader sees it, vector or raster.
    """
    regions: list[tuple[pymupdf.Rect, str]] = []
    page_rect = page.rect
    page_area = abs(page_rect.width * page_rect.height)

    # Anything that draws ink: vector paths and placed images alike.
    ink: list[pymupdf.Rect] = []
    try:
        ink.extend(pymupdf.Rect(d["rect"]) for d in page.get_drawings())
    except Exception:
        pass
    for xref, *_ in page.get_images(full=True):
        try:
            ink.extend(page.get_image_rects(xref))
        except Exception:
            continue

    for caption in page_blocks:
        if not caption.is_caption:
            continue
        cx0, cy0, cx1, cy1 = caption.bbox
        is_table = caption.text.lower().lstrip().startswith(("table", "tab."))

        # The search band runs from the caption to the page edge, on the side
        # the artwork sits: above for figures, below for tables.
        if is_table:
            band = pymupdf.Rect(cx0 - 40, cy1, cx1 + 40, page_rect.y1 - 18)
        else:
            band = pymupdf.Rect(cx0 - 40, page_rect.y0 + 18, cx1 + 40, cy0)
        if band.height < 40:
            continue

        # The ink inside that band defines the figure. Driving the bounds from
        # ink rather than from surrounding text matters: a diagram's own labels
        # ("backbone", axis ticks) are text blocks sitting *inside* the figure,
        # and treating them as boundaries slices the figure in half.
        inside = [r for r in ink if band.intersects(r) and r.get_area() > 4]
        if not inside and not is_table:
            continue

        if inside:
            hull = inside[0]
            for r in inside[1:]:
                hull = hull | r
            hull = hull & band
        else:
            hull = band  # a ruled-line-free table: fall back to the band

        # Absorb text blocks lying within the hull -- these are the figure's own
        # labels, and they belong in the crop.
        for b in page_blocks:
            if b is caption or b.is_caption:
                continue
            bb = pymupdf.Rect(b.bbox)
            if (bb & hull).get_area() > 0.5 * bb.get_area():
                hull = hull | bb

        # Safety clip: never cross into body text that sits outside the figure.
        for b in page_blocks:
            if b is caption or b.is_caption:
                continue
            bb = pymupdf.Rect(b.bbox)
            if (bb & hull).get_area() > 0.5 * bb.get_area():
                continue  # already absorbed as a label
            if _horizontal_overlap((hull.x0, hull.y0, hull.x1, hull.y1), b.bbox) < 0.25:
                continue
            if not is_table and bb.y1 <= hull.y0 + 2:
                hull.y0 = max(hull.y0, bb.y1 + 2)
            if is_table and bb.y0 >= hull.y1 - 2:
                hull.y1 = min(hull.y1, bb.y0 - 2)

        rect = pymupdf.Rect(
            max(page_rect.x0, hull.x0 - 8), max(page_rect.y0, hull.y0 - 8),
            min(page_rect.x1, hull.x1 + 8), min(page_rect.y1, hull.y1 + 8),
        )

        if rect.width < 50 or rect.height < 40:
            continue
        if page_area and (rect.width * rect.height) / page_area < 0.008:
            continue
        regions.append((rect, caption.text))

    # Drop regions that substantially overlap an earlier one (a two-part
    # caption, or a subfigure caption sitting under its parent figure).
    kept: list[tuple[pymupdf.Rect, str]] = []
    for rect, cap in sorted(regions, key=lambda rc: -rc[0].get_area()):
        if any((rect & k).get_area() > 0.6 * rect.get_area() for k, _ in kept):
            continue
        kept.append((rect, cap))
    return kept


def parse_pdf(path: str | Path, figures_dir: Path | None = None) -> ParsedDoc:
    path = Path(path)
    figures_dir = Path(figures_dir or settings.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(path)
    title = (doc.metadata or {}).get("title") or path.stem
    parsed = ParsedDoc(source=path.name, title=title.strip() or path.stem,
                       n_pages=doc.page_count)

    current_section = ""
    for pno in range(doc.page_count):
        page = doc[pno]
        page_area = abs(page.rect.width * page.rect.height)
        raw = page.get_text("dict")

        page_blocks: list[TextBlock] = []
        for block in raw.get("blocks", []):
            if block.get("type") != 0:  # 0 = text
                continue
            text = _block_text(block)
            if not text:
                continue
            tb = TextBlock(
                text=text,
                page=pno + 1,
                bbox=tuple(block["bbox"]),
                size=_dominant_size(block),
                is_caption=bool(CAPTION_RE.match(text)),
            )
            page_blocks.append(tb)

        # Section tracking: a heading applies to everything after it until the
        # next one, including figures, which is how a figure inherits context.
        for tb in page_blocks:
            first_line = tb.text.split("\n", 1)[0].strip()
            if not tb.is_caption and len(first_line) < 70 and HEADING_RE.match(first_line):
                current_section = first_line
            tb.section = current_section
        parsed.blocks.extend(page_blocks)

        # --- figures ---------------------------------------------------
        # Primary path: captions anchor the figure, which works for vector and
        # raster alike. Secondary path: large raster images with no caption
        # nearby, which would otherwise be missed entirely.
        candidates: list[tuple[pymupdf.Rect, str]] = caption_anchored_regions(
            page, page_blocks
        )
        claimed = [r for r, _ in candidates]

        for xref, *_ in page.get_images(full=True):
            try:
                rects = page.get_image_rects(xref)
            except Exception:  # malformed xref: skip, do not fail the document
                continue
            for rect in rects:
                if not _keep_image(rect, page_area):
                    continue
                if any((rect & c).get_area() > 0.5 * rect.get_area() for c in claimed):
                    continue
                candidates.append((rect, find_caption(tuple(rect), page_blocks)))

        for rect, caption in candidates:
            fid = hashlib.sha1(
                f"{path.name}:{pno}:{rect.x0:.1f}:{rect.y0:.1f}:"
                f"{rect.x1:.1f}:{rect.y1:.1f}".encode()
            ).hexdigest()[:16]
            out_path = figures_dir / f"{fid}.png"
            if not out_path.exists():
                try:
                    # Render the region rather than pulling the raw xref: axis
                    # labels, arrows and vector overlays drawn on top of a
                    # bitmap belong to the figure and are lost otherwise.
                    page.get_pixmap(clip=rect, dpi=160).save(out_path)
                except Exception:
                    continue
            parsed.figures.append(
                Figure(
                    figure_id=fid,
                    source=path.name,
                    page=pno + 1,
                    bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                    image_path=str(out_path),
                    caption=caption,
                    section=current_section,
                )
            )

    doc.close()
    return parsed
