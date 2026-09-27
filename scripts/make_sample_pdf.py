"""Generate a small synthetic paper with figures and captions.

The test suite runs against this, so the whole pipeline is testable with no
network, no API key and no copyrighted PDFs in the repository.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


def _diagram(c: canvas.Canvas, x: float, y: float, w: float, h: float, label: str) -> None:
    c.saveState()
    c.setLineWidth(1.2)
    c.rect(x, y, w, h)
    c.line(x + w / 3, y, x + w / 3, y + h)
    c.line(x + 2 * w / 3, y, x + 2 * w / 3, y + h)
    c.setFont("Helvetica-Bold", 9)
    for i, part in enumerate(["backbone", "neck", "head"]):
        c.drawCentredString(x + w * (i + 0.5) / 3, y + h / 2, part)
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(x, y + h + 4, label)
    c.restoreState()


def build(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4

    # page 1
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, height - 2.5 * cm, "A Synthetic Paper on Small Object Detection")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, height - 4 * cm, "Abstract")
    c.setFont("Helvetica", 10)
    text = c.beginText(2 * cm, height - 4.6 * cm)
    for line in [
        "We study the detection of small objects in degraded imagery. Tiled inference",
        "raises recall on objects below twenty pixels, at the cost of increased latency.",
        "We report results on a synthetic benchmark of combusting particles.",
    ]:
        text.textLine(line)
    c.drawText(text)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, height - 7 * cm, "1 Introduction")
    c.setFont("Helvetica", 10)
    text = c.beginText(2 * cm, height - 7.6 * cm)
    for line in [
        "Small object detection fails for two reasons: the object occupies few pixels,",
        "and downsampling in the backbone removes what little signal remains. Slicing",
        "the image before inference preserves effective resolution. The tile overlap",
        "parameter controls duplicate detections at tile borders.",
    ]:
        text.textLine(line)
    c.drawText(text)
    c.showPage()

    # page 2 -- figure with caption below
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, height - 2.5 * cm, "2 Method")
    c.setFont("Helvetica", 10)
    text = c.beginText(2 * cm, height - 3.1 * cm)
    for line in [
        "Our detector follows a three-stage design. Tiles are processed independently",
        "and merged with non-maximum suppression across tile boundaries.",
    ]:
        text.textLine(line)
    c.drawText(text)

    _diagram(c, 2 * cm, height - 9 * cm, 12 * cm, 4 * cm, "")
    c.setFont("Helvetica", 9)
    c.drawString(2 * cm, height - 9.8 * cm,
                 "Figure 1: Overall architecture of the three-stage detector.")
    c.drawString(2 * cm, height - 10.3 * cm,
                 "Tiles enter the backbone independently before merging.")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, height - 12 * cm, "3 Experiments")
    c.setFont("Helvetica", 10)
    text = c.beginText(2 * cm, height - 12.6 * cm)
    for line in [
        "Tiled inference improves recall on the smallest size bucket substantially,",
        "while full-frame inference remains faster. The tracker associates detections",
        "across frames using an intersection-over-union cost and a Kalman filter.",
    ]:
        text.textLine(line)
    c.drawText(text)
    c.showPage()

    # page 3 -- second figure
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, height - 2.5 * cm, "4 Results")
    _diagram(c, 2 * cm, height - 8 * cm, 11 * cm, 3.5 * cm, "")
    c.setFont("Helvetica", 9)
    c.drawString(2 * cm, height - 8.8 * cm,
                 "Figure 2: Tracking pipeline showing association and Kalman update.")
    c.setFont("Helvetica", 10)
    text = c.beginText(2 * cm, height - 10.5 * cm)
    for line in [
        "Velocity is estimated from consecutive track positions and the frame rate.",
        "Size and velocity are negatively correlated in our synthetic benchmark.",
    ]:
        text.textLine(line)
    c.drawText(text)
    c.showPage()
    c.save()
    return path


if __name__ == "__main__":
    out = build(Path(__file__).resolve().parent.parent / "data" / "sample" / "synthetic_paper.pdf")
    print(f"wrote {out}")
