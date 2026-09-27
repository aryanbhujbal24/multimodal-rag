from app.ingest.parser import CAPTION_RE, HEADING_RE, find_caption


def test_caption_regex_matches_common_forms():
    for text in ["Figure 1: Overall architecture", "Fig. 2. Results",
                 "Table 3 - Ablation", "ALGORITHM 1 Association"]:
        assert CAPTION_RE.match(text), text


def test_caption_regex_rejects_prose():
    for text in ["The figure shows a detector.", "In Figure 1 we see",
                 "Figures are useful"]:
        assert not CAPTION_RE.match(text), text


def test_heading_regex():
    assert HEADING_RE.match("3.1 Detection")
    assert HEADING_RE.match("Abstract")
    assert HEADING_RE.match("IV. Experiments")
    assert not HEADING_RE.match("We propose a method that works well.")


def test_figures_extracted_with_captions(parsed):
    assert len(parsed.figures) == 2
    captions = " ".join(f.caption for f in parsed.figures)
    assert "Overall architecture" in captions
    assert "Tracking pipeline" in captions


def test_figure_crops_written_and_non_trivial(parsed):
    from pathlib import Path

    for fig in parsed.figures:
        path = Path(fig.image_path)
        assert path.exists()
        assert path.stat().st_size > 1000  # not a blank sliver
        assert fig.bbox[3] - fig.bbox[1] > 40


def test_figure_pages_are_one_based(parsed):
    assert {f.page for f in parsed.figures} == {2, 3}


def test_caption_not_stolen_across_columns():
    """A caption far to the side must not be claimed by an unrelated figure."""
    from app.ingest.parser import TextBlock

    blocks = [TextBlock(text="Figure 9: unrelated", page=1,
                        bbox=(400, 100, 560, 115), size=9, is_caption=True)]
    assert find_caption((40, 60, 200, 95), blocks) == ""
