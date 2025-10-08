from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from pdfqanda.ingest import build_document_artifact, compute_doc_hash


def _create_sample_pdf(path: Path) -> None:
    doc = fitz.open()
    try:
        page = doc.new_page(width=400, height=400)
        page.insert_text((40, 60), "Sample page with an image^1")
        rect = fitz.Rect(120, 140, 220, 240)
        page.draw_rect(rect, color=(0.1, 0.3, 0.8), fill=(0.1, 0.3, 0.8))
        page.insert_text((40, 320), "References")
        page.insert_text((40, 350), "1. This is a sample footnote.")
        doc.save(path)
    finally:
        doc.close()


def test_notes_and_graphics_smoke(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _create_sample_pdf(pdf_path)

    doc_hash, artifact_path, pages = build_document_artifact(pdf_path)
    assert artifact_path.exists()
    assert doc_hash == compute_doc_hash(pdf_path)
    assert pages, "Expected at least one page to be extracted"

    artifact_dir = Path("artifacts") / doc_hash
    graphics_dir = artifact_dir / "graphics"
    assert graphics_dir.exists()

    artifact_data = json.loads(artifact_path.read_text())
    assert artifact_data["doc_hash"] == doc_hash
    has_notes = any(page["notes"] for page in artifact_data["pages"])
    has_graphics = any(page["graphics"] for page in artifact_data["pages"])
    assert has_notes or has_graphics

    png_files = list(graphics_dir.glob("*.png"))
    if has_graphics:
        assert png_files, "Expected rasterized graphics to be saved"
    for png in png_files:
        assert png.exists()

    for page in artifact_data["pages"]:
        for bbox in page["bbox_blocks"]:
            for value in bbox:
                assert 0.0 <= value <= 1.0
        for note in page["notes"]:
            for value in note["bbox"]:
                assert 0.0 <= value <= 1.0
        for graphic in page["graphics"]:
            for value in graphic["bbox"]:
                assert 0.0 <= value <= 1.0

    # Clean up generated artifacts to keep the workspace tidy
    shutil.rmtree(artifact_dir)
    artifact_path.unlink()
    cache_dir = Path(".cache/pdf") / doc_hash
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
