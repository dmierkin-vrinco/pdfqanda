from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile

from pdfqanda.db import Database
from pdfqanda.ingest import PdfIngestPipeline

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "input" / "sample.pdf"


def test_ingest_populates_markdowns(tmp_path, monkeypatch):
    db_path = tmp_path / "kb.sqlite"
    monkeypatch.setenv("PDFQANDA_DATABASE_URL", f"sqlite:///{db_path}")

    pdf_path = tmp_path / "sample.pdf"
    copyfile(SAMPLE_PATH, pdf_path)

    database = Database(f"sqlite:///{db_path}")
    database.initialize()

    pipeline = PdfIngestPipeline(database)
    artifacts = pipeline.ingest(pdf_path)

    assert artifacts.artifact_path.exists()
    assert artifacts.page_count >= 1

    rows = database.fetch_markdowns()
    assert rows, "expected markdown rows in kb_markdowns"

    embedding = json.loads(rows[0]["emb"])
    assert len(embedding) == 3072
    assert any(value != 0 for value in embedding)

    combined_text = "\n".join(row["content"] for row in rows)
    assert "hybrid retrieval" in combined_text.lower()
