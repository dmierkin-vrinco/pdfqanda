from __future__ import annotations

from pathlib import Path
from shutil import copyfile

from pdfqanda.config import get_settings
from pdfqanda.db import Database
from pdfqanda.ingest import PdfIngestor
from pdfqanda.retrieval import Retriever

SAMPLE = Path(__file__).resolve().parents[1] / "input" / "sample.pdf"


def test_ingest_and_ask(tmp_path, monkeypatch):
    db_path = tmp_path / "kb.sqlite"
    monkeypatch.setenv("DB_DSN", f"sqlite:///{db_path}")
    get_settings.cache_clear()

    local_pdf = tmp_path / "sample.pdf"
    copyfile(SAMPLE, local_pdf)

    settings = get_settings()
    database = Database(settings.db_dsn)
    database.initialize()

    ingestor = PdfIngestor(database)
    result = ingestor.ingest(local_pdf)
    assert result.chunk_count > 0

    retriever = Retriever(database)
    hits = retriever.search("What is the project about?", k=3)
    assert hits
    assert all("【" in hit.citation for hit in hits)
