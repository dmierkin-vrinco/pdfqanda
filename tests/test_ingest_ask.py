from __future__ import annotations

import os
from pathlib import Path
from shutil import copyfile

import pytest

from pdfqanda.config import get_settings
from pdfqanda.ingest import PdfIngestor
from pdfqanda.retrieval import Retriever, format_answer
from pdfqanda.util.db import Database
from pdfqanda.util.embeddings import EmbeddingClient

SAMPLE = Path(__file__).resolve().parents[1] / "input" / "sample.pdf"


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "kb.sqlite"
    monkeypatch.setenv("DB_DSN", f"sqlite:///{db_path}")
    get_settings.cache_clear()
    settings = get_settings()
    database = Database(settings.db_dsn)
    database.initialize()
    return database


@pytest.fixture(scope="module")
def postgres_db():
    dsn = os.getenv(
        "POSTGRES_TEST_DSN",
        "postgresql+psycopg://pdfqanda:pdfqanda@localhost:5432/pdfqanda",
    )
    get_settings.cache_clear()
    try:
        database = Database(dsn)
    except Exception as exc:  # pragma: no cover - optional dependency path
        pytest.skip(f"Postgres unavailable: {exc}")
    if not database.is_postgres:
        pytest.skip("Postgres driver not available")

    # ensure a clean schema for the test run
    with database.engine.begin() as conn:  # type: ignore[union-attr]
        conn.exec_driver_sql("DROP SCHEMA IF EXISTS kb CASCADE;")
        conn.exec_driver_sql("DROP SCHEMA IF EXISTS pdf_tables CASCADE;")
    database.initialize(Path(__file__).resolve().parents[1] / "schema.sql")
    yield database
    with database.engine.begin() as conn:  # type: ignore[union-attr]
        conn.exec_driver_sql("DROP SCHEMA IF EXISTS kb CASCADE;")
        conn.exec_driver_sql("DROP SCHEMA IF EXISTS pdf_tables CASCADE;")


def test_ingest_and_ask(temp_db, tmp_path):
    database = temp_db
    local_pdf = tmp_path / "sample.pdf"
    copyfile(SAMPLE, local_pdf)

    ingestor = PdfIngestor(database)
    result = ingestor.ingest(local_pdf)
    assert result.chunk_count > 0

    # ensure rows were written
    cursor = database.sqlite_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM kb_markdowns")
    assert cursor.fetchone()[0] > 0

    retriever = Retriever(database)
    hits = retriever.search("What is the project about?", k=3)
    answer = format_answer(hits)
    assert hits
    assert "【doc:" in answer


def test_ingest_and_ask_postgres(postgres_db, tmp_path):
    database = postgres_db
    local_pdf = tmp_path / "sample.pdf"
    copyfile(SAMPLE, local_pdf)

    ingestor = PdfIngestor(database)
    result = ingestor.ingest(local_pdf)
    assert result.chunk_count > 0

    retriever = Retriever(database)
    hits = retriever.search("What is the project about?", k=3)
    answer = format_answer(hits)
    assert hits
    assert "【doc:" in answer

def test_embedding_dimension():
    get_settings.cache_clear()
    settings = get_settings()
    client = EmbeddingClient(settings.embedding_model, settings.embedding_dim)
    vector = client.embed_query("hello world")
    assert len(vector) == 3072
