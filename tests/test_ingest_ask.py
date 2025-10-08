from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import pytest

from pdfqanda.config import get_settings
from pdfqanda.ingest import PdfIngestor
from pdfqanda.retrieval import Retriever, format_answer
from pdfqanda.util.cache import FileCache, stable_hash
from pdfqanda.util.db import Database
from pdfqanda.util.embeddings import EmbeddingClient

SAMPLE = Path(__file__).resolve().parents[1] / "input" / "sample.pdf"


@pytest.fixture()
def openai_embedder(tmp_path):
    settings = get_settings()
    cache = FileCache(tmp_path / "llm")
    return EmbeddingClient(
        settings.embedding_model,
        settings.embedding_dim,
        cache=cache,
    )


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "kb.sqlite"
    monkeypatch.setenv("DB_PATH", str(db_path))
    get_settings.cache_clear()
    settings = get_settings()
    database = Database(settings.db_path)
    database.initialize()
    yield database
    database.close()
    get_settings.cache_clear()


def test_ingest_and_ask(temp_db, tmp_path, openai_embedder):
    database = temp_db
    local_pdf = tmp_path / "sample.pdf"
    copyfile(SAMPLE, local_pdf)

    ingestor = PdfIngestor(database, embedder=openai_embedder)
    result = ingestor.ingest(local_pdf)
    assert result.chunk_count > 0

    # ensure rows were written
    cursor = database.sqlite_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM kb_markdowns")
    assert cursor.fetchone()[0] > 0

    retriever = Retriever(database, embedder=openai_embedder)
    hits = retriever.search("What is the project about?", k=3)
    answer = format_answer(hits)
    assert hits
    assert "【doc:" in answer
    # ensure index files created
    index_dir = Path(database.path).with_name(Path(database.path).name + ".index")
    assert index_dir.exists()
    tables_cache = Path(".cache/tables")
    assert tables_cache.exists()
    cached_layout = ingestor.table_cache.get(
        "layouts",
        stable_hash([result.sha256, "sections:v1"]),
    )
    assert cached_layout

    second = ingestor.ingest(local_pdf)
    assert second.chunk_count == result.chunk_count

def test_embedding_dimension(openai_embedder):
    vector = openai_embedder.embed_query("hello world")
    assert len(vector) == 1536
