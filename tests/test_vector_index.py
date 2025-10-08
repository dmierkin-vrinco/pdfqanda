from __future__ import annotations

from pdfqanda.util.db import Database
from pdfqanda.util.vector_index import VectorIndex, VectorItem


def test_vector_index_round_trip(tmp_path):
    index_dir = tmp_path / "index"
    index = VectorIndex(index_dir, name="test")
    index.upsert(
        [
            VectorItem(id="a", embedding=[1.0, 0.0], metadata={"document_id": "doc", "section_id": None}),
            VectorItem(id="b", embedding=[0.0, 1.0], metadata={"document_id": "doc", "section_id": None}),
        ]
    )
    hits = index.search([1.0, 0.0], limit=1)
    assert hits and hits[0][0] == "a"

    index.delete(["a"])
    hits = index.search([1.0, 0.0], limit=2)
    assert hits and hits[0][0] == "b"
    index.close()


def test_database_updates_vector_index(tmp_path):
    db_path = tmp_path / "kb.sqlite"
    database = Database(str(db_path))
    database.initialize()

    database.insert_document(doc_id="doc", title="Title", sha256="abc", created_at="now")
    database.insert_sections(
        [
            {
                "id": "sec",
                "document_id": "doc",
                "parent_id": None,
                "title": "Title",
                "level": 1,
                "start_page": 0,
                "end_page": 0,
                "path": "Title",
                "meta": {},
            }
        ]
    )
    database.insert_markdowns(
        [
            {
                "id": "chunk",
                "document_id": "doc",
                "section_id": "sec",
                "content": "hello world",
                "token_count": 2,
                "char_start": 0,
                "char_end": 10,
                "start_page": 0,
                "end_page": 0,
                "emb": [1.0, 0.0],
                "tsv": "hello world",
            }
        ]
    )

    hits = database.vector_search([1.0, 0.0], limit=1)
    assert hits and hits[0]["id"] == "chunk"

    database.delete_document("abc")
    assert database.index.count() == 0
    assert database.vector_search([1.0, 0.0], limit=1) == []
    database.close()


def test_initialize_applies_migrations(tmp_path):
    db_path = tmp_path / "kb.sqlite"
    database = Database(str(db_path))
    database.initialize()

    cursor = database.sqlite_conn.cursor()
    for table in ("kb_tables", "kb_graphics", "kb_notes", "schema_migrations"):
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
            (table,),
        )
        assert cursor.fetchone(), f"{table} table missing"

    cursor.execute("SELECT COUNT(*) FROM schema_migrations")
    applied = cursor.fetchone()[0]
    assert applied >= 2
    database.close()
