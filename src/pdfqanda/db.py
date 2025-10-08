"""Database utilities for pdfqanda."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

_SQLALCHEMY_AVAILABLE = importlib.util.find_spec("sqlalchemy") is not None

if _SQLALCHEMY_AVAILABLE:
    from sqlalchemy import bindparam, create_engine, text
    from sqlalchemy.engine import Engine
else:  # pragma: no cover - exercised in CI fallback
    bindparam = create_engine = text = Engine = None  # type: ignore[assignment]


class Database:
    """Lightweight wrapper around a SQLAlchemy engine."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.engine: Engine | None = None
        self.sqlite_conn: sqlite3.Connection | None = None
        if _SQLALCHEMY_AVAILABLE:
            self.engine = create_engine(dsn, future=True)
            self.dialect = self.engine.dialect.name
            self.is_postgres = self.dialect.startswith("postgres")
            if self.is_postgres:
                self._register_vector()
        else:
            self.dialect = "sqlite"
            self.is_postgres = False
            if not dsn.startswith("sqlite:///"):
                msg = "SQLAlchemy is required for non-SQLite connections"
                raise ModuleNotFoundError(msg)
            path = dsn.split("sqlite:///")[1]
            self.sqlite_conn = sqlite3.connect(path)
            self.sqlite_conn.row_factory = sqlite3.Row

    def _register_vector(self) -> None:
        spec = importlib.util.find_spec("pgvector.sqlalchemy")
        if spec is None:
            return
        module = importlib.import_module("pgvector.sqlalchemy")
        module.register_vector(self.engine)

    def initialize(self, schema_path: Path | None = None) -> None:
        """Initialise database schemas and tables."""

        if self.is_postgres:
            if schema_path is None:
                schema_path = Path(__file__).resolve().parents[1] / "db" / "schema.sql"
            schema_sql = schema_path.read_text()
            if not self.engine:
                raise RuntimeError("Engine not available")
            with self.engine.begin() as conn:
                conn.execute(text(schema_sql))
        else:
            self._initialize_sqlite()

    def _initialize_sqlite(self) -> None:
        """Create a minimal SQLite schema for development and tests."""

        ddl = """
        CREATE TABLE IF NOT EXISTS kb_documents (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            sha256 TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS kb_sections (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            level INTEGER NOT NULL,
            start_page INTEGER,
            end_page INTEGER
        );

        CREATE TABLE IF NOT EXISTS kb_markdowns (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
            section_id TEXT REFERENCES kb_sections(id) ON DELETE SET NULL,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            start_page INTEGER,
            end_page INTEGER,
            emb TEXT,
            tsv TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_sqlite_markdowns_doc
            ON kb_markdowns(document_id);
        """
        if self.engine is not None:
            with self.engine.begin() as conn:
                for statement in ddl.split(";\n\n"):
                    stmt = statement.strip()
                    if stmt:
                        conn.execute(text(stmt))
        elif self.sqlite_conn is not None:
            self.sqlite_conn.executescript(ddl)
            self.sqlite_conn.commit()

    @contextmanager
    def connect(self):
        if self.engine is None:
            raise RuntimeError("SQLAlchemy engine is not available in SQLite fallback mode")
        with self.engine.begin() as conn:
            yield conn

    # -- Ingestion helpers -------------------------------------------------
    def delete_document(self, sha256: str) -> None:
        """Delete an existing document by hash."""

        if self.engine is not None:
            with self.engine.begin() as conn:
                if self.is_postgres:
                    conn.execute(text("DELETE FROM kb.markdowns WHERE document_id IN (SELECT id FROM kb.documents WHERE sha256 = :sha)"), {"sha": sha256})
                    conn.execute(text("DELETE FROM kb.sections WHERE document_id IN (SELECT id FROM kb.documents WHERE sha256 = :sha)"), {"sha": sha256})
                    conn.execute(text("DELETE FROM kb.documents WHERE sha256 = :sha"), {"sha": sha256})
                else:
                    conn.execute(text("DELETE FROM kb_markdowns WHERE document_id IN (SELECT id FROM kb_documents WHERE sha256 = :sha)"), {"sha": sha256})
                    conn.execute(text("DELETE FROM kb_sections WHERE document_id IN (SELECT id FROM kb_documents WHERE sha256 = :sha)"), {"sha": sha256})
                    conn.execute(text("DELETE FROM kb_documents WHERE sha256 = :sha"), {"sha": sha256})
        elif self.sqlite_conn is not None:
            cursor = self.sqlite_conn.cursor()
            cursor.execute(
                "DELETE FROM kb_markdowns WHERE document_id IN (SELECT id FROM kb_documents WHERE sha256 = ?)",
                (sha256,),
            )
            cursor.execute(
                "DELETE FROM kb_sections WHERE document_id IN (SELECT id FROM kb_documents WHERE sha256 = ?)",
                (sha256,),
            )
            cursor.execute("DELETE FROM kb_documents WHERE sha256 = ?", (sha256,))
            self.sqlite_conn.commit()

    def insert_document(self, *, doc_id: str, title: str, sha256: str, created_at: str) -> None:
        """Insert a new document row."""

        if self.engine is not None:
            with self.engine.begin() as conn:
                if self.is_postgres:
                    conn.execute(
                        text(
                            "INSERT INTO kb.documents (id, title, sha256, created_at) "
                            "VALUES (:id, :title, :sha, :created)"
                        ),
                        {"id": doc_id, "title": title, "sha": sha256, "created": created_at},
                    )
                else:
                    conn.execute(
                        text(
                            "INSERT INTO kb_documents (id, title, sha256, created_at) "
                            "VALUES (:id, :title, :sha, :created)"
                        ),
                        {"id": doc_id, "title": title, "sha": sha256, "created": created_at},
                    )
        elif self.sqlite_conn is not None:
            cursor = self.sqlite_conn.cursor()
            cursor.execute(
                "INSERT INTO kb_documents (id, title, sha256, created_at) VALUES (?, ?, ?, ?)",
                (doc_id, title, sha256, created_at),
            )
            self.sqlite_conn.commit()

    def insert_sections(self, rows: Iterable[dict[str, object]]) -> None:
        rows_list = list(rows)
        if not rows_list:
            return
        if self.engine is not None:
            with self.engine.begin() as conn:
                if self.is_postgres:
                    conn.execute(
                        text(
                            "INSERT INTO kb.sections (id, document_id, parent_id, title, level, start_page, end_page, path) "
                            "VALUES (:id, :document_id, :parent_id, :title, :level, :start_page, :end_page, :path)"
                        ),
                        rows_list,
                    )
                else:
                    sqlite_rows = []
                    for row in rows_list:
                        sqlite_rows.append(
                            {
                                "id": row["id"],
                                "document_id": row["document_id"],
                                "title": row.get("title"),
                                "level": row.get("level", 0),
                                "start_page": row.get("start_page"),
                                "end_page": row.get("end_page"),
                            }
                        )
                    conn.execute(
                        text(
                            "INSERT INTO kb_sections (id, document_id, title, level, start_page, end_page) "
                            "VALUES (:id, :document_id, :title, :level, :start_page, :end_page)"
                        ),
                        sqlite_rows,
                    )
        elif self.sqlite_conn is not None:
            cursor = self.sqlite_conn.cursor()
            cursor.executemany(
                "INSERT INTO kb_sections (id, document_id, title, level, start_page, end_page) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        row["id"],
                        row["document_id"],
                        row.get("title"),
                        row.get("level", 0),
                        row.get("start_page"),
                        row.get("end_page"),
                    )
                    for row in rows_list
                ],
            )
            self.sqlite_conn.commit()

    def insert_markdowns(self, rows: Iterable[dict[str, object]]) -> None:
        prepared = []
        for row in rows:
            if self.is_postgres:
                prepared.append(row)
            else:
                sqlite_row = row.copy()
                emb = sqlite_row.get("emb")
                if emb is not None:
                    sqlite_row["emb"] = json.dumps(emb)
                sqlite_row["tsv"] = sqlite_row.get("tsv") or ""
                prepared.append(sqlite_row)
        if not prepared:
            return
        if self.engine is not None:
            with self.engine.begin() as conn:
                if self.is_postgres:
                    conn.execute(
                        text(
                            "INSERT INTO kb.markdowns (id, document_id, section_id, content, token_count, start_page, end_page, emb) "
                            "VALUES (:id, :document_id, :section_id, :content, :token_count, :start_page, :end_page, :emb)"
                        ),
                        prepared,
                    )
                else:
                    conn.execute(
                        text(
                            "INSERT INTO kb_markdowns (id, document_id, section_id, content, token_count, start_page, end_page, emb, tsv) "
                            "VALUES (:id, :document_id, :section_id, :content, :token_count, :start_page, :end_page, :emb, :tsv)"
                        ),
                        prepared,
                    )
        elif self.sqlite_conn is not None:
            cursor = self.sqlite_conn.cursor()
            cursor.executemany(
                "INSERT INTO kb_markdowns (id, document_id, section_id, content, token_count, start_page, end_page, emb, tsv) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        row["id"],
                        row["document_id"],
                        row.get("section_id"),
                        row.get("content"),
                        row.get("token_count"),
                        row.get("start_page"),
                        row.get("end_page"),
                        row.get("emb"),
                        row.get("tsv", ""),
                    )
                    for row in prepared
                ],
            )
            self.sqlite_conn.commit()

    # -- Retrieval helpers -------------------------------------------------
    def fetch_all_markdowns(self) -> list[dict[str, object]]:
        if self.engine is not None:
            query = text(
                "SELECT id, document_id, section_id, content, token_count, start_page, end_page, emb, tsv "
                + ("FROM kb.markdowns" if self.is_postgres else "FROM kb_markdowns")
            )
            with self.engine.connect() as conn:
                rows = conn.execute(query).mappings().all()
            result = []
            for row in rows:
                mapping = dict(row)
                if not self.is_postgres and isinstance(mapping.get("emb"), str):
                    mapping["emb"] = json.loads(mapping["emb"])
                result.append(mapping)
            return result
        if self.sqlite_conn is not None:
            cursor = self.sqlite_conn.cursor()
            cursor.execute(
                "SELECT id, document_id, section_id, content, token_count, start_page, end_page, emb, tsv FROM kb_markdowns"
            )
            rows = cursor.fetchall()
            result = []
            for row in rows:
                mapping = dict(row)
                emb = mapping.get("emb")
                if isinstance(emb, str):
                    mapping["emb"] = json.loads(emb)
                result.append(mapping)
            return result
        return []

    def vector_search(self, query_vector: list[float], limit: int) -> list[dict[str, object]]:
        if self.is_postgres:
            sql = text(
                "SELECT id, document_id, section_id, content, start_page, end_page, "
                "emb <=> :query AS distance "
                "FROM kb.markdowns WHERE emb IS NOT NULL ORDER BY emb <=> :query LIMIT :limit"
            )
            with self.engine.connect() as conn:
                rows = conn.execute(sql, {"query": query_vector, "limit": limit}).mappings().all()
            return [dict(row) for row in rows]

        # SQLite fallback: brute-force cosine similarity
        rows = self.fetch_all_markdowns()
        scored: list[tuple[float, dict[str, object]]] = []
        for row in rows:
            embedding = row.get("emb") or []
            if not embedding:
                continue
            score = cosine_similarity(query_vector, embedding)
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [dict(row) | {"distance": 1 - score} for score, row in scored[:limit]]

    def fts_refine(self, query: str, candidate_ids: list[str]) -> dict[str, float]:
        if not candidate_ids or not self.is_postgres:
            return {}
        sql = (
            text(
                "SELECT id, ts_rank(tsv, plainto_tsquery('english', :query)) AS rank "
                "FROM kb.markdowns WHERE id = ANY(:ids)"
            )
            .bindparams(bindparam("ids", expanding=True))
        )
        with self.engine.connect() as conn:
            rows = conn.execute(sql, {"query": query, "ids": candidate_ids}).mappings().all()
        return {row["id"]: float(row["rank"]) for row in rows}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a ** 0.5 * norm_b ** 0.5)
