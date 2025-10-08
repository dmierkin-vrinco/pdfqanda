"""Database utilities built around SQLAlchemy with SQLite fallbacks."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence

_SQLALCHEMY_AVAILABLE = importlib.util.find_spec("sqlalchemy") is not None

if _SQLALCHEMY_AVAILABLE:  # pragma: no cover - exercised in integration tests
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import Engine
else:  # pragma: no cover - fallback path tested under CI
    create_engine = text = Engine = None  # type: ignore[assignment]


class Database:
    """Lightweight wrapper exposing the few SQL features the project relies on."""

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
        if not _SQLALCHEMY_AVAILABLE or self.dialect == "sqlite":
            if dsn.startswith("sqlite:///"):
                path = dsn.split("sqlite:///")[1]
            else:
                path = dsn
            self.sqlite_conn = sqlite3.connect(path)
            self.sqlite_conn.row_factory = sqlite3.Row

    # ------------------------------------------------------------------
    def _register_vector(self) -> None:  # pragma: no cover - optional dependency
        spec = importlib.util.find_spec("pgvector.sqlalchemy")
        if spec is None:
            return
        module = importlib.import_module("pgvector.sqlalchemy")
        module.register_vector(self.engine)

    # ------------------------------------------------------------------
    def initialize(self, schema_path: Path | None = None) -> None:
        if self.is_postgres and self.engine is not None:
            if schema_path is None:
                schema_path = Path("schema.sql")
                if not schema_path.exists():
                    schema_path = Path(__file__).resolve().parents[2] / "schema.sql"
            statements = self._load_statements(schema_path)
            with self.engine.begin() as conn:
                for stmt in statements:
                    conn.execute(text(stmt))
        else:
            self._initialize_sqlite()

    def _load_statements(self, schema_path: Path) -> list[str]:
        raw = schema_path.read_text(encoding="utf-8")
        statements: list[str] = []
        buffer: list[str] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Skip DO blocks for non-Postgres engines
            if stripped.upper().startswith("DO "):
                return [raw]  # execute as-is on Postgres
            buffer.append(line)
            if stripped.endswith(";"):
                statements.append("\n".join(buffer))
                buffer = []
        if buffer:
            statements.append("\n".join(buffer))
        return statements

    def _initialize_sqlite(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS kb_documents (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            sha256 TEXT NOT NULL UNIQUE,
            meta TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS kb_sections (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
            parent_id TEXT REFERENCES kb_sections(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            level INTEGER NOT NULL,
            start_page INTEGER NOT NULL,
            end_page INTEGER NOT NULL,
            path TEXT,
            meta TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS kb_markdowns (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
            section_id TEXT REFERENCES kb_sections(id) ON DELETE SET NULL,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            char_start INTEGER,
            char_end INTEGER,
            start_page INTEGER,
            end_page INTEGER,
            emb TEXT NOT NULL,
            tsv TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_sqlite_markdowns_doc
            ON kb_markdowns(document_id);
        """
        if self.engine is not None:
            with self.engine.begin() as conn:  # pragma: no cover - not exercised
                for statement in ddl.split(";\n\n"):
                    stmt = statement.strip()
                    if stmt:
                        conn.execute(text(stmt))
        elif self.sqlite_conn is not None:
            self.sqlite_conn.executescript(ddl)
            self.sqlite_conn.commit()

    # Context manager ------------------------------------------------------
    @contextmanager
    def connect(self):
        if self.engine is None:
            raise RuntimeError("SQLAlchemy engine not available")
        with self.engine.begin() as conn:
            yield conn

    # Mutation helpers -----------------------------------------------------
    def delete_document(self, sha256: str) -> None:
        if self.is_postgres and self.engine is not None:
            with self.engine.begin() as conn:
                conn.execute(text("DELETE FROM kb.markdowns WHERE document_id IN (SELECT id FROM kb.documents WHERE sha256 = :sha)"), {"sha": sha256})
                conn.execute(text("DELETE FROM kb.sections WHERE document_id IN (SELECT id FROM kb.documents WHERE sha256 = :sha)"), {"sha": sha256})
                conn.execute(text("DELETE FROM kb.documents WHERE sha256 = :sha"), {"sha": sha256})
            return
        if self.sqlite_conn is None:
            return
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

    def insert_document(self, *, doc_id: str, title: str, sha256: str, created_at: str, meta: str = "{}") -> None:
        if self.is_postgres and self.engine is not None:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO kb.documents (id, title, sha256, meta, created_at) "
                        "VALUES (:id, :title, :sha, CAST(:meta AS jsonb), :created)"
                    ),
                    {"id": doc_id, "title": title, "sha": sha256, "meta": meta, "created": created_at},
                )
            return
        if self.sqlite_conn is None:
            raise RuntimeError("SQLite connection not configured")
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            "INSERT INTO kb_documents (id, title, sha256, meta, created_at) VALUES (?, ?, ?, ?, ?)",
            (doc_id, title, sha256, meta, created_at),
        )
        self.sqlite_conn.commit()

    def insert_sections(self, rows: Iterable[dict[str, object]]) -> None:
        items = list(rows)
        if not items:
            return
        if self.is_postgres and self.engine is not None:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO kb.sections (id, document_id, parent_id, title, level, start_page, end_page, path, meta) "
                        "VALUES (:id, :document_id, :parent_id, :title, :level, :start_page, :end_page, :path, CAST(:meta AS jsonb))"
                    ),
                    items,
                )
            return
        if self.sqlite_conn is None:
            raise RuntimeError("SQLite connection not configured")
        cursor = self.sqlite_conn.cursor()
        cursor.executemany(
            "INSERT INTO kb_sections (id, document_id, parent_id, title, level, start_page, end_page, path, meta)"
            " VALUES (:id, :document_id, :parent_id, :title, :level, :start_page, :end_page, :path, :meta)",
            [
                {
                    "id": row.get("id"),
                    "document_id": row.get("document_id"),
                    "parent_id": row.get("parent_id"),
                    "title": row.get("title"),
                    "level": row.get("level"),
                    "start_page": row.get("start_page"),
                    "end_page": row.get("end_page"),
                    "path": row.get("path"),
                    "meta": json.dumps(row.get("meta", {})),
                }
                for row in items
            ],
        )
        self.sqlite_conn.commit()

    def insert_markdowns(self, rows: Iterable[dict[str, object]]) -> None:
        items = list(rows)
        if not items:
            return
        if self.is_postgres and self.engine is not None:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO kb.markdowns (id, document_id, section_id, content, token_count, char_start, char_end, start_page, end_page, emb, tsv) "
                        "VALUES (:id, :document_id, :section_id, :content, :token_count, :char_start, :char_end, :start_page, :end_page, :emb, to_tsvector('english', :content))"
                    ),
                    items,
                )
            return
        if self.sqlite_conn is None:
            raise RuntimeError("SQLite connection not configured")
        cursor = self.sqlite_conn.cursor()
        cursor.executemany(
            "INSERT INTO kb_markdowns (id, document_id, section_id, content, token_count, char_start, char_end, start_page, end_page, emb, tsv)"
            " VALUES (:id, :document_id, :section_id, :content, :token_count, :char_start, :char_end, :start_page, :end_page, :emb, :tsv)",
            [
                {
                    "id": row.get("id"),
                    "document_id": row.get("document_id"),
                    "section_id": row.get("section_id"),
                    "content": row.get("content"),
                    "token_count": row.get("token_count"),
                    "char_start": row.get("char_start"),
                    "char_end": row.get("char_end"),
                    "start_page": row.get("start_page"),
                    "end_page": row.get("end_page"),
                    "emb": json.dumps(row.get("emb")),
                    "tsv": row.get("tsv") or self._build_tsv(row.get("content", "")),
                }
                for row in items
            ],
        )
        self.sqlite_conn.commit()

    # Query helpers --------------------------------------------------------
    def fetch_sections(self, document_id: str) -> dict[str, dict[str, object]]:
        if self.is_postgres and self.engine is not None:
            with self.engine.begin() as conn:
                result = conn.execute(
                    text("SELECT * FROM kb.sections WHERE document_id = :doc"),
                    {"doc": document_id},
                )
                rows = result.mappings().all()
        else:
            if self.sqlite_conn is None:
                return {}
            cursor = self.sqlite_conn.cursor()
            cursor.execute("SELECT * FROM kb_sections WHERE document_id = ?", (document_id,))
            rows = cursor.fetchall()
        return {str(row["id"]): dict(row) for row in rows}

    def vector_search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
        keywords: Sequence[str] | None = None,
    ) -> list[dict[str, object]]:
        if self.is_postgres and self.engine is not None:
            clauses = []
            params: dict[str, object] = {"embedding": embedding, "limit": limit}
            sql = (
                "SELECT id, document_id, section_id, content, start_page, end_page, token_count, "
                "1 - (emb <#> :embedding) AS score "
                "FROM kb.markdowns"
            )
            if keywords:
                clauses.append("tsv @@ websearch_to_tsquery('english', :ts_query)")
                params["ts_query"] = " ".join(keywords)
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            sql += " ORDER BY emb <#> :embedding LIMIT :limit"
            with self.engine.begin() as conn:
                result = conn.execute(text(sql), params)
                rows = [dict(row) for row in result.mappings().all()]
            return rows
        # SQLite fallback -------------------------------------------------
        if self.sqlite_conn is None:
            return []
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            "SELECT id, document_id, section_id, content, start_page, end_page, token_count, emb, tsv FROM kb_markdowns"
        )
        rows = cursor.fetchall()
        filtered = []
        keyword_list = [kw.lower() for kw in keywords or []]
        for row in rows:
            content = row["content"]
            if keyword_list and not any(kw in content.lower() for kw in keyword_list):
                continue
            vector = json.loads(row["emb"])
            score = self._cosine_similarity(embedding, vector)
            filtered.append(
                {
                    "id": row["id"],
                    "document_id": row["document_id"],
                    "section_id": row["section_id"],
                    "content": content,
                    "start_page": row["start_page"],
                    "end_page": row["end_page"],
                    "token_count": row["token_count"],
                    "score": score,
                }
            )
        filtered.sort(key=lambda item: item["score"], reverse=True)
        return filtered[:limit]

    # Utilities ------------------------------------------------------------
    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) != len(b):
            raise ValueError("Embedding dimension mismatch")
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _build_tsv(text_value: str) -> str:
        return " ".join(part.lower() for part in text_value.split())
