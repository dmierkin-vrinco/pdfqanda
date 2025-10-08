"""Persistence layer for pdfqanda."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .models import (
    DocumentRecord,
    GraphicRecord,
    MarkdownChunk,
    NoteRecord,
    SectionRecord,
    TableMetadataRecord,
)


@dataclass(slots=True)
class Database:
    """Lightweight database wrapper supporting SQLite and Postgres."""

    url: str
    _conn: sqlite3.Connection | None = None
    _backend: str = "sqlite"

    def __post_init__(self) -> None:
        if self.url.startswith("postgres://") or self.url.startswith("postgresql://"):
            self._backend = "postgres"
        elif self.url.startswith("sqlite://"):
            self._backend = "sqlite"
        else:
            msg = f"Unsupported database URL: {self.url}"
            raise ValueError(msg)

    # -- Connection management -------------------------------------------------
    def connect(self):  # type: ignore[override]
        if self._conn is not None:
            return self._conn
        if self._backend == "sqlite":
            db_path = self.url.split("sqlite://", 1)[1]
            if db_path.startswith("/"):
                path = Path(db_path)
            elif db_path == ":memory:":
                path = Path(db_path)
            else:
                path = Path.cwd() / db_path
            if str(path) != ":memory":
                path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(path))
            conn.execute("PRAGMA foreign_keys = ON")
            conn.row_factory = sqlite3.Row
            self._conn = conn
            return conn
        import importlib

        psycopg = importlib.import_module("psycopg")
        self._conn = psycopg.connect(self.url, autocommit=False)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- Schema management -----------------------------------------------------
    def initialize(self) -> None:
        if self._backend == "sqlite":
            self._initialize_sqlite()
        else:
            self._initialize_postgres()

    def _initialize_sqlite(self) -> None:
        conn = self.connect()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS kb_documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                sha256 TEXT NOT NULL UNIQUE,
                meta TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS kb_sections (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                parent_id TEXT,
                title TEXT NOT NULL,
                level INTEGER NOT NULL,
                start_page INTEGER NOT NULL,
                end_page INTEGER NOT NULL,
                path TEXT,
                meta TEXT,
                FOREIGN KEY(document_id) REFERENCES kb_documents(id) ON DELETE CASCADE,
                FOREIGN KEY(parent_id) REFERENCES kb_sections(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS kb_markdowns (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                section_id TEXT,
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                char_start INTEGER,
                char_end INTEGER,
                start_page INTEGER,
                end_page INTEGER,
                start_line INTEGER,
                end_line INTEGER,
                emb TEXT NOT NULL,
                tsv TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES kb_documents(id) ON DELETE CASCADE,
                FOREIGN KEY(section_id) REFERENCES kb_sections(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS kb_notes (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                section_id TEXT,
                kind TEXT NOT NULL,
                ref_anchor TEXT,
                content TEXT NOT NULL,
                page INTEGER,
                bbox TEXT,
                FOREIGN KEY(document_id) REFERENCES kb_documents(id) ON DELETE CASCADE,
                FOREIGN KEY(section_id) REFERENCES kb_sections(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS kb_graphics (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                section_id TEXT,
                caption TEXT,
                nearby_text TEXT,
                path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                page INTEGER,
                bbox TEXT,
                FOREIGN KEY(document_id) REFERENCES kb_documents(id) ON DELETE CASCADE,
                FOREIGN KEY(section_id) REFERENCES kb_sections(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS kb_tables_metadata (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                section_id TEXT,
                table_name TEXT NOT NULL,
                caption TEXT,
                columns_json TEXT,
                units_json TEXT,
                FOREIGN KEY(document_id) REFERENCES kb_documents(id) ON DELETE CASCADE,
                FOREIGN KEY(section_id) REFERENCES kb_sections(id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_markdowns_document ON kb_markdowns(document_id);
            CREATE INDEX IF NOT EXISTS idx_markdowns_section ON kb_markdowns(section_id);
            CREATE INDEX IF NOT EXISTS idx_markdowns_tsv ON kb_markdowns(tsv);
            """
        )
        conn.commit()

    def _initialize_postgres(self) -> None:
        schema_path = Path(__file__).resolve().parent.parent / "schema.sql"
        if not schema_path.exists():
            msg = "schema.sql not found; cannot initialize Postgres schema"
            raise FileNotFoundError(msg)
        conn = self.connect()
        sql = schema_path.read_text()
        statements = [stmt.strip() for stmt in sql.split(";") if stmt.strip()]
        cur = conn.cursor()
        try:
            for stmt in statements:
                cur.execute(stmt)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    # -- Persistence helpers ---------------------------------------------------
    @contextmanager
    def transaction(self):
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def insert_document_bundle(
        self,
        document: DocumentRecord,
        sections: Iterable[SectionRecord],
        markdowns: Iterable[MarkdownChunk],
        notes: Iterable[NoteRecord],
        graphics: Iterable[GraphicRecord],
        tables: Iterable[TableMetadataRecord],
    ) -> None:
        with self.transaction() as conn:
            self._insert_document(conn, document)
            self._insert_sections(conn, sections)
            self._insert_markdowns(conn, markdowns)
            self._insert_notes(conn, notes)
            self._insert_graphics(conn, graphics)
            self._insert_tables(conn, tables)

    # -- Low level insert helpers ---------------------------------------------
    def _insert_document(self, conn, document: DocumentRecord) -> None:
        if self._backend == "sqlite":
            conn.execute(
                "INSERT OR REPLACE INTO kb_documents (id, title, sha256, meta, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    document.id,
                    document.title,
                    document.sha256,
                    json.dumps(document.meta),
                    document.created_at.isoformat(),
                ),
            )
        else:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO kb.documents (id, title, sha256, meta, created_at)"
                " VALUES (%s, %s, %s, %s::jsonb, %s)"
                " ON CONFLICT (sha256) DO UPDATE SET title = EXCLUDED.title",
                (
                    document.id,
                    document.title,
                    document.sha256,
                    json.dumps(document.meta),
                    document.created_at,
                ),
            )
            cur.close()

    def _insert_sections(self, conn, sections: Iterable[SectionRecord]) -> None:
        for section in sections:
            payload = (
                section.id,
                section.document_id,
                section.parent_id,
                section.title,
                section.level,
                section.start_page,
                section.end_page,
                section.path,
                json.dumps(section.meta),
            )
            if self._backend == "sqlite":
                conn.execute(
                    "INSERT OR REPLACE INTO kb_sections "
                    "(id, document_id, parent_id, title, level, start_page, end_page, path, meta)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    payload,
                )
            else:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO kb.sections "
                    "(id, document_id, parent_id, title, level, start_page, end_page, path, meta)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)"
                    " ON CONFLICT (id) DO NOTHING",
                    payload,
                )
                cur.close()

    def _insert_markdowns(self, conn, markdowns: Iterable[MarkdownChunk]) -> None:
        for chunk in markdowns:
            emb_json = json.dumps(list(chunk.embedding))
            payload = (
                chunk.id,
                chunk.document_id,
                chunk.section_id,
                chunk.content,
                chunk.token_count,
                chunk.char_start,
                chunk.char_end,
                chunk.start_page,
                chunk.end_page,
                chunk.start_line,
                chunk.end_line,
                emb_json,
                chunk.tsv,
            )
            if self._backend == "sqlite":
                conn.execute(
                    "INSERT OR REPLACE INTO kb_markdowns "
                    "(id, document_id, section_id, content, token_count, char_start, char_end, "
                    " start_page, end_page, start_line, end_line, emb, tsv)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    payload,
                )
            else:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO kb.markdowns "
                    "(id, document_id, section_id, content, token_count, char_start, char_end,"
                    " start_page, end_page, start_line, end_line, emb, tsv)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, to_tsvector(%s))",
                    payload[:-1] + (chunk.tsv,),
                )
                cur.close()

    def _insert_notes(self, conn, notes: Iterable[NoteRecord]) -> None:
        for note in notes:
            payload = (
                note.id,
                note.document_id,
                note.section_id,
                note.kind,
                note.ref_anchor,
                note.content,
                note.page,
                json.dumps(note.bbox) if note.bbox else None,
            )
            if self._backend == "sqlite":
                conn.execute(
                    "INSERT OR REPLACE INTO kb_notes "
                    "(id, document_id, section_id, kind, ref_anchor, content, page, bbox)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    payload,
                )
            else:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO kb.notes "
                    "(id, document_id, section_id, kind, ref_anchor, content, page, bbox)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
                    payload,
                )
                cur.close()

    def _insert_graphics(self, conn, graphics: Iterable[GraphicRecord]) -> None:
        for graphic in graphics:
            payload = (
                graphic.id,
                graphic.document_id,
                graphic.section_id,
                graphic.caption,
                graphic.nearby_text,
                graphic.path,
                graphic.sha256,
                graphic.page,
                json.dumps(graphic.bbox) if graphic.bbox else None,
            )
            if self._backend == "sqlite":
                conn.execute(
                    "INSERT OR REPLACE INTO kb_graphics "
                    "(id, document_id, section_id, caption, nearby_text, path, sha256, page, bbox)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    payload,
                )
            else:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO kb.graphics "
                    "(id, document_id, section_id, caption, nearby_text, path, sha256, page, bbox)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)"
                    " ON CONFLICT (id) DO NOTHING",
                    payload,
                )
                cur.close()

    def _insert_tables(self, conn, tables: Iterable[TableMetadataRecord]) -> None:
        for table in tables:
            payload = (
                table.id,
                table.document_id,
                table.section_id,
                table.table_name,
                table.caption,
                json.dumps(table.columns_json) if table.columns_json is not None else None,
                json.dumps(table.units_json) if table.units_json is not None else None,
            )
            if self._backend == "sqlite":
                conn.execute(
                    "INSERT OR REPLACE INTO kb_tables_metadata "
                    "(id, document_id, section_id, table_name, caption, columns_json, units_json)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    payload,
                )
            else:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO kb.tables_metadata "
                    "(id, document_id, section_id, table_name, caption, columns_json, units_json)"
                    " VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)",
                    payload,
                )
                cur.close()

    # -- Query helpers ---------------------------------------------------------
    def fetch_markdowns(self, document_id: str | None = None) -> list[dict[str, object]]:
        conn = self.connect()
        if self._backend == "sqlite":
            if document_id is None:
                cur = conn.execute(
                    "SELECT id, document_id, section_id, content, token_count, char_start, "
                    "char_end, start_page, end_page, start_line, end_line, emb, tsv FROM kb_markdowns"
                )
            else:
                cur = conn.execute(
                    "SELECT id, document_id, section_id, content, token_count, char_start, "
                    "char_end, start_page, end_page, start_line, end_line, emb, tsv FROM kb_markdowns "
                    "WHERE document_id = ?",
                    (document_id,),
                )
            rows = [dict(row) for row in cur.fetchall()]
            cur.close()
            return rows
        query = (
            "SELECT id, document_id, section_id, content, token_count, char_start, char_end,"
            " start_page, end_page, start_line, end_line, emb, tsv"
            " FROM kb.markdowns"
        )
        params: tuple[object, ...] = tuple()
        if document_id is not None:
            query += " WHERE document_id = %s"
            params = (document_id,)
        cur = self.connect().cursor()
        cur.execute(query, params)
        rows = [dict(zip([desc[0] for desc in cur.description], row)) for row in cur.fetchall()]
        cur.close()
        return rows

    def list_documents(self) -> list[dict[str, object]]:
        conn = self.connect()
        if self._backend == "sqlite":
            cur = conn.execute(
                "SELECT id, title, sha256, meta, created_at FROM kb_documents ORDER BY created_at DESC"
            )
            rows = [dict(row) for row in cur.fetchall()]
            cur.close()
            return rows
        cur = conn.cursor()
        cur.execute(
            "SELECT id, title, sha256, meta::text AS meta, created_at FROM kb.documents"
            " ORDER BY created_at DESC"
        )
        rows = [dict(zip([desc[0] for desc in cur.description], row)) for row in cur.fetchall()]
        cur.close()
        return rows
