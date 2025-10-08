"""Database utilities built around sqlite3 with a vector index companion."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence

from .vector_index import VectorIndex, VectorItem


class Database:
    """Lightweight wrapper exposing the few SQL features the project relies on."""

    def __init__(self, path: str) -> None:
        self.path = self._normalize_path(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.sqlite_conn = sqlite3.connect(self.path)
        self.sqlite_conn.row_factory = sqlite3.Row
        index_dir = Path(self.path).with_name(Path(self.path).name + ".index")
        self.index = VectorIndex(index_dir)

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_path(raw: str) -> str:
        if raw.startswith("sqlite:///"):
            return raw[len("sqlite:///") :]
        return raw

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        self._initialize_sqlite()

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
        self.sqlite_conn.executescript(ddl)
        self.sqlite_conn.commit()

    # Context manager --------------------------------------------------
    @contextmanager
    def connect(self):
        yield self.sqlite_conn

    # Mutation helpers -------------------------------------------------
    def delete_document(self, sha256: str) -> None:
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            "SELECT id FROM kb_markdowns WHERE document_id IN (SELECT id FROM kb_documents WHERE sha256 = ?)",
            (sha256,),
        )
        chunk_ids = [row[0] for row in cursor.fetchall()]
        if chunk_ids:
            self.index.delete(chunk_ids)
        cursor.execute("DELETE FROM kb_markdowns WHERE document_id IN (SELECT id FROM kb_documents WHERE sha256 = ?)", (sha256,))
        cursor.execute("DELETE FROM kb_sections WHERE document_id IN (SELECT id FROM kb_documents WHERE sha256 = ?)", (sha256,))
        cursor.execute("DELETE FROM kb_documents WHERE sha256 = ?", (sha256,))
        self.sqlite_conn.commit()

    def insert_document(self, *, doc_id: str, title: str, sha256: str, created_at: str, meta: str = "{}") -> None:
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
                    "tsv": row.get("tsv"),
                }
                for row in items
            ],
        )
        self.sqlite_conn.commit()
        self.index.upsert(
            VectorItem(
                id=str(row.get("id")),
                embedding=[float(value) for value in row.get("emb", [])],
                metadata={
                    "document_id": row.get("document_id"),
                    "section_id": row.get("section_id"),
                },
            )
            for row in items
        )

    # Query helpers ----------------------------------------------------
    def fetch_sections(self, document_id: str) -> dict[str, dict[str, object]]:
        cursor = self.sqlite_conn.cursor()
        cursor.execute("SELECT * FROM kb_sections WHERE document_id = ?", (document_id,))
        rows = cursor.fetchall()
        return {str(row["id"]): dict(row) for row in rows}

    def fetch_markdowns(self) -> list[dict[str, object]]:
        cursor = self.sqlite_conn.cursor()
        cursor.execute("SELECT * FROM kb_markdowns")
        return [dict(row) for row in cursor.fetchall()]

    def vector_search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
        keywords: Sequence[str] | None = None,
    ) -> list[dict[str, object]]:
        total = self.index.count()
        if total == 0:
            return []
        raw_hits = self.index.search(embedding, limit=total)
        if not raw_hits:
            return []
        chunk_ids = [chunk_id for chunk_id, _ in raw_hits]
        placeholders = ",".join("?" for _ in chunk_ids)
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            f"SELECT id, document_id, section_id, content, start_page, end_page, token_count, tsv FROM kb_markdowns WHERE id IN ({placeholders})",
            chunk_ids,
        )
        rows = {row["id"]: row for row in cursor.fetchall()}
        keywords_normalized = [kw.lower() for kw in keywords or []]
        results: list[dict[str, object]] = []
        for chunk_id, score in raw_hits:
            row = rows.get(chunk_id)
            if row is None:
                continue
            if keywords_normalized:
                tokens = str(row["tsv"]).split()
                if not any(term in tokens for term in keywords_normalized):
                    continue
            results.append(
                {
                    "id": chunk_id,
                    "document_id": row["document_id"],
                    "section_id": row["section_id"],
                    "content": row["content"],
                    "start_page": row["start_page"],
                    "end_page": row["end_page"],
                    "token_count": row["token_count"],
                    "score": score,
                }
            )
            if len(results) >= limit:
                break
        if len(results) < limit and keywords_normalized:
            # fall back to scanning rows not surfaced by the index yet
            seen = {row["id"] for row in results}
            cursor.execute(
                "SELECT id, document_id, section_id, content, start_page, end_page, token_count, emb, tsv FROM kb_markdowns"
            )
            for row in cursor.fetchall():
                if row["id"] in seen:
                    continue
                tokens = str(row["tsv"]).split()
                if not any(term in tokens for term in keywords_normalized):
                    continue
                vector = json.loads(row["emb"])
                score = self._cosine_similarity(embedding, vector)
                results.append(
                    {
                        "id": row["id"],
                        "document_id": row["document_id"],
                        "section_id": row["section_id"],
                        "content": row["content"],
                        "start_page": row["start_page"],
                        "end_page": row["end_page"],
                        "token_count": row["token_count"],
                        "score": score,
                    }
                )
                if len(results) >= limit:
                    break
        results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return results[:limit]

    # Utilities --------------------------------------------------------
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


__all__ = ["Database"]
