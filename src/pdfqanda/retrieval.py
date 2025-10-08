"""Query the knowledge base using vector search with optional keyword filtering."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .config import get_settings
from .util.db import Database
from .util.embeddings import EmbeddingClient

_KEYWORD_RE = re.compile(r"[A-Za-z]{4,}")


@dataclass(slots=True)
class RetrievalHit:
    document_id: str
    section_id: str | None
    content: str
    score: float
    start_page: int
    end_page: int
    citation: str


class Retriever:
    """Run vector search with optional full-text filtering."""

    def __init__(self, database: Database | None = None) -> None:
        settings = get_settings()
        self.database = database or Database(settings.db_dsn)
        self.settings = settings
        self.embedder = EmbeddingClient(settings.embedding_model, settings.embedding_dim)

    def search(self, query: str, k: int = 6) -> list[RetrievalHit]:
        query = query.strip()
        if not query:
            return []
        embedding = self.embedder.embed_query(query)
        keywords = self._keywords(query)
        raw_hits = self.database.vector_search(embedding, limit=max(12, k), keywords=keywords)
        ranked = sorted(raw_hits, key=lambda row: float(row.get("score", 0.0)), reverse=True)
        hits: list[RetrievalHit] = []
        for row in ranked[:k]:
            citation = self._citation(row)
            hits.append(
                RetrievalHit(
                    document_id=str(row.get("document_id")),
                    section_id=str(row.get("section_id")) if row.get("section_id") else None,
                    content=str(row.get("content", "")),
                    score=float(row.get("score", 0.0)),
                    start_page=int(row.get("start_page") or 0),
                    end_page=int(row.get("end_page") or row.get("start_page") or 0),
                    citation=citation,
                )
            )
        return hits

    @staticmethod
    def _keywords(query: str) -> list[str]:
        return _KEYWORD_RE.findall(query)

    @staticmethod
    def _citation(row: dict[str, object]) -> str:
        doc_id = row.get("document_id", "doc")
        section_id = row.get("section_id") or "root"
        start_page = int(row.get("start_page") or 0) + 1
        end_page = int(row.get("end_page") or row.get("start_page") or 0) + 1
        return f"【doc:{doc_id} §{section_id} p.{start_page}-{end_page}】"


def format_answer(hits: Iterable[RetrievalHit]) -> str:
    """Render a plain-text answer with inline citations."""

    snippets = []
    for hit in hits:
        snippet = hit.content.strip()
        if not snippet:
            continue
        snippets.append(f"{snippet}\n{hit.citation}")
    return "\n\n".join(snippets)
