"""Database-backed retrieval utilities."""

from __future__ import annotations

from dataclasses import dataclass

from .config import get_settings
from .db import Database
from .embedding import build_tsvector, deterministic_embedding


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
    """Run vector search and optional FTS refinement."""

    def __init__(self, database: Database | None = None) -> None:
        settings = get_settings()
        self.database = database or Database(settings.db_dsn)
        self.settings = settings

    def search(self, query: str, k: int = 6) -> list[RetrievalHit]:
        if not query.strip():
            return []

        vector = list(deterministic_embedding(query, self.settings.embedding_dim))
        vector_hits = self.database.vector_search(vector, max(k, 12))
        keyword_bonus = self._fts_bonus(query, vector_hits)

        scored = []
        for row in vector_hits:
            distance = float(row.get("distance", 1.0))
            score = 1.0 - distance
            score += keyword_bonus.get(row["id"], 0.0)
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)

        final_hits: list[RetrievalHit] = []
        for score, row in scored[:k]:
            citation = self._format_citation(row)
            final_hits.append(
                RetrievalHit(
                    document_id=row["document_id"],
                    section_id=row.get("section_id"),
                    content=row.get("content", ""),
                    score=score,
                    start_page=int(row.get("start_page") or 0),
                    end_page=int(row.get("end_page") or row.get("start_page") or 0),
                    citation=citation,
                )
            )
        return final_hits

    def _fts_bonus(self, query: str, hits: list[dict[str, object]]) -> dict[str, float]:
        terms = [term for term in build_tsvector(query).split() if term]
        if not terms:
            return {}
        candidate_ids = [row["id"] for row in hits]
        bonus = self.database.fts_refine(query, candidate_ids)
        if bonus:
            return bonus
        # SQLite fallback using simple keyword overlap
        bonuses: dict[str, float] = {}
        for row in hits:
            tsv = row.get("tsv") or ""
            if not tsv:
                continue
            overlap = len(set(tsv.split()) & set(terms))
            if overlap:
                bonuses[row["id"]] = 0.05 * overlap
        return bonuses

    @staticmethod
    def _format_citation(row: dict[str, object]) -> str:
        doc_id = row.get("document_id", "doc")
        section_id = row.get("section_id") or "root"
        start_page = int(row.get("start_page") or 0) + 1
        end_page = int(row.get("end_page") or row.get("start_page") or 0) + 1
        return f"【doc:{doc_id} §{section_id} p.{start_page}-{end_page}】"
