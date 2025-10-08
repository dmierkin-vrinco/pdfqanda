"""Hybrid retrieval Researcher agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from .config import get_settings
from .db import Database
from .embedding import build_tsvector, count_term_hits, cosine_similarity
from .models import ResearchHit
from .util.embeddings import EmbeddingClient


@dataclass(slots=True)
class ResearchOutput:
    """Payload returned by the Researcher search pipeline."""

    hits: list[ResearchHit]
    exhausted: bool
    sql: str | None = None


class Researcher:
    """Performs hybrid semantic + lexical retrieval with optional SQL scaffolding."""

    def __init__(self, database: Database, embedder: EmbeddingClient | None = None) -> None:
        settings = get_settings()
        self.database = database
        self.embedder = embedder or EmbeddingClient(settings.embedding_model, settings.embedding_dim)

    def search(self, question: str, top_k: int = 6) -> ResearchOutput:
        """Search the knowledge base and return ranked evidence snippets."""

        if not question.strip():
            return ResearchOutput(hits=[], exhausted=True)

        query_embedding = self.embedder.embed_query(question)
        rows = self.database.fetch_markdowns()
        scored = []
        for row in rows:
            embedding_blob = row["emb"]
            if isinstance(embedding_blob, str):
                embedding = json.loads(embedding_blob)
            else:  # pragma: no cover - legacy vector adapters
                embedding = list(embedding_blob)
            score = cosine_similarity(query_embedding, embedding)
            scored.append((score, row, embedding))
        scored.sort(key=lambda item: item[0], reverse=True)
        vector_top_k = scored[: max(top_k, 12)]

        query_terms = [term for term in build_tsvector(question).split() if term]
        reranked = []
        for score, row, embedding in vector_top_k:
            tsv = row.get("tsv", "")
            lexical_hits = count_term_hits(tsv, query_terms)
            rerank_score = score + 0.05 * lexical_hits
            reranked.append((rerank_score, lexical_hits, row, embedding))
        reranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        final_hits = reranked[: max(top_k, 8)]

        hits: list[ResearchHit] = []
        for score, lexical_hits, row, embedding in final_hits[:top_k]:
            citation = self._build_citation(row)
            hits.append(
                ResearchHit(
                    document_id=row["document_id"],
                    section_id=row.get("section_id"),
                    content=row.get("content", ""),
                    score=score,
                    citation=citation,
                    start_page=row.get("start_page") or 0,
                    end_page=row.get("end_page") or row.get("start_page") or 0,
                    start_line=row.get("start_line") or 1,
                    end_line=row.get("end_line") or row.get("start_line") or 1,
                )
            )
        exhausted = len(final_hits) <= top_k
        sql = self._maybe_generate_sql(question, hits)
        return ResearchOutput(hits=hits, exhausted=exhausted, sql=sql)

    def _build_citation(self, row: dict[str, object]) -> str:
        document_id = row.get("document_id", "doc")
        section_id = row.get("section_id") or "root"
        start_page = int(row.get("start_page") or 0) + 1
        end_page = int(row.get("end_page") or start_page - 1) + 1
        start_line = int(row.get("start_line") or 1)
        end_line = int(row.get("end_line") or start_line)
        return (
            f"【doc:{document_id} §{section_id} p.{start_page}-{end_page} "
            f"| L{start_line}-{end_line}】"
        )

    def _maybe_generate_sql(self, question: str, hits: Sequence[ResearchHit]) -> str | None:
        lowered = question.lower()
        if "select" in lowered or "drop" in lowered:
            return None
        if "table" not in lowered and "column" not in lowered:
            return None
        if not hits:
            return None
        # Basic scaffold referencing the top document for manual refinement.
        doc_id = hits[0].document_id
        return f"SELECT * FROM pdf_tables.{doc_id[:8]} LIMIT 10;"
