"""PDF ingestion pipeline backed by Postgres/pgvector."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

_FITZ = None
if importlib.util.find_spec("fitz") is not None:  # pragma: no cover - optional dependency
    _FITZ = importlib.import_module("fitz")

from .config import get_settings
from .db import Database
from .embedding import build_tsvector, deterministic_embedding


@dataclass(slots=True)
class Chunk:
    id: str
    document_id: str
    section_id: str | None
    content: str
    token_count: int
    start_page: int
    end_page: int
    embedding: list[float]
    tsv: str


@dataclass(slots=True)
class IngestResult:
    document_id: str
    sha256: str
    chunk_count: int


class PdfIngestor:
    """Extracts text from PDFs, chunks content, and stores into the database."""

    def __init__(self, database: Database | None = None) -> None:
        settings = get_settings()
        self.database = database or Database(settings.db_dsn)
        self.settings = settings

    def ingest(self, pdf_path: Path, title: str | None = None) -> IngestResult:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():  # pragma: no cover - guardrail
            raise FileNotFoundError(pdf_path)

        sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        title = title or pdf_path.stem

        document_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        self.database.delete_document(sha256)
        self.database.insert_document(doc_id=document_id, title=title, sha256=sha256, created_at=now)

        section_id = str(uuid.uuid4())
        self.database.insert_sections(
            [
                {
                    "id": section_id,
                    "document_id": document_id,
                    "parent_id": None,
                    "title": title,
                    "level": 1,
                    "start_page": 0,
                    "end_page": 0,
                    "path": title,
                }
            ]
        )

        chunks = list(self._chunk_pdf(pdf_path, document_id, section_id))
        self.database.insert_markdowns(
            [
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "section_id": chunk.section_id,
                    "content": chunk.content,
                    "token_count": chunk.token_count,
                    "start_page": chunk.start_page,
                    "end_page": chunk.end_page,
                    "emb": chunk.embedding,
                    "tsv": chunk.tsv,
                }
                for chunk in chunks
            ]
        )

        return IngestResult(document_id=document_id, sha256=sha256, chunk_count=len(chunks))

    # ------------------------------------------------------------------
    def _chunk_pdf(self, pdf_path: Path, document_id: str, section_id: str) -> Iterable[Chunk]:
        paragraphs = self._extract_paragraphs(pdf_path)

        target_tokens = self.settings.chunk_target_tokens
        overlap_tokens = max(1, int(target_tokens * self.settings.chunk_overlap_ratio))

        buffer: list[tuple[int, str, int]] = []
        token_total = 0
        for page_index, text in paragraphs:
            tokens = self._count_tokens(text)
            if not text.strip():
                continue
            if token_total + tokens > target_tokens and buffer:
                yield self._emit_chunk(buffer, document_id, section_id)
                buffer = self._apply_overlap(buffer, overlap_tokens)
                token_total = sum(item[2] for item in buffer)
            buffer.append((page_index, text, tokens))
            token_total += tokens
        if buffer:
            yield self._emit_chunk(buffer, document_id, section_id)

    def _emit_chunk(self, buffer: list[tuple[int, str, int]], document_id: str, section_id: str) -> Chunk:
        content = "\n\n".join(text for _, text, _ in buffer).strip()
        token_count = sum(tokens for _, _, tokens in buffer)
        start_page = buffer[0][0]
        end_page = buffer[-1][0]
        embedding = list(deterministic_embedding(content, self.settings.embedding_dim))
        tsv = build_tsvector(content)
        return Chunk(
            id=str(uuid.uuid4()),
            document_id=document_id,
            section_id=section_id,
            content=content,
            token_count=token_count,
            start_page=start_page,
            end_page=end_page,
            embedding=embedding,
            tsv=tsv,
        )

    def _apply_overlap(
        self, buffer: list[tuple[int, str, int]], overlap_tokens: int
    ) -> list[tuple[int, str, int]]:
        retained: list[tuple[int, str, int]] = []
        running = 0
        for item in reversed(buffer):
            retained.insert(0, item)
            running += item[2]
            if running >= overlap_tokens:
                break
        return retained

    @staticmethod
    def _normalize_paragraphs(text: str) -> list[str]:
        paragraphs = []
        for raw in text.split("\n\n"):
            normalized = " ".join(segment.strip() for segment in raw.splitlines()).strip()
            if normalized:
                paragraphs.append(normalized)
        return paragraphs or ([text.strip()] if text.strip() else [])

    @staticmethod
    def _count_tokens(text: str) -> int:
        return max(1, len(text.split()))

    def _extract_paragraphs(self, pdf_path: Path) -> list[tuple[int, str]]:
        if _FITZ is not None:
            doc = _FITZ.open(pdf_path)
            paragraphs: list[tuple[int, str]] = []
            for page_index, page in enumerate(doc):
                text = page.get_text("text")
                for paragraph in self._normalize_paragraphs(text):
                    paragraphs.append((page_index, paragraph))
            doc.close()
            return paragraphs
        raw_text = self._extract_text_without_pymupdf(pdf_path)
        return [(0, paragraph) for paragraph in self._normalize_paragraphs(raw_text)]

    @staticmethod
    def _extract_text_without_pymupdf(pdf_path: Path) -> str:
        data = pdf_path.read_bytes()
        stream_pattern = re.compile(rb"stream(.*?)endstream", re.DOTALL)
        text_pattern = re.compile(rb"\((.*?)\)")
        fragments: list[str] = []
        for stream_match in stream_pattern.finditer(data):
            stream_data = stream_match.group(1)
            for text_match in text_pattern.finditer(stream_data):
                fragments.append(PdfIngestor._decode_pdf_text(text_match.group(1)))
        return "\n".join(fragment for fragment in fragments if fragment.strip())

    @staticmethod
    def _decode_pdf_text(data: bytes) -> str:
        text = data.decode("latin1")
        text = text.replace("\\r", "\r").replace("\\n", "\n").replace("\\t", "\t")
        text = text.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")
        return text
