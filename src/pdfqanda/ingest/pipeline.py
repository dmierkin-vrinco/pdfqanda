"""Ingestion pipeline implementations."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from ..config import get_settings
from ..embedding import build_tsvector
from ..util.cache import FileCache, stable_hash
from ..util.db import Database
from ..util.embeddings import EmbeddingClient

__all__ = ["Section", "Chunk", "IngestResult", "PdfIngestor"]


@dataclass(slots=True)
class Section:
    """Represents a logical section within an ingested document."""

    id: str
    document_id: str
    title: str
    level: int
    start_page: int
    end_page: int
    path: str


@dataclass(slots=True)
class Chunk:
    """A semantic chunk ready for persistence."""

    id: str
    document_id: str
    section_id: str
    content: str
    token_count: int
    char_start: int
    char_end: int
    start_page: int
    end_page: int
    embedding: list[float]
    tsv: str


@dataclass(slots=True)
class IngestResult:
    """Summary of a successful ingestion run."""

    document_id: str
    sha256: str
    chunk_count: int


class PdfIngestor:
    """Extracts text from PDFs, segments content, and stores it in the database."""

    _FITZ_AVAILABLE = importlib.util.find_spec("fitz") is not None

    def __init__(
        self,
        database: Database | None = None,
        embedder: EmbeddingClient | None = None,
    ) -> None:
        settings = get_settings()
        self.database = database or Database(settings.db_path)
        self.settings = settings
        self.embedder = embedder or EmbeddingClient(
            settings.embedding_model, settings.embedding_dim
        )
        self.pdf_cache = FileCache(Path(".cache/pdf"))
        self.table_cache = FileCache(Path(".cache/tables"))

    # ------------------------------------------------------------------
    def ingest(self, pdf_path: Path, title: str | None = None) -> IngestResult:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():  # pragma: no cover - guardrail
            raise FileNotFoundError(pdf_path)

        sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        title = title or pdf_path.stem

        document_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        self.database.delete_document(sha256)
        self.database.insert_document(
            doc_id=document_id, title=title, sha256=sha256, created_at=now
        )

        pages = self._load_pages(pdf_path, sha256)
        layout_key = stable_hash([sha256, "sections:v1"])
        cached_sections = self.table_cache.get("layouts", layout_key)
        if cached_sections:
            sections = [
                Section(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    title=str(section["title"]),
                    level=int(section["level"]),
                    start_page=int(section["start_page"]),
                    end_page=int(section["end_page"]),
                    path=str(section["path"]),
                )
                for section in cached_sections
            ]
        else:
            sections = self._derive_sections(document_id, title, pages)
        if not sections:
            root_section = Section(
                id=str(uuid.uuid4()),
                document_id=document_id,
                title=title,
                level=1,
                start_page=0,
                end_page=max(0, len(pages) - 1),
                path=title,
            )
            sections = [root_section]
        self.database.insert_sections(
            [
                {
                    "id": section.id,
                    "document_id": section.document_id,
                    "parent_id": None,
                    "title": section.title,
                    "level": section.level,
                    "start_page": section.start_page,
                    "end_page": section.end_page,
                    "path": section.path,
                    "meta": {},
                }
                for section in sections
            ]
        )
        if not cached_sections:
            self.table_cache.set(
                "layouts",
                layout_key,
                [
                    {
                        "title": section.title,
                        "level": section.level,
                        "start_page": section.start_page,
                        "end_page": section.end_page,
                        "path": section.path,
                    }
                    for section in sections
                ],
            )

        chunks = self._segment(document_id, sections[0], pages)
        embeddings = self.embedder.embed_documents([chunk.content for chunk in chunks])
        for idx, chunk in enumerate(chunks):
            chunk.embedding = embeddings[idx]

        self.database.insert_markdowns(
            [
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "section_id": chunk.section_id,
                    "content": chunk.content,
                    "token_count": chunk.token_count,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "start_page": chunk.start_page,
                    "end_page": chunk.end_page,
                    "emb": chunk.embedding,
                    "tsv": chunk.tsv,
                }
                for chunk in chunks
            ]
        )

        return IngestResult(document_id=document_id, sha256=sha256, chunk_count=len(chunks))

    # Internal helpers -----------------------------------------------------
    def _load_pages(self, pdf_path: Path, sha256: str) -> list[str]:
        cached = self.pdf_cache.get("pages", sha256)
        if cached is not None:
            return list(cached)
        pages = self._extract_pages(pdf_path)
        self.pdf_cache.set("pages", sha256, pages)
        return pages

    def _extract_pages(self, pdf_path: Path) -> list[str]:
        if self._FITZ_AVAILABLE:  # pragma: no cover - requires PyMuPDF
            import fitz  # type: ignore[import]

            doc = fitz.open(pdf_path)
            pages = [page.get_text("text") for page in doc]
            doc.close()
            return pages
        return self._fallback_extract(pdf_path)

    def _fallback_extract(self, pdf_path: Path) -> list[str]:
        data = pdf_path.read_bytes()
        stream_pattern = re.compile(rb"stream(.*?)endstream", re.DOTALL)
        text_pattern = re.compile(rb"\((.*?)\)")
        fragments: list[str] = []
        for stream_match in stream_pattern.finditer(data):
            stream_data = stream_match.group(1)
            for text_match in text_pattern.finditer(stream_data):
                fragments.append(self._decode_pdf_text(text_match.group(1)))
        combined = "\n".join(fragment for fragment in fragments if fragment.strip())
        return [combined]

    @staticmethod
    def _decode_pdf_text(data: bytes) -> str:
        text = data.decode("latin1")
        text = text.replace("\\r", "\r").replace("\\n", "\n").replace("\\t", "\t")
        text = text.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")
        return text

    def _derive_sections(
        self, document_id: str, title: str, pages: Sequence[str]
    ) -> list[Section]:
        if not pages:
            return []
        end_page = max(0, len(pages) - 1)
        return [
            Section(
                id=str(uuid.uuid4()),
                document_id=document_id,
                title=title,
                level=1,
                start_page=0,
                end_page=end_page,
                path=title,
            )
        ]

    def _segment(self, document_id: str, section: Section, pages: Sequence[str]) -> list[Chunk]:
        paragraphs: list[tuple[int, str]] = []
        for page_idx, page_text in enumerate(pages):
            for paragraph in self._normalize_paragraphs(page_text):
                paragraphs.append((page_idx, paragraph))
        if not paragraphs:
            paragraphs = [(0, "")] if pages else []

        target_tokens = max(1, self.settings.chunk_target_tokens)
        overlap_tokens = max(1, int(target_tokens * self.settings.chunk_overlap_ratio))

        chunks: list[Chunk] = []
        buffer: list[tuple[int, str, int, int, int]] = []  # page, text, tokens, char_start, char_end
        running_tokens = 0
        running_chars = 0
        for page_idx, text in paragraphs:
            tokens = self._count_tokens(text)
            if not text.strip():
                running_chars += len(text) + 2
                continue
            paragraph_start = running_chars
            paragraph_end = paragraph_start + len(text)
            running_chars = paragraph_end + 2
            if running_tokens + tokens > target_tokens and buffer:
                chunks.append(self._emit_chunk(document_id, section, buffer))
                buffer = self._apply_overlap(buffer, overlap_tokens)
                running_tokens = sum(item[2] for item in buffer)
            buffer.append((page_idx, text, tokens, paragraph_start, paragraph_end))
            running_tokens += tokens
        if buffer:
            chunks.append(self._emit_chunk(document_id, section, buffer))
        return chunks

    def _emit_chunk(
        self,
        document_id: str,
        section: Section,
        buffer: list[tuple[int, str, int, int, int]],
    ) -> Chunk:
        content = "\n\n".join(item[1] for item in buffer).strip()
        token_count = sum(item[2] for item in buffer)
        char_start = buffer[0][3]
        char_end = buffer[-1][4]
        start_page = buffer[0][0]
        end_page = buffer[-1][0]
        return Chunk(
            id=str(uuid.uuid4()),
            document_id=document_id,
            section_id=section.id,
            content=content,
            token_count=token_count,
            char_start=char_start,
            char_end=char_end,
            start_page=start_page,
            end_page=end_page,
            embedding=[],
            tsv=build_tsvector(content),
        )

    def _apply_overlap(
        self,
        buffer: list[tuple[int, str, int, int, int]],
        overlap_tokens: int,
    ) -> list[tuple[int, str, int, int, int]]:
        retained: list[tuple[int, str, int, int, int]] = []
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
        return paragraphs

    @staticmethod
    def _count_tokens(text: str) -> int:
        return max(1, len(text.split()))
