"""High-level interface for building and querying PDF Q&A indexes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .ingest import extract_text_from_pdf, load_documents
from .splitter import TextSplitter
from .vectorstore import TfidfVectorStore, VectorStoreResult


@dataclass
class Answer:
    """Answer returned by the QA engine."""

    text: str
    score: float
    source_index: int
    start_char: int
    end_char: int


class PdfQaEngine:
    """Build a simple retrieval-based Q&A system for PDF documents."""

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        top_k: int = 3,
    ) -> None:
        self.splitter = TextSplitter(chunk_size=chunk_size, overlap=overlap)
        self.top_k = top_k
        self.store = TfidfVectorStore()

    def build_index(self, pdf_paths: Iterable[str | Path]) -> None:
        """Load PDFs and build the TF-IDF index."""

        combined_text = load_documents(pdf_paths)
        chunks = self.splitter.split_text(combined_text)
        if not chunks:
            msg = "No text extracted from provided documents"
            raise ValueError(msg)
        self.store.fit(chunks)

    def query(self, question: str) -> list[Answer]:
        results = self.store.query(question, top_k=self.top_k)
        return [
            Answer(
                text=result.chunk.content.strip(),
                score=result.score,
                source_index=result.chunk.index,
                start_char=result.chunk.start_char,
                end_char=result.chunk.end_char,
            )
            for result in results
        ]

    def save(self, path: str | Path) -> None:
        self.store.save(path)

    @classmethod
    def load(cls, path: str | Path) -> "PdfQaEngine":
        engine = cls()
        engine.store = TfidfVectorStore.load(path)
        return engine

    def build_index_with_progress(self, pdf_paths: Iterable[str | Path]) -> None:
        """Build index while displaying a progress bar."""

        pdf_paths = list(pdf_paths)
        combined_texts = []
        total = len(pdf_paths)
        for idx, pdf_path in enumerate(pdf_paths, start=1):
            combined_texts.append(extract_text_from_pdf(pdf_path))
            print(f"Reading PDFs: {idx}/{total}", end="\r", flush=True)
        print()
        combined_text = "\n\n".join(filter(None, combined_texts))
        chunks = self.splitter.split_text(combined_text)
        if not chunks:
            msg = "No text extracted from provided documents"
            raise ValueError(msg)
        self.store.fit(chunks)

    def query_with_sources(self, question: str) -> list[tuple[Answer, VectorStoreResult]]:
        results = self.store.query(question, top_k=self.top_k)
        return [
            (
                Answer(
                    text=result.chunk.content.strip(),
                    score=result.score,
                    source_index=result.chunk.index,
                    start_char=result.chunk.start_char,
                    end_char=result.chunk.end_char,
                ),
                result,
            )
            for result in results
        ]
