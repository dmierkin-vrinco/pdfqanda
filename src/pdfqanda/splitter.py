"""Simple text chunking utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextChunk:
    """Representation of a chunk of text with metadata."""

    content: str
    index: int
    start_char: int
    end_char: int


class TextSplitter:
    """Split text into overlapping chunks."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200) -> None:
        if chunk_size <= 0:
            msg = "chunk_size must be positive"
            raise ValueError(msg)
        if not 0 <= overlap < chunk_size:
            msg = "overlap must be between 0 and chunk_size"
            raise ValueError(msg)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> list[TextChunk]:
        """Split text into chunks of approximately ``chunk_size`` characters."""

        chunks: list[TextChunk] = []
        if not text:
            return chunks

        start = 0
        index = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size)
            chunk_text = text[start:end]
            chunks.append(TextChunk(content=chunk_text, index=index, start_char=start, end_char=end))
            index += 1
            if end == len(text):
                break
            start = end - self.overlap
        return chunks
