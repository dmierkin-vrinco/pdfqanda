"""Utilities for loading text from PDF documents."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

_STREAM_PATTERN = re.compile(rb"stream(.*?)endstream", re.DOTALL)
_TEXT_PATTERN = re.compile(rb"\((.*?)\)")


def _decode_pdf_text(data: bytes) -> str:
    text = data.decode("latin1")
    text = text.replace("\\r", "\r").replace("\\n", "\n").replace("\\t", "\t")
    text = text.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")
    return text


def extract_text_from_pdf(path: str | Path) -> str:
    """Extract text content from a PDF file by scanning content streams."""

    pdf_path = Path(path)
    if not pdf_path.exists():
        msg = f"PDF file does not exist: {pdf_path}"
        raise FileNotFoundError(msg)

    data = pdf_path.read_bytes()
    text_fragments: list[str] = []
    for stream_match in _STREAM_PATTERN.finditer(data):
        stream_data = stream_match.group(1)
        for text_match in _TEXT_PATTERN.finditer(stream_data):
            text_fragments.append(_decode_pdf_text(text_match.group(1)))

    return "\n".join(fragment.strip() for fragment in text_fragments if fragment.strip())


def load_documents(paths: Iterable[str | Path]) -> str:
    """Load and concatenate text from multiple PDF documents."""

    contents: list[str] = []
    for path in paths:
        contents.append(extract_text_from_pdf(path))
    return "\n\n".join(filter(None, contents))
