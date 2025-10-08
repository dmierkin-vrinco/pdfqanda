"""Domain models used across ingestion and retrieval pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Sequence


@dataclass(slots=True)
class BBox:
    """Normalized bounding box within a page."""

    x0: float
    y0: float
    x1: float
    y1: float

    def to_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


NoteKind = Literal["footnote", "annotation", "reference"]


@dataclass(slots=True)
class Note:
    """Detected note information associated with a page."""

    page: int
    bbox: BBox
    kind: NoteKind
    text: str
    ref: str | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "page": self.page,
            "bbox": self.bbox.to_list(),
            "kind": self.kind,
            "text": self.text,
        }
        if self.ref is not None:
            data["ref"] = self.ref
        return data


@dataclass(slots=True)
class Graphic:
    """Metadata about extracted graphics for a page."""

    page: int
    bbox: BBox
    nearby_text: str
    path: str
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "page": self.page,
            "bbox": self.bbox.to_list(),
            "nearby_text": self.nearby_text,
            "path": self.path,
            "sha256": self.sha256,
        }


@dataclass(slots=True)
class Page:
    """Representation of a PDF page with detected metadata."""

    index: int
    text_blocks: list[str] = field(default_factory=list)
    bbox_blocks: list[BBox] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    graphics: list[Graphic] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "text_blocks": self.text_blocks,
            "bbox_blocks": [bbox.to_list() for bbox in self.bbox_blocks],
            "notes": [note.to_dict() for note in self.notes],
            "graphics": [graphic.to_dict() for graphic in self.graphics],
        }


@dataclass(slots=True)
class DocumentRecord:
    """Canonical representation of a document ready for persistence."""

    id: str
    title: str
    sha256: str
    meta: dict[str, Any]
    created_at: datetime


@dataclass(slots=True)
class SectionRecord:
    """Logical section within a document outline."""

    id: str
    document_id: str
    title: str
    level: int
    start_page: int
    end_page: int
    path: str
    parent_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MarkdownChunk:
    """Semantic chunk of text produced by the segmenter."""

    id: str
    document_id: str
    section_id: str | None
    content: str
    token_count: int
    char_start: int
    char_end: int
    start_page: int
    end_page: int
    start_line: int
    end_line: int
    embedding: Sequence[float]
    tsv: str


@dataclass(slots=True)
class NoteRecord:
    """Note persisted in the database."""

    id: str
    document_id: str
    section_id: str | None
    kind: str
    ref_anchor: str | None
    content: str
    page: int | None
    bbox: Sequence[float] | None


@dataclass(slots=True)
class GraphicRecord:
    """Graphic metadata persisted in the database."""

    id: str
    document_id: str
    section_id: str | None
    caption: str | None
    nearby_text: str
    path: str
    sha256: str
    page: int | None
    bbox: Sequence[float] | None


@dataclass(slots=True)
class TableMetadataRecord:
    """Metadata describing an extracted table."""

    id: str
    document_id: str
    section_id: str | None
    table_name: str
    caption: str | None
    columns_json: dict[str, Any] | list[Any] | None
    units_json: dict[str, Any] | list[Any] | None


@dataclass(slots=True)
class ResearchHit:
    """Result returned by the Researcher hybrid retrieval pipeline."""

    document_id: str
    section_id: str | None
    content: str
    score: float
    citation: str
    start_page: int
    end_page: int
    start_line: int
    end_line: int
