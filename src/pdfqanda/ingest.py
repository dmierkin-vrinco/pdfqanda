"""PDF ingestion pipeline producing canonical database records."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from .config import Settings, get_settings
from .db import Database
from .embedding import build_tsvector, deterministic_embedding
from .models import (
    BBox,
    DocumentRecord,
    Graphic,
    GraphicRecord,
    MarkdownChunk,
    Note,
    NoteRecord,
    Page,
    SectionRecord,
    TableMetadataRecord,
)
from .segmenter import SemanticSegmenter, char_to_line, locate_pages

_CACHE_SETTINGS = get_settings()


@dataclass(slots=True)
class IngestionArtifacts:
    """Artifacts persisted alongside the structured database rows."""

    doc_hash: str
    artifact_path: Path
    page_count: int


_FOOTNOTE_PATTERN = re.compile(r"^(\d+[\.\)]\s+|\*+|[†]+)")
_SUPERSCRIPT_PATTERN = re.compile(r"\^(\d+)")
_REF_SECTION_TITLES = {"references", "notes", "footnotes"}
_STREAM_PATTERN = re.compile(rb"stream(.*?)endstream", re.DOTALL)
_TEXT_PATTERN = re.compile(rb"\((.*?)\)")


def compute_doc_hash(path: str | Path) -> str:
    pdf_path = Path(path)
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()


def _decode_pdf_text(data: bytes) -> str:
    text = data.decode("latin1")
    text = text.replace("\\r", "\r").replace("\\n", "\n").replace("\\t", "\t")
    text = text.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")
    return text


def _extract_text_from_pdf_stream(path: Path) -> str:
    data = path.read_bytes()
    text_fragments: list[str] = []
    for stream_match in _STREAM_PATTERN.finditer(data):
        stream_data = stream_match.group(1)
        for text_match in _TEXT_PATTERN.finditer(stream_data):
            text_fragments.append(_decode_pdf_text(text_match.group(1)))
    return "\n".join(fragment.strip() for fragment in text_fragments if fragment.strip())


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _ensure_pdf_dirs(doc_hash: str) -> tuple[Path, Path]:
    cache_dir = _CACHE_SETTINGS.cache_pdf_dir / doc_hash
    graphics_dir = cache_dir / "graphics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if graphics_dir.exists():
        shutil.rmtree(graphics_dir)
    graphics_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir, graphics_dir


def _load_cached_blocks(block_path: Path) -> list[dict[str, object]]:
    if not block_path.exists():
        return []
    return json.loads(block_path.read_text())


def _save_blocks(block_path: Path, blocks: Sequence[Sequence[object]]) -> None:
    serializable = []
    for block in blocks:
        if len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[:5]
        serializable.append({
            "bbox": [float(x0), float(y0), float(x1), float(y1)],
            "text": str(text),
        })
    block_path.write_text(json.dumps(serializable))


def _normalize_bbox(raw_bbox: Sequence[float], width: float, height: float) -> BBox:
    x0, y0, x1, y1 = raw_bbox
    return BBox(
        _clamp(x0 / width if width else 0.0),
        _clamp(y0 / height if height else 0.0),
        _clamp(x1 / width if width else 0.0),
        _clamp(y1 / height if height else 0.0),
    )


def _is_footnote_candidate(text: str, page_text: str) -> tuple[bool, str | None]:
    stripped = text.strip()
    if not stripped:
        return False, None
    match = _FOOTNOTE_PATTERN.match(stripped)
    if match:
        leading_digits = re.match(r"^(\d+)", stripped)
        if leading_digits:
            digits = leading_digits.group(1)
            if re.search(rf"\^{re.escape(digits)}", page_text):
                return True, digits
            return True, digits
        marker = match.group(1)
        return True, marker.strip()
    superscripts = set(_SUPERSCRIPT_PATTERN.findall(page_text))
    if superscripts:
        leading_digits = re.match(r"^(\d+)", stripped)
        if leading_digits and leading_digits.group(1) in superscripts:
            return True, leading_digits.group(1)
    return False, None


def _detect_notes(
    blocks: list[dict[str, object]],
    normalized_blocks: list[BBox],
    page_text: str,
    page_index: int,
) -> list[Note]:
    notes: list[Note] = []
    bottom_threshold = 0.8
    in_reference_section = False
    for idx, block in enumerate(blocks):
        text = str(block["text"]).strip()
        if not text:
            continue
        bbox = normalized_blocks[idx]
        if bbox.y0 >= bottom_threshold:
            lowered = text.lower()
            if lowered in _REF_SECTION_TITLES:
                in_reference_section = True
                continue
            candidate, ref = _is_footnote_candidate(text, page_text)
            if candidate or in_reference_section:
                notes.append(Note(page=page_index, bbox=bbox, kind="footnote", text=text, ref=ref))
        elif in_reference_section:
            notes.append(Note(page=page_index, bbox=bbox, kind="footnote", text=text))
    return notes


def _find_nearby_text(blocks: list[dict[str, object]], rect, page_height: float) -> str:
    band = page_height * 0.05
    best_distance: float | None = None
    best_text = ""
    for block in blocks:
        text = str(block["text"]).strip()
        if not text:
            continue
        bx0, by0, bx1, by1 = block["bbox"]
        vertical_distance = 0.0
        if rect.y1 <= by0:
            vertical_distance = by0 - rect.y1
        elif by1 <= rect.y0:
            vertical_distance = rect.y0 - by1
        overlap = max(0.0, min(rect.x1, bx1) - max(rect.x0, bx0))
        if overlap <= 0 and vertical_distance > band:
            continue
        if vertical_distance <= band:
            if best_distance is None or vertical_distance < best_distance:
                best_distance = vertical_distance
                best_text = text
    return best_text


def _save_graphic(page, rect, page_index: int, graphic_index: int, graphics_dir: Path) -> tuple[str, str]:
    pix = page.get_pixmap(clip=rect, dpi=144)
    path = graphics_dir / f"p{page_index}_g{graphic_index}.png"
    pix.save(path)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return str(path), sha256


def _extract_graphics(page, page_index: int, blocks: list[dict[str, object]], graphics_dir: Path) -> list[Graphic]:
    graphics: list[Graphic] = []
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)

    def _to_bbox(rect) -> BBox:
        return _normalize_bbox([rect.x0, rect.y0, rect.x1, rect.y1], page_width, page_height)

    seen_xrefs: set[int] = set()
    graphic_index = 0
    for image in page.get_images(full=True):
        xref = image[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        try:
            rects = page.get_image_bbox(xref)
        except Exception:  # pragma: no cover
            continue
        if not rects:
            continue
        for rect in rects:
            try:
                path, sha = _save_graphic(page, rect, page_index, graphic_index, graphics_dir)
            except Exception:  # pragma: no cover
                continue
            nearby_text = _find_nearby_text(blocks, rect, page_height)
            graphics.append(
                Graphic(
                    page=page_index,
                    bbox=_to_bbox(rect),
                    nearby_text=nearby_text,
                    path=path,
                    sha256=sha,
                )
            )
            graphic_index += 1

    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is None:
            continue
        try:
            path, sha = _save_graphic(page, rect, page_index, graphic_index, graphics_dir)
        except Exception:  # pragma: no cover
            continue
        nearby_text = _find_nearby_text(blocks, rect, page_height)
        graphics.append(
            Graphic(
                page=page_index,
                bbox=_to_bbox(rect),
                nearby_text=nearby_text,
                path=path,
                sha256=sha,
            )
        )
        graphic_index += 1

    return graphics


def extract_pages(pdf_path: str | Path) -> tuple[str, list[Page], list[str], dict[str, object]]:
    path = Path(pdf_path)
    if not path.exists():
        msg = f"PDF file does not exist: {path}"
        raise FileNotFoundError(msg)
    doc_hash = compute_doc_hash(path)
    try:
        import fitz  # type: ignore
    except ImportError:  # pragma: no cover
        text = _extract_text_from_pdf_stream(path)
        fallback_page = Page(
            index=0,
            text_blocks=[text],
            bbox_blocks=[BBox(0.0, 0.0, 1.0, 1.0)],
            notes=[],
            graphics=[],
        )
        cache_dir, _ = _ensure_pdf_dirs(doc_hash)
        artifact_path = cache_dir / "artifact.json"
        artifact_path.write_text(
            json.dumps(
                {
                    "doc_hash": doc_hash,
                    "pages": [fallback_page.to_dict()],
                },
                indent=2,
            )
        )
        return doc_hash, [fallback_page], [text], {"title": path.stem}

    cache_dir, graphics_dir = _ensure_pdf_dirs(doc_hash)
    pages: list[Page] = []
    page_texts: list[str] = []
    metadata: dict[str, object] = {"title": path.stem}
    with fitz.open(path) as doc:
        metadata.update({k: v for k, v in doc.metadata.items() if v})
        for page_index, page in enumerate(doc):
            page_width = float(page.rect.width)
            page_height = float(page.rect.height)
            block_cache = cache_dir / f"blocks_p{page_index}.json"
            block_dicts = _load_cached_blocks(block_cache)
            if not block_dicts:
                blocks = page.get_text("blocks")
                _save_blocks(block_cache, blocks)
                block_dicts = _load_cached_blocks(block_cache)
            normalized_blocks = [
                _normalize_bbox(block["bbox"], page_width, page_height) for block in block_dicts
            ]
            text_blocks = [str(block["text"]) for block in block_dicts]
            page_text = page.get_text("text")
            page_texts.append(page_text)
            notes = _detect_notes(block_dicts, normalized_blocks, page_text, page_index)
            graphics = _extract_graphics(page, page_index, block_dicts, graphics_dir)
            pages.append(
                Page(
                    index=page_index,
                    text_blocks=text_blocks,
                    bbox_blocks=normalized_blocks,
                    notes=notes,
                    graphics=graphics,
                )
            )
    artifact = {
        "doc_hash": doc_hash,
        "pages": [page.to_dict() for page in pages],
    }
    artifact_path = cache_dir / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2))
    return doc_hash, pages, page_texts, metadata


def _compose_document_text(page_texts: Sequence[str]) -> tuple[str, list[tuple[int, int, int]]]:
    if not page_texts:
        return "", []
    buffer: list[str] = []
    page_ranges: list[tuple[int, int, int]] = []
    cursor = 0
    for page_index, text in enumerate(page_texts):
        buffer.append(text)
        start = cursor
        cursor += len(text)
        page_ranges.append((page_index, start, cursor))
        if page_index < len(page_texts) - 1:
            buffer.append("\n\n")
            cursor += 2
    return "".join(buffer), page_ranges


def _serialize_notes(doc_id: str, section_id: str | None, pages: Iterable[Page]) -> list[NoteRecord]:
    records: list[NoteRecord] = []
    for page in pages:
        for note in page.notes:
            records.append(
                NoteRecord(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    section_id=section_id,
                    kind=note.kind,
                    ref_anchor=note.ref,
                    content=note.text,
                    page=note.page,
                    bbox=note.bbox.to_list(),
                )
            )
    return records


def _serialize_graphics(doc_id: str, section_id: str | None, pages: Iterable[Page]) -> list[GraphicRecord]:
    records: list[GraphicRecord] = []
    for page in pages:
        for graphic in page.graphics:
            records.append(
                GraphicRecord(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    section_id=section_id,
                    caption=None,
                    nearby_text=graphic.nearby_text,
                    path=graphic.path,
                    sha256=graphic.sha256,
                    page=graphic.page,
                    bbox=graphic.bbox.to_list(),
                )
            )
    return records


def _cache_segments(doc_hash: str, segments: Sequence[MarkdownChunk]) -> None:
    payload = [
        {
            "id": segment.id,
            "content": segment.content,
            "token_count": segment.token_count,
            "char_start": segment.char_start,
            "char_end": segment.char_end,
            "start_page": segment.start_page,
            "end_page": segment.end_page,
            "start_line": segment.start_line,
            "end_line": segment.end_line,
        }
        for segment in segments
    ]
    path = _CACHE_SETTINGS.cache_llm_dir / f"{doc_hash}_segments.json"
    path.write_text(json.dumps(payload, indent=2))


def _cache_embeddings(doc_hash: str, segments: Sequence[MarkdownChunk]) -> None:
    payload = [
        {
            "id": segment.id,
            "embedding": list(segment.embedding),
        }
        for segment in segments
    ]
    path = _CACHE_SETTINGS.cache_emb_dir / f"{doc_hash}_embeddings.json"
    path.write_text(json.dumps(payload, indent=2))


def _cache_tables(doc_hash: str, tables: Sequence[TableMetadataRecord]) -> None:
    payload = [
        {
            "id": table.id,
            "table_name": table.table_name,
            "caption": table.caption,
        }
        for table in tables
    ]
    path = _CACHE_SETTINGS.cache_tables_dir / f"{doc_hash}_tables.json"
    path.write_text(json.dumps(payload, indent=2))


class PdfIngestPipeline:
    """High-level orchestrator for PDF ingestion."""

    def __init__(self, database: Database, settings: Settings | None = None) -> None:
        self.database = database
        self.settings = settings or _CACHE_SETTINGS
        self.segmenter = SemanticSegmenter(
            target_tokens=self.settings.chunk_target_tokens,
            overlap_ratio=self.settings.chunk_overlap_ratio,
        )

    def ingest(self, pdf_path: str | Path, title: str | None = None) -> IngestionArtifacts:
        pdf = Path(pdf_path)
        doc_hash, pages, page_texts, metadata = extract_pages(pdf)
        document_text, page_ranges = _compose_document_text(page_texts)
        if not document_text.strip():
            msg = f"No text extracted from document: {pdf}"
            raise ValueError(msg)

        document_id = str(uuid.uuid4())
        doc_title = title or metadata.get("title") or pdf.stem
        document = DocumentRecord(
            id=document_id,
            title=str(doc_title),
            sha256=doc_hash,
            meta={"source_path": str(pdf)},
            created_at=datetime.utcnow(),
        )

        root_section_id = str(uuid.uuid4())
        sections = [
            SectionRecord(
                id=root_section_id,
                document_id=document_id,
                title=str(doc_title),
                level=1,
                start_page=page_ranges[0][0] if page_ranges else 0,
                end_page=page_ranges[-1][0] if page_ranges else 0,
                path=str(doc_title),
            )
        ]

        segments = self.segmenter.segment(document_text)
        markdowns: list[MarkdownChunk] = []
        for segment in segments:
            content = document_text[segment.start : segment.end].strip()
            if not content:
                continue
            start_page, end_page = locate_pages(page_ranges, segment.start, segment.end)
            markdowns.append(
                MarkdownChunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    section_id=root_section_id,
                    content=content,
                    token_count=segment.token_count,
                    char_start=segment.start,
                    char_end=segment.end,
                    start_page=start_page,
                    end_page=end_page,
                    start_line=char_to_line(document_text, segment.start),
                    end_line=char_to_line(document_text, segment.end),
                    embedding=deterministic_embedding(content),
                    tsv=build_tsvector(content),
                )
            )
        if not markdowns:
            msg = "Segmentation yielded no chunks"
            raise ValueError(msg)

        notes = _serialize_notes(document_id, root_section_id, pages)
        graphics = _serialize_graphics(document_id, root_section_id, pages)
        tables: list[TableMetadataRecord] = []

        self.database.insert_document_bundle(document, sections, markdowns, notes, graphics, tables)

        _cache_segments(doc_hash, markdowns)
        _cache_embeddings(doc_hash, markdowns)
        _cache_tables(doc_hash, tables)

        artifact_path = _CACHE_SETTINGS.cache_pdf_dir / doc_hash / "artifact.json"
        return IngestionArtifacts(
            doc_hash=doc_hash,
            artifact_path=artifact_path,
            page_count=len(pages),
        )
