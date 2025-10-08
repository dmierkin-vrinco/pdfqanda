"""Utilities for loading text and artifact metadata from PDF documents."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Iterable, Sequence

from .models import BBox, Graphic, Note, Page

_CACHE_ROOT = Path(".cache/pdf")
_ARTIFACT_ROOT = Path("artifacts")
_FOOTNOTE_PATTERN = re.compile(r"^(\d+[\.\)]\s+|\*+|[†]+)")
_SUPERSCRIPT_PATTERN = re.compile(r"\^(\d+)")
_REF_SECTION_TITLES = {"references", "notes", "footnotes"}

_STREAM_PATTERN = re.compile(rb"stream(.*?)endstream", re.DOTALL)
_TEXT_PATTERN = re.compile(rb"\((.*?)\)")


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


def compute_doc_hash(path: str | Path) -> str:
    """Return a deterministic hash of the PDF file bytes."""

    pdf_path = Path(path)
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _ensure_dirs(doc_hash: str) -> tuple[Path, Path]:
    cache_dir = _CACHE_ROOT / doc_hash
    artifact_dir = _ARTIFACT_ROOT / doc_hash
    graphics_dir = artifact_dir / "graphics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    if graphics_dir.exists():
        # Clear stale graphics for deterministic outputs
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


def _load_or_compute_blocks(page, cache_dir: Path, page_index: int) -> list[dict[str, object]]:
    block_cache = cache_dir / f"blocks_p{page_index}.json"
    cached = _load_cached_blocks(block_cache)
    if cached:
        return cached
    blocks = page.get_text("blocks")
    _save_blocks(block_cache, blocks)
    return _load_cached_blocks(block_cache)


def _cache_page_image(page, cache_dir: Path, page_index: int) -> None:
    image_path = cache_dir / f"p{page_index}.png"
    if image_path.exists():
        return
    pix = page.get_pixmap(dpi=144)
    pix.save(image_path)


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


def _find_nearby_text(
    blocks: list[dict[str, object]],
    rect,
    page_height: float,
) -> str:
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


def _extract_graphics(
    page,
    page_index: int,
    blocks: list[dict[str, object]],
    graphics_dir: Path,
) -> list[Graphic]:
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
        except Exception:  # pragma: no cover - PyMuPDF specific failures
            continue
        if not rects:
            continue
        for rect in rects:
            try:
                path, sha = _save_graphic(page, rect, page_index, graphic_index, graphics_dir)
            except Exception:  # pragma: no cover - rendering failures
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


def extract_pages(pdf_path: str | Path) -> tuple[str, list[Page]]:
    """Extract page-level metadata for a PDF document."""

    path = Path(pdf_path)
    if not path.exists():
        msg = f"PDF file does not exist: {path}"
        raise FileNotFoundError(msg)

    doc_hash = compute_doc_hash(path)

    try:
        import fitz  # type: ignore
    except ImportError:  # pragma: no cover - dependency missing
        text = _extract_text_from_pdf_stream(path)
        fallback_page = Page(
            index=0,
            text_blocks=[text],
            bbox_blocks=[BBox(0.0, 0.0, 1.0, 1.0)],
            notes=[],
            graphics=[],
        )
        return doc_hash, [fallback_page]

    cache_dir, graphics_dir = _ensure_dirs(doc_hash)

    pages: list[Page] = []
    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc):
            page_width = float(page.rect.width)
            page_height = float(page.rect.height)
            _cache_page_image(page, cache_dir, page_index)
            block_dicts = _load_or_compute_blocks(page, cache_dir, page_index)
            normalized_blocks = [
                _normalize_bbox(block["bbox"], page_width, page_height) for block in block_dicts
            ]
            text_blocks = [str(block["text"]) for block in block_dicts]
            page_text = page.get_text("text")
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

    return doc_hash, pages


def write_artifact(doc_hash: str, pages: Sequence[Page]) -> Path:
    """Persist page metadata to an artifact JSON file."""

    _ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    artifact_path = _ARTIFACT_ROOT / f"{doc_hash}.json"
    payload = {
        "doc_hash": doc_hash,
        "pages": [page.to_dict() for page in pages],
    }
    artifact_path.write_text(json.dumps(payload, indent=2))
    return artifact_path


def build_document_artifact(pdf_path: str | Path) -> tuple[str, Path, list[Page]]:
    """Extract metadata for ``pdf_path`` and write the artifact to disk."""

    doc_hash, pages = extract_pages(pdf_path)
    artifact_path = write_artifact(doc_hash, pages)
    return doc_hash, artifact_path, pages


def extract_text_from_pdf(path: str | Path) -> str:
    """Extract text content from a PDF, preferring PyMuPDF with a stream fallback."""

    pdf_path = Path(path)
    if not pdf_path.exists():
        msg = f"PDF file does not exist: {pdf_path}"
        raise FileNotFoundError(msg)

    try:
        import fitz  # type: ignore
    except ImportError:  # pragma: no cover - dependency missing
        return _extract_text_from_pdf_stream(pdf_path)

    try:
        with fitz.open(pdf_path) as doc:
            texts = [page.get_text("text").strip() for page in doc]
        return "\n\n".join(filter(None, texts))
    except Exception:
        return _extract_text_from_pdf_stream(pdf_path)


def load_documents(paths: Iterable[str | Path]) -> str:
    """Load and concatenate text from multiple PDF documents."""

    contents: list[str] = []
    for path in paths:
        contents.append(extract_text_from_pdf(path))
    return "\n\n".join(filter(None, contents))
