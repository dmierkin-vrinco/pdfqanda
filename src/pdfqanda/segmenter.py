"""Semantic segmentation utilities."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence


@dataclass(slots=True)
class Segment:
    """Represents a contiguous range of text within the document."""

    start: int
    end: int
    token_count: int


class SemanticSegmenter:
    """Deterministic text segmenter approximating ~1k token windows with overlap."""

    def __init__(self, target_tokens: int = 1000, overlap_ratio: float = 0.12) -> None:
        if target_tokens <= 0:
            msg = "target_tokens must be positive"
            raise ValueError(msg)
        if not 0.0 <= overlap_ratio < 1.0:
            msg = "overlap_ratio must be in [0.0, 1.0)"
            raise ValueError(msg)
        self.target_tokens = target_tokens
        self.overlap_ratio = overlap_ratio

    _TOKEN_REGEX = re.compile(r"\S+\s*")

    def _iter_tokens(self, text: str) -> Iterator[tuple[int, int]]:
        for match in self._TOKEN_REGEX.finditer(text):
            yield match.start(), match.end()

    def segment(self, text: str) -> list[Segment]:
        """Segment ``text`` into overlapping windows."""

        token_positions = list(self._iter_tokens(text))
        if not token_positions:
            return []
        window = self.target_tokens
        step = max(1, int(math.ceil(window * (1.0 - self.overlap_ratio))))
        segments: list[Segment] = []
        for start_idx in range(0, len(token_positions), step):
            window_positions = token_positions[start_idx : start_idx + window]
            if not window_positions:
                continue
            start_char = window_positions[0][0]
            end_char = window_positions[-1][1]
            segments.append(Segment(start=start_char, end=end_char, token_count=len(window_positions)))
            if start_idx + window >= len(token_positions):
                break
        # Always include trailing remainder if not already captured.
        last_end = token_positions[-1][1]
        if not segments or segments[-1].end < last_end:
            segments.append(Segment(start=segments[-1].start if segments else 0, end=last_end, token_count=len(token_positions)))
        return segments


def char_to_line(text: str, index: int) -> int:
    """Return 1-indexed line number for ``index`` in ``text``."""

    return text.count("\n", 0, index) + 1


def locate_pages(page_ranges: Sequence[tuple[int, int, int]], start: int, end: int) -> tuple[int, int]:
    """Given ``page_ranges`` return the inclusive page span for ``start``/``end``."""

    if not page_ranges:
        return 0, 0
    start_page = page_ranges[0][0]
    end_page = page_ranges[-1][0]
    for page_index, page_start, page_end in page_ranges:
        if page_start <= start <= page_end:
            start_page = page_index
        if page_start <= end <= page_end:
            end_page = page_index
        if end > page_end:
            end_page = page_index
    return start_page, end_page
