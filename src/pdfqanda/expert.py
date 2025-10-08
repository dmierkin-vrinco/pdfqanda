"""Expert agent that enforces hard citations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .models import ResearchHit


class CitationError(RuntimeError):
    """Raised when the expert cannot satisfy citation requirements."""


@dataclass(slots=True)
class Expert:
    """Compose Markdown answers backed by Researcher evidence."""

    max_sentences: int = 2

    SENTENCE_REGEX = re.compile(r"(?<=[.!?])\s+")

    def compose_answer(self, question: str, hits: Sequence[ResearchHit]) -> str:
        if not hits:
            raise CitationError("No evidence supplied for answer")

        bullet_lines: list[str] = []
        for hit in hits:
            summary = self._summarize(hit.content)
            if not summary:
                continue
            bullet_lines.append(f"- {summary} {hit.citation}")
        if not bullet_lines:
            raise CitationError("Unable to summarize evidence with citations")

        answer = "\n".join(["### Answer", "", *bullet_lines])
        self._validate(answer)
        return answer

    def _summarize(self, content: str) -> str:
        text = content.strip()
        if not text:
            return ""
        sentences = self.SENTENCE_REGEX.split(text)
        selected = sentences[: self.max_sentences]
        summary = " ".join(sentence.strip() for sentence in selected if sentence.strip())
        return summary[:500]

    def _validate(self, markdown: str) -> None:
        if "【" not in markdown or "】" not in markdown:
            raise CitationError("Missing citation markers")
        for line in markdown.splitlines():
            stripped = line.strip()
            if stripped.startswith("-") and "【" not in stripped:
                raise CitationError("Evidence bullet lacks citation")
