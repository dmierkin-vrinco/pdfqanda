"""Compatibility helpers for embedding-related utilities."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Iterable, Sequence

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


@lru_cache(maxsize=4096)
def build_tsvector(text: str) -> str:
    """Produce a normalized token payload similar to SQL ``tsvector`` outputs."""

    tokens = {token.lower() for token in _TOKEN_PATTERN.findall(text)}
    ordered = sorted(tokens)
    return " ".join(ordered)


def count_term_hits(tsv: str, terms: Iterable[str]) -> int:
    """Return the number of terms that appear in ``tsv``."""

    token_set = set(tsv.split())
    hits = 0
    for term in terms:
        normalized = term.lower()
        if normalized in token_set:
            hits += 1
    return hits


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute the cosine similarity between two embedding vectors."""

    if len(a) != len(b):
        raise ValueError("Embedding dimension mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


__all__ = ["build_tsvector", "count_term_hits", "cosine_similarity"]
