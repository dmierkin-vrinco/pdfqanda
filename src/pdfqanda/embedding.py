"""Compatibility helpers for embedding-related utilities."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

from .util.embeddings import deterministic_embedding

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


@lru_cache(maxsize=4096)
def build_tsvector(text: str) -> str:
    """Produce a normalized token payload similar to Postgres ``tsvector``."""

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


__all__ = ["build_tsvector", "count_term_hits", "deterministic_embedding"]
