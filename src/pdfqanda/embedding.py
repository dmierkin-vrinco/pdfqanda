"""Deterministic embedding and text utilities."""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Iterable, Sequence

EMBEDDING_DIM = 3072
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


@lru_cache(maxsize=4096)
def deterministic_embedding(text: str) -> tuple[float, ...]:
    """Return a deterministic pseudo-embedding for ``text``.

    The implementation avoids external API dependencies by folding the SHA256
    digest of ``text`` into a vector within [-1, 1]. While simplistic, this
    approach provides stable ordering for unit tests and caching layers.
    """

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for idx in range(EMBEDDING_DIM):
        byte = digest[idx % len(digest)]
        values.append((byte / 255.0) * 2.0 - 1.0)
    return tuple(values)


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


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
