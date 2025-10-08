"""Embedding helpers backed by OpenAI with deterministic fallbacks."""

from __future__ import annotations

import hashlib
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]

from .cache import FileCache


def deterministic_embedding(text: str, dim: int) -> list[float]:
    """Generate a deterministic pseudo-embedding for offline scenarios."""

    seed = hashlib.sha256(text.encode("utf-8")).digest()
    rng = random.Random(seed)
    values = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


@dataclass(slots=True)
class EmbeddingClient:
    """High level embedding helper with caching."""

    model: str
    dimension: int
    cache: FileCache | None = None
    _client: object | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        base = Path(".cache/llm")
        base.mkdir(parents=True, exist_ok=True)
        if self.cache is None:
            self.cache = FileCache(base)
        if OpenAI is not None and os.getenv("OPENAI_API_KEY"):
            try:  # pragma: no cover - network not available in CI
                self._client = OpenAI()
            except Exception:
                self._client = None

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        outputs: list[list[float]] = []
        for text in texts:
            key = f"{self.model}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
            cached = self.cache.get("embeddings", key) if self.cache else None
            if cached is not None:
                outputs.append([float(v) for v in cached])
                continue
            embedding = self._embed_single(text)
            if self.cache is not None:
                self.cache.set("embeddings", key, embedding)
            outputs.append(embedding)
        return outputs

    # ------------------------------------------------------------------
    def _embed_single(self, text: str) -> list[float]:
        if self._client is not None:  # pragma: no cover - requires network
            response = self._client.embeddings.create(model=self.model, input=[text])
            vector = response.data[0].embedding
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.dimension}, got {len(vector)}"
                )
            return list(map(float, vector))
        return deterministic_embedding(text, self.dimension)

    # Convenience wrappers -------------------------------------------------
    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_texts(texts)
