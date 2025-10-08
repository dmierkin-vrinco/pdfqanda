"""Embedding helpers backed by the OpenAI embeddings API."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from openai import OpenAI

from .cache import FileCache


@dataclass(slots=True)
class EmbeddingClient:
    """High level embedding helper with caching."""

    model: str
    dimension: int
    cache: FileCache | None = None
    client: object | None = None
    _client: object = field(init=False)

    def __post_init__(self) -> None:
        base = Path(".cache/llm")
        base.mkdir(parents=True, exist_ok=True)
        if self.cache is None:
            self.cache = FileCache(base)
        if self.client is not None:
            self._client = self.client
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY must be set to use OpenAI embeddings")
            self._client = OpenAI()

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
        if self._client is None:
            raise RuntimeError("OpenAI client is not initialised")
        response = self._client.embeddings.create(model=self.model, input=[text])
        vector = response.data[0].embedding
        if len(vector) != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {len(vector)}"
            )
        return list(map(float, vector))

    # Convenience wrappers -------------------------------------------------
    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_texts(texts)
