"""Simple TF-IDF backed vector store for text chunks."""

from __future__ import annotations

import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from .splitter import TextChunk


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _compute_norm(vector: Dict[str, float]) -> float:
    return math.sqrt(sum(weight * weight for weight in vector.values()))


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) < len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    dot = sum(weight * vec_b.get(term, 0.0) for term, weight in vec_a.items())
    norm_a = _compute_norm(vec_a)
    norm_b = _compute_norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class VectorStoreResult:
    chunk: TextChunk
    score: float


class TfidfVectorStore:
    """In-memory TF-IDF vector store using a lightweight implementation."""

    def __init__(self) -> None:
        self._vectors: list[Dict[str, float]] = []
        self._chunks: list[TextChunk] = []
        self._idf: Dict[str, float] = {}

    @property
    def is_fit(self) -> bool:
        return bool(self._vectors) and bool(self._chunks)

    def fit(self, chunks: Iterable[TextChunk]) -> None:
        chunk_list = list(chunks)
        if not chunk_list:
            msg = "At least one chunk is required to build the index"
            raise ValueError(msg)

        tokenized: list[List[str]] = [_tokenize(chunk.content) for chunk in chunk_list]
        document_frequency: Dict[str, int] = {}
        for tokens in tokenized:
            seen = set(tokens)
            for token in seen:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        total_docs = len(chunk_list)
        self._idf = {
            term: math.log((1 + total_docs) / (1 + df)) + 1.0
            for term, df in document_frequency.items()
        }

        vectors: list[Dict[str, float]] = []
        for tokens in tokenized:
            counts: Dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            length = len(tokens) or 1
            vector: Dict[str, float] = {}
            for token, count in counts.items():
                tf = count / length
                idf = self._idf.get(token, 0.0)
                vector[token] = tf * idf
            vectors.append(vector)

        self._vectors = vectors
        self._chunks = chunk_list

    def _vectorize_query(self, text: str) -> Dict[str, float]:
        tokens = _tokenize(text)
        if not tokens:
            return {}
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        length = len(tokens)
        vector: Dict[str, float] = {}
        for token, count in counts.items():
            idf = self._idf.get(token, 0.0)
            if idf == 0:
                continue
            tf = count / length
            vector[token] = tf * idf
        return vector

    def query(self, text: str, top_k: int = 3) -> list[VectorStoreResult]:
        if not self.is_fit:
            msg = "Vector store must be fit before querying"
            raise ValueError(msg)
        if not text:
            return []

        query_vector = self._vectorize_query(text)
        scored = [
            (index, _cosine_similarity(query_vector, vector))
            for index, vector in enumerate(self._vectors)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        top_results = scored[:top_k]
        return [VectorStoreResult(chunk=self._chunks[index], score=score) for index, score in top_results]

    def save(self, path: str | Path) -> None:
        if not self.is_fit:
            msg = "Vector store must be fit before saving"
            raise ValueError(msg)
        data = {
            "vectors": self._vectors,
            "chunks": self._chunks,
            "idf": self._idf,
        }
        with Path(path).open("wb") as file:
            pickle.dump(data, file)

    @classmethod
    def load(cls, path: str | Path) -> "TfidfVectorStore":
        with Path(path).open("rb") as file:
            data = pickle.load(file)
        store = cls()
        store._vectors = data["vectors"]
        store._chunks = data["chunks"]
        store._idf = data["idf"]
        return store
