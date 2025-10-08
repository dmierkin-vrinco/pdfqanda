"""Persistence-agnostic vector index with Chroma and NumPy fallbacks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import chromadb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    chromadb = None  # type: ignore[assignment]


@dataclass(slots=True)
class VectorItem:
    """Payload for vector index updates."""

    id: str
    embedding: Sequence[float]
    metadata: dict[str, object]


class VectorIndex:
    """Simple vector store that prefers Chroma but falls back to NumPy arrays."""

    def __init__(self, base_path: Path, name: str = "kb") -> None:
        self.name = name
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._backend: _BaseBackend
        if chromadb is not None:
            try:  # pragma: no cover - optional dependency
                self._backend = _ChromaBackend(self.base_path, name)
            except Exception:
                self._backend = _NumpyBackend(self.base_path, name)
        else:
            self._backend = _NumpyBackend(self.base_path, name)

    # ------------------------------------------------------------------
    def upsert(self, items: Iterable[VectorItem]) -> None:
        payload = list(items)
        if not payload:
            return
        self._backend.upsert(payload)

    def delete(self, ids: Iterable[str]) -> None:
        ids = [item for item in ids]
        if not ids:
            return
        self._backend.delete(ids)

    def search(self, embedding: Sequence[float], limit: int | None = None) -> list[tuple[str, float]]:
        return self._backend.search(embedding, limit)

    def count(self) -> int:
        return self._backend.count()

    def get_embeddings(self, ids: Iterable[str]) -> dict[str, list[float]]:
        return self._backend.get_embeddings(ids)


class _BaseBackend:
    def upsert(self, items: list[VectorItem]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def delete(self, ids: list[str]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def search(self, embedding: Sequence[float], limit: int | None) -> list[tuple[str, float]]:  # pragma: no cover - interface
        raise NotImplementedError

    def count(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def get_embeddings(self, ids: Iterable[str]) -> dict[str, list[float]]:  # pragma: no cover - interface
        raise NotImplementedError


class _NumpyBackend(_BaseBackend):
    """Persist vectors to disk as normalised NumPy arrays."""

    def __init__(self, base_path: Path, name: str) -> None:
        self.root = Path(base_path)
        self.vectors_path = self.root / f"{name}_vectors.npy"
        self.meta_path = self.root / f"{name}_meta.json"
        self.ids: list[str] = []
        self.metadata: dict[str, dict[str, object]] = {}
        self.dimension: int | None = None
        self.vectors: np.ndarray | None = None
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self.meta_path.exists():
            payload = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.ids = list(payload.get("ids", []))
            self.metadata = {str(k): v for k, v in payload.get("metadata", {}).items()}
            self.dimension = payload.get("dimension")
        if self.vectors_path.exists():
            self.vectors = np.load(self.vectors_path, allow_pickle=False)
            if self.vectors.size == 0:
                self.vectors = np.empty((0, 0), dtype=np.float32)
        else:
            self.vectors = np.empty((0, 0), dtype=np.float32)
        if self.vectors.size and self.dimension is None:
            self.dimension = int(self.vectors.shape[1])

    def _persist(self) -> None:
        if self.vectors is None:
            self.vectors = np.empty((0, 0), dtype=np.float32)
        np.save(self.vectors_path, self.vectors, allow_pickle=False)
        payload = {
            "ids": self.ids,
            "metadata": self.metadata,
            "dimension": self.dimension,
        }
        self.meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    def upsert(self, items: list[VectorItem]) -> None:
        if self.vectors is None or self.vectors.size == 0:
            self.vectors = np.empty((0, 0), dtype=np.float32)
        vectors = self.vectors
        id_to_index = {id_: idx for idx, id_ in enumerate(self.ids)}
        for item in items:
            vector = np.asarray(item.embedding, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                raise ValueError("Cannot index zero-length embedding")
            vector = vector / norm
            if self.dimension is None:
                self.dimension = vector.shape[0]
            if vector.shape[0] != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.dimension}, got {vector.shape[0]}"
                )
            if item.id in id_to_index:
                idx = id_to_index[item.id]
                vectors[idx] = vector
            else:
                if vectors.size == 0:
                    vectors = vector.reshape(1, -1)
                else:
                    if vector.shape[0] != vectors.shape[1]:
                        raise ValueError(
                            f"Embedding dimension mismatch: expected {vectors.shape[1]}, got {vector.shape[0]}"
                        )
                    vectors = np.vstack([vectors, vector])
                self.ids.append(item.id)
                id_to_index[item.id] = len(self.ids) - 1
            self.metadata[item.id] = dict(item.metadata)
        self.vectors = vectors
        self._persist()

    def delete(self, ids: list[str]) -> None:
        if self.vectors is None or not self.ids:
            return
        remove = {id_ for id_ in ids}
        keep_indices = [idx for idx, id_ in enumerate(self.ids) if id_ not in remove]
        if len(keep_indices) == len(self.ids):
            return
        if keep_indices:
            self.vectors = self.vectors[keep_indices]
        else:
            self.vectors = np.empty((0, self.vectors.shape[1] if self.vectors.size else 0), dtype=np.float32)
        self.ids = [self.ids[idx] for idx in keep_indices]
        for id_ in ids:
            self.metadata.pop(id_, None)
        self._persist()

    def search(self, embedding: Sequence[float], limit: int | None) -> list[tuple[str, float]]:
        if self.vectors is None or self.vectors.size == 0:
            return []
        query = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return []
        query = query / norm
        scores = self.vectors @ query
        limit = limit or scores.shape[0]
        limit = min(limit, scores.shape[0])
        order = np.argsort(scores)[::-1][:limit]
        return [(self.ids[idx], float(scores[idx])) for idx in order]

    def count(self) -> int:
        return len(self.ids)

    def get_embeddings(self, ids: Iterable[str]) -> dict[str, list[float]]:
        if self.vectors is None or self.vectors.size == 0:
            return {}
        id_to_index = {id_: idx for idx, id_ in enumerate(self.ids)}
        output: dict[str, list[float]] = {}
        for id_ in ids:
            idx = id_to_index.get(id_)
            if idx is None:
                continue
            output[id_] = self.vectors[idx].astype(float).tolist()
        return output


class _ChromaBackend(_BaseBackend):  # pragma: no cover - optional dependency
    def __init__(self, base_path: Path, name: str) -> None:
        self.root = Path(base_path) / "chroma"
        self.root.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.root))
        self.collection = self.client.get_or_create_collection(name)

    def upsert(self, items: list[VectorItem]) -> None:
        self.collection.upsert(
            ids=[item.id for item in items],
            embeddings=[list(map(float, item.embedding)) for item in items],
            metadatas=[dict(item.metadata) for item in items],
        )

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.collection.delete(ids=ids)

    def search(self, embedding: Sequence[float], limit: int | None) -> list[tuple[str, float]]:
        count = self.collection.count()
        if count == 0:
            return []
        limit = limit or count
        result = self.collection.query(query_embeddings=[list(map(float, embedding))], n_results=min(limit, count))
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        scores = [1.0 - float(dist) for dist in distances]
        return list(zip(ids, scores))

    def count(self) -> int:
        return self.collection.count()

    def get_embeddings(self, ids: Iterable[str]) -> dict[str, list[float]]:
        ids = list(ids)
        if not ids:
            return {}
        result = self.collection.get(ids=ids, include=["embeddings"])
        output: dict[str, list[float]] = {}
        for idx, id_ in enumerate(result.get("ids", [])):
            embeddings = result.get("embeddings", [])
            if idx < len(embeddings):
                output[id_] = list(map(float, embeddings[idx]))
        return output


__all__ = ["VectorIndex", "VectorItem"]
