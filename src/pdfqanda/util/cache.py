"""File-system backed cache utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


def stable_hash(parts: Sequence[str]) -> str:
    """Return a deterministic SHA256 hash over the provided string parts."""

    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


@dataclass(slots=True)
class FileCache:
    """Simple directory-backed cache for structured data."""

    base_dir: Path

    def __post_init__(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, namespace: str, key: str) -> Path:
        digest = stable_hash([key])
        return self.base_dir / namespace / f"{digest}.json"

    def get(self, namespace: str, key: str) -> Any | None:
        path = self._key_path(namespace, key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return data

    def set(self, namespace: str, key: str, value: Any) -> None:
        path = self._key_path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def get_or_compute(
        self,
        namespace: str,
        key: str,
        factory: Callable[[], Any],
    ) -> Any:
        cached = self.get(namespace, key)
        if cached is not None:
            return cached
        value = factory()
        self.set(namespace, key, value)
        return value

    def purge(self, namespace: str | None = None) -> None:
        targets: Iterable[Path]
        if namespace is None:
            targets = [self.base_dir]
        else:
            targets = [self.base_dir / namespace]
        for target in targets:
            if not target.exists():
                continue
            for path in target.rglob("*"):
                if path.is_file():
                    path.unlink()
            # remove empty directories bottom up
            for path in sorted(target.rglob("*"), reverse=True):
                if path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass


__all__ = ["FileCache", "stable_hash"]
