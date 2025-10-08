"""Runtime configuration helpers for pdfqanda."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _load_env_file() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key, value)


_load_env_file()


@dataclass(frozen=True)
class Settings:
    """Typed wrapper around environment-driven configuration."""

    db_dsn: str
    embedding_model: str
    embedding_dim: int
    chunk_target_tokens: int
    chunk_overlap_ratio: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached configuration values loaded from the environment."""

    db_dsn = os.getenv("DB_DSN", "sqlite:///pdfqanda.db")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))
    chunk_target = int(os.getenv("CHUNK_TARGET_TOKENS", "1000"))
    overlap_ratio = float(os.getenv("CHUNK_OVERLAP_RATIO", "0.12"))

    return Settings(
        db_dsn=db_dsn,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        chunk_target_tokens=chunk_target,
        chunk_overlap_ratio=overlap_ratio,
    )
