"""Configuration helpers for pdfqanda."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime settings derived from environment variables."""

    database_url: str
    chunk_target_tokens: int
    chunk_overlap_ratio: float
    cache_dir: Path
    cache_pdf_dir: Path
    cache_llm_dir: Path
    cache_emb_dir: Path
    cache_tables_dir: Path


_DEFAULT_CACHE_ROOT = Path(os.environ.get("PDFQANDA_CACHE", ".cache"))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_settings() -> Settings:
    """Return cached runtime settings."""

    database_url = os.environ.get("PDFQANDA_DATABASE_URL", "sqlite:///pdfqanda.db")
    chunk_target_tokens = int(os.environ.get("PDFQANDA_CHUNK_TOKENS", "1000"))
    overlap_ratio = float(os.environ.get("PDFQANDA_CHUNK_OVERLAP", "0.12"))

    cache_dir = _ensure_dir(_DEFAULT_CACHE_ROOT)
    cache_pdf = _ensure_dir(cache_dir / "pdf")
    cache_llm = _ensure_dir(cache_dir / "llm")
    cache_emb = _ensure_dir(cache_dir / "emb")
    cache_tables = _ensure_dir(cache_dir / "tables")

    return Settings(
        database_url=database_url,
        chunk_target_tokens=chunk_target_tokens,
        chunk_overlap_ratio=overlap_ratio,
        cache_dir=cache_dir,
        cache_pdf_dir=cache_pdf,
        cache_llm_dir=cache_llm,
        cache_emb_dir=cache_emb,
        cache_tables_dir=cache_tables,
    )
