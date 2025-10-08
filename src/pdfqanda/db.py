"""Backwards-compatible import wrapper for the database utilities."""

from __future__ import annotations

from .util.db import Database

__all__ = ["Database"]
