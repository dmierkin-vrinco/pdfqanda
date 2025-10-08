"""Simple migration runner for SQLite-backed databases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Migration:
    """Represents a SQL migration that can be applied exactly once."""

    identifier: str
    statements: str


class MigrationRunner:
    """Apply SQL migrations while recording which ones ran."""

    def __init__(self, connection) -> None:  # pragma: no cover - runtime connection
        self.connection = connection

    def apply(self, migrations: Sequence[Migration]) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        applied = {
            row[0]
            for row in cursor.execute("SELECT id FROM schema_migrations")
        }
        for migration in migrations:
            if migration.identifier in applied:
                continue
            cursor.executescript(migration.statements)
            cursor.execute(
                "INSERT INTO schema_migrations (id) VALUES (?)",
                (migration.identifier,),
            )
        self.connection.commit()


def apply_migrations(connection, migrations: Iterable[Migration]) -> None:
    """Helper for applying migrations without instantiating runner directly."""

    MigrationRunner(connection).apply(list(migrations))


__all__ = ["Migration", "MigrationRunner", "apply_migrations"]
