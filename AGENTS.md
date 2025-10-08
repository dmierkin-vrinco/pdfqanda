# Repository Guidelines

Welcome! This project ingests PDF documents into a SQLite-backed knowledge base
and exposes a CLI for retrieval with mandatory citations.

## Development workflow
- Prefer Python 3.10+ and install dependencies with `pip install -e .[dev]`.
- Clear `pdfqanda.config.get_settings` cache after mutating environment variables
during tests: `from pdfqanda.config import get_settings; get_settings.cache_clear()`.
- Run the full test suite with `pytest` before committing. The smoke tests run the
real ingestion pipeline against fixture PDFs—no mocks.
- When changing the schema or database behavior, update the SQLite schema in
  `schema.sql` and any docs/tests that mention the affected columns.

## Database setup
- CLI commands auto-create `pdfqanda.db` in the working directory when `DB_PATH`
  is left unset.
- The embedding dimension is 1536 by default, matching `text-embedding-3-small`.
  Keep SQLite JSON columns in sync with this value.

## CLI reminders
- `pdfqanda ingest <pdf path>` loads PDFs into the knowledge base.
- `pdfqanda ask "<question>"` prints a cited answer and exits non-zero if no
  citations are present.
- Cached artifacts live under `.cache/pdf/` and `.cache/llm/`—they may be safely
  deleted between runs if needed.

