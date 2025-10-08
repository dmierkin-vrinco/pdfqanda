# pdfqanda

`pdfqanda` turns local PDFs into a cite-everything knowledge base. The current
MVP ships a deterministic ingestion pipeline, a Postgres/pgvector friendly
schema, and a CLI for loading PDFs and retrieving cited snippets.

## Highlights

- **Canonical schema** — `schema.sql` provisions `kb.*` tables with
  pgvector-compatible embedding columns, HNSW indexes, and `tsvector` full-text
  search triggers.
- **Deterministic ingestion** — PyMuPDF-powered extraction (with a lightweight
  pure-Python fallback) produces paragraph chunks (~1k token windows with ~12 %
  overlap), stores them in Postgres (or a SQLite fallback), and populates
  embeddings via the deterministic stand-in in `embedding.py`.
- **Vector + lexical retrieval** — `Retriever.search` runs cosine similarity via
  pgvector (or a cosine implementation on SQLite) and can optionally filter on
  keyword matches using the stored `tsv` payloads.
- **CLI** — `pdfqanda db init` creates schemas, `pdfqanda ingest` pushes a PDF
  into the knowledge base, and `pdfqanda ask` prints the top cited snippets.

## Getting Started

### Prerequisites

- Python 3.10+
- Optional: Postgres 15+ with the [pgvector](https://github.com/pgvector/pgvector)
  extension enabled.

### Installation

```bash
pip install -e .[dev]
```

### Database

By default the CLI uses `sqlite:///pdfqanda.db`. To target Postgres set
`DB_DSN` before running commands:

```bash
export DB_DSN="postgresql://user:password@localhost:5432/pdfqanda"
```

Run the schema once (for Postgres deployments) using the bundled CLI command:

```bash
pdfqanda db init
```

### Ingesting a PDF

```bash
pdfqanda ingest path/to/document.pdf
```

This extracts paragraphs, writes canonical rows in `kb.*`, and stores
deterministic embeddings alongside the text chunks.

### Asking a Question

```bash
pdfqanda ask "What is the executive summary?"
```

The retriever returns the highest-scoring chunks along with citations in the
form `【doc:… §… p.…】`.

## Testing

```bash
pytest
```

The smoke suite ingests fixture PDFs, verifies rows exist in `kb.markdowns`, and
asserts that `ask` returns cited snippets.

## Development Notes

- Embeddings use a deterministic SHA256-based projection (stand-in for
  `text-embedding-3-large`) so unit tests remain offline and reproducible.
- The SQLite fallback stores vectors as JSON and skips full-text triggers; it is
  intended for local testing only.

## License

MIT — see [LICENSE](LICENSE).
