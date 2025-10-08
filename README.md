# pdfqanda

`pdfqanda` turns local PDFs into a cite-everything knowledge base. The M1 skeleton
ships a deterministic ingestion pipeline, a Postgres-friendly schema, and a
hybrid Researcher→Expert retrieval flow that refuses to answer without
citations.

## Highlights

- **Canonical schema** — `schema.sql` provisions `kb.*` and `pdf_tables.*` with
  pgvector-compatible embeddings and `tsvector` full text search indices.
- **Deterministic ingestion** — PyMuPDF powered extraction with cached blocks,
  footnote and graphic detection, and semantic segmentation (~1k token windows,
  12% overlap). Intermediate artifacts, embeddings, and table metadata are
  cached under `.cache/`.
- **Hybrid retrieval** — the Researcher agent ranks vector + lexical hits,
  surfaces reranked evidence, and scaffolds guarded `SELECT` SQL for table
  follow-ups.
- **Hard citation enforcement** — the Expert agent refuses to emit Markdown
  answers unless every bullet carries a `【doc:…】` citation.
- **CLI** — `pdfqanda ingest` pushes a document into the knowledge base; `pdfqanda
  ask` runs the Researcher→Expert chain and prints cited Markdown (with optional
  JSON export).

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
`PDFQANDA_DATABASE_URL` before running commands:

```bash
export PDFQANDA_DATABASE_URL="postgresql://user:password@localhost:5432/pdfqanda"
```

Run the schema once (for Postgres deployments):

```bash
psql "$PDFQANDA_DATABASE_URL" -f schema.sql
```

### Ingesting a PDF

If you just want to see the pipeline end-to-end, the repository ships a
curated [`input/sample.pdf`](input/sample.pdf) fixture that spans narrative
text, tables, and annotated graphics. It doubles as the integration fixture used
by the test suite, so running the commands below will match what CI exercises.

```bash
pdfqanda ingest path/to/document.pdf
```

This extracts pages, notes, graphics, and semantic chunks, writes canonical rows
in `kb.*`, and caches artifacts beneath `.cache/` (including normalized page
metadata, segmentation JSON, and deterministic embeddings).

### Asking a Question

```bash
pdfqanda ask "What is the executive summary?"
```

The Researcher retrieves hybrid evidence, the Expert assembles Markdown with
citations, and the CLI prints the answer. Use `--json` for structured output.

## Testing

```bash
pytest
```

The smoke suite ingests the bundled sample PDF, verifies vectors land in
`kb_markdowns`, exercises the Researcher→Expert flow, and ensures missing
citations hard-fail.

## Development Notes

- Cached artifacts live under `.cache/pdf`, `.cache/llm`, `.cache/emb`, and
  `.cache/tables`.
- Embeddings use a deterministic SHA256-based projection (stand-in for
  `text-embedding-3-large`) so unit tests remain offline and reproducible.
- The Researcher enforces SELECT-only SQL scaffolding and returns an `exhausted`
  flag when fewer results than requested are available.

## License

MIT — see [LICENSE](LICENSE).
