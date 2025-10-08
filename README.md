# pdfqanda

`pdfqanda` turns local PDFs into a cite-everything knowledge base. The current
MVP ships a SQLite-friendly schema and a CLI for loading PDFs and retrieving
cited snippets backed by OpenAI embeddings.

## Highlights

- **Canonical schema** — `schema.sql` provisions `kb_*` tables with SQLite
  columns for cached embeddings and lightweight token payloads populated during
  ingestion.
- **Ingestion pipeline** — PyMuPDF-powered extraction (with a lightweight
  pure-Python fallback) produces paragraph chunks (~1k token windows with ~12 %
  overlap), stores them in SQLite, and populates embeddings via
  `text-embedding-3-small`.
- **Vector + lexical retrieval** — `Retriever.search` runs cosine similarity via
  the bundled vector index and can optionally filter on keyword matches using
  the stored `tsv` payloads.
- **CLI** — `pdfqanda db init` ensures the SQLite schema exists, `pdfqanda
  ingest` pushes a PDF into the knowledge base, and `pdfqanda ask` prints the top
  cited snippets.

## Getting Started

### Prerequisites

- Python 3.10+
- An OpenAI API key with access to `text-embedding-3-small`
- Optional: [ChromaDB](https://docs.trychroma.com/) if you prefer a managed
  vector backend instead of the default NumPy store.

### Installation

```bash
pip install -e .[dev]
```

### Database

By default the CLI uses `pdfqanda.db` in the working directory. Override the
location with `DB_PATH` before running commands:

```bash
export DB_PATH="/tmp/pdfqanda.sqlite"
```

Run the schema initialization to create tables if they do not yet exist:

```bash
pdfqanda db init
```

### Environment

Export your OpenAI credential so the embedding client can authenticate:

```bash
export OPENAI_API_KEY="sk-..."
```

### Ingesting a PDF

```bash
pdfqanda ingest path/to/document.pdf
```

This extracts paragraphs, writes canonical rows in `kb.*`, and stores OpenAI
embeddings alongside the text chunks.

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

The smoke suite ingests fixture PDFs, verifies rows exist in `kb_markdowns`, and
asserts that `ask` returns cited snippets.

## Development Notes

- The embedding helper writes cache files under `.cache/llm/` to avoid redundant
  OpenAI requests when ingesting the same content repeatedly.
- The SQLite storage keeps embeddings as JSON while the dedicated vector index
  stores normalised vectors on disk (or in Chroma when available).

## License

MIT — see [LICENSE](LICENSE).
