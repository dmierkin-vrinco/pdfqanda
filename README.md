# pdfqanda

pdfqanda is a lightweight retrieval-based assistant that lets you ask questions about the
contents of your local PDF documents. The project provides utilities to extract text and page
metadata, break text into overlapping chunks, index those chunks with a TF-IDF vector store, and
query for the most relevant excerpts. Text extraction now prefers PyMuPDF for fast, layout-aware
parsing while retaining a content-stream fallback for environments where PyMuPDF is unavailable.

During ingestion pdfqanda also produces page-level “artifacts” that capture detected text blocks,
footnotes, and graphics. These artifacts live under `artifacts/{doc_hash}.json` with any rendered
graphics stored alongside them in `artifacts/{doc_hash}/graphics/`. Rasterized page images and block
caches are stored under `.cache/pdf/{doc_hash}/` to avoid redundant work across runs.

## Features

- Extract text from one or more PDF files using a PyMuPDF-powered fast path with a content-stream
  fallback.
- Split long documents into overlapping text chunks for improved retrieval.
- Index text chunks with a TF-IDF vectorizer and cosine similarity search.
- Command line interface for building indexes, asking questions, and inspecting detected page
  metadata.
- Optional JSON export of retrieved answers for downstream automation.
- Per-document artifact generation with normalized bounding boxes for notes and graphics.

## Getting Started

### Prerequisites

- Python 3.10 or newer.

### Installation

```bash
pip install -e .[dev]
```

### Building an Index

```bash
pdfqanda build path/to/document.pdf path/to/another.pdf --output my-index.pkl
```

In addition to writing the TF-IDF index the build command emits per-document artifacts in the
`artifacts/` directory and caches rasterized pages in `.cache/pdf/` for faster re-processing.

### Asking a Question

```bash
pdfqanda ask --index my-index.pkl "What is the main conclusion?"
```

Answers are displayed in a table sorted by cosine similarity score. You can also export the
results to JSON:

```bash
pdfqanda ask --index my-index.pkl "Summarize section 3" --json-output answers.json
```

### Inspecting Notes and Graphics

Use `pdfqanda peek` to quickly review extracted footnotes and graphics metadata without building an
index:

```bash
pdfqanda peek path/to/document.pdf --page 2
```

This command prints normalized bounding boxes, detected note markers, nearby text for graphics, and
paths to the rendered image snippets saved under `artifacts/{doc_hash}/graphics/`.

## Development

### Running Tests

```bash
pytest
```

> **Note**
> The historical `tests/test_fedex_rates.py` checks currently fail because the baseline TF-IDF
> engine is unchanged; this is expected for the project in its present state.

### Linting

```bash
ruff check .
```

## License

This project is licensed under the terms of the MIT License. See [LICENSE](LICENSE) for details.
