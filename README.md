# pdfqanda

pdfqanda is a lightweight retrieval-based assistant that lets you ask questions about the
contents of your local PDF documents. The project provides a small set of utilities to extract
text, break it into overlapping chunks, index those chunks with a TF-IDF vector store, and query
for the most relevant excerpts. The PDF extractor focuses on simple text-based PDFs by scanning
their content streams and may not work for heavily compressed or scanned documents.

## Features

- Extract text from one or more PDF files using a lightweight parser for text-based PDFs.
- Split long documents into overlapping text chunks for improved retrieval.
- Index text chunks with a TF-IDF vectorizer and cosine similarity search.
- Command line interface for building indexes and asking questions.
- Optional JSON export of retrieved answers for downstream automation.

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

### Asking a Question

```bash
pdfqanda ask --index my-index.pkl "What is the main conclusion?"
```

Answers are displayed in a table sorted by cosine similarity score. You can also export the
results to JSON:

```bash
pdfqanda ask --index my-index.pkl "Summarize section 3" --json-output answers.json
```

## Development

### Running Tests

```bash
pytest
```

### Linting

```bash
ruff check .
```

## License

This project is licensed under the terms of the MIT License. See [LICENSE](LICENSE) for details.
