# FINAL_STATE.md — Vision and End Architecture

## 1. Purpose

The **pdfqanda** system is an **agent-driven knowledge engine** that ingests PDFs of any complexity, converts them into structured, queryable knowledge, and enables **citation-based, explainable answers** across text, tables, figures, and notes.
It fuses robust PDF parsing, structured storage (Postgres + pgvector), multi-agent reasoning (Google A2A + PoML), and deterministic caching for reproducibility.

---

## 2. Core Principles

1. **Faithful Extraction** — capture every page element: text, tables, graphics, notes.
2. **Explainable Retrieval** — all answers carry precise citations (`【doc:§page|Lx–Ly】`).
3. **Deterministic Reasoning** — temperature = 0, all LLM calls cached and replayable.
4. **Extensible Modularity** — every extractor or agent is swappable without refactoring.
5. **Local-First** — CLI-driven, runs offline except for OpenAI API calls.

---

## 3. System Architecture Overview

### 3.1 Components

| Layer           | Key Modules                                                            | Purpose                                                       |
| --------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------- |
| **Ingestion**   | `text_extract.py`, `tables.py`, `layout.py`, `graphics.py`, `notes.py` | Convert PDF into structured artifacts                         |
| **Persistence** | Postgres + pgvector                                                    | Store documents, sections, markdowns, notes, graphics, tables |
| **Retrieval**   | `researcher.py`                                                        | Hybrid semantic + FTS + SQL search                            |
| **Reasoning**   | `expert.py`                                                            | Compose answers with citations                                |
| **Interface**   | CLI + A2A agents                                                       | Ingest / Ask / Peek / SQL                                     |
| **Cache**       | `.cache/llm`, `.cache/pdf`, `.cache/tables`                            | Deterministic reuse of extracted & LLM outputs                |

---

## 4. Data Model (Postgres)

### Schemas

* `kb` — canonical content
* `pdf_tables` — dynamically generated SQL tables for large tabular data

### Key Tables

* **documents** — metadata (`id, title, sha256, meta`)
* **sections** — hierarchy & page spans (`level, start_page, end_page`)
* **markdowns** — semantic text chunks (`content, emb VECTOR(1536), tsv`)
* **notes** — footnotes / annotations (`kind, ref_anchor, content`)
* **graphics** — images / figures (`bbox, nearby_text, path, sha256`)
* **tables_metadata** — links to `pdf_tables.*` (`columns_json, units_json`)

Indices:

* `HNSW` on embeddings (`cosine`)
* `GIN` on `tsvector`
* FKs cascade deletions per document

---

## 5. Ingestion Pipeline

### Steps

1. **Outline Detection** — build section tree via PDF outline or LLM layout agent.
2. **Page Extraction** — PyMuPDF yields text blocks + images + drawings (bboxes).
3. **Layout Normalization** — remove headers/footers, fix multi-column order.
4. **Tables Extraction** — Camelot → PDFPlumber fallback → manual merge logic.
5. **Notes Extraction** — detect bottom-page or footnote patterns; link refs.
6. **Graphics Capture** — extract image bytes → `.png`, record bbox + nearby text.
7. **Semantic Segmentation** — LLM-guided chunks (~1 k tokens + 12% overlap).
8. **Embedding + FTS** — compute `text-embedding-3-small`; build tsvector.
9. **Commit** — insert all entities in one transaction; write artifacts → cache.

### Caching Rules

* Deterministic keys = `SHA256(pdf_path + task + model + params)`
* Persistent directories: `.cache/pdf`, `.cache/llm`, `.cache/tables`, `.cache/emb`.

---

## 6. Retrieval & Question Answering

### 6.1 Researcher Agent

* Embeds query → vector search (`top k=12`)
* Optional FTS refine → rerank to 8 → final 6 snippets
* Table targeting via caption/column match + text-to-SQL
* Returns JSON evidence + `exhausted` flag

### 6.2 Expert Agent

* Receives question + evidence
* Plans sub-queries (A2A tasks) → requests Researcher
* Composes final Markdown answer
* **Citation hard-fail**: if any claim uncited → return `CITATION_CHECK_FAILED`

### 6.3 Citation Format

`【doc:{doc_id} §{section_id} p.{start_page}-{end_page} | L{start}-{end}】`

### 6.4 User Agent (optional GUI phase)

Reformats Expert output per user tone/length.

---

## 7. Agents & Protocols

| Agent              | Responsibility                         | Tooling              |
| ------------------ | -------------------------------------- | -------------------- |
| **Coordinator**    | orchestrates ingestion/retrieval flows | Python orchestrator  |
| **OutlineAgent**   | derive structure                       | PyMuPDF + LLM        |
| **LayoutAgent**    | fix page merges                        | heuristics + LLM     |
| **TableAgent**     | detect / merge tables                  | Camelot / Plumber    |
| **GraphicsAgent**  | extract bbox + caption                 | PyMuPDF              |
| **SegmenterAgent** | semantic chunking                      | LLM                  |
| **Researcher**     | search + SQL                           | pgvector + FTS + LLM |
| **Expert**         | compose answers                        | GPT-4o-mini          |
| **UserAgent**      | optional formatting/UI                 | CLI → future GUI     |

All communicate via **A2A protocol**, defined by `agent.json` contracts and **POML** templates in `prompts/`.

---

## 8. CLI Interface

```
pdfqanda ingest <pdf_path>       # full parse + DB insert
pdfqanda ask "<question>"        # run Researcher→Expert flow
pdfqanda peek <doc|page|section> # inspect extracted content
pdfqanda sql "<query>"          # direct Postgres SELECT
pdfqanda dump-tables <doc_id>    # list SQL tables created
```

Outputs default to **Markdown**; `--json` flag returns structured payloads.

---

## 9. Testing & Quality

* **Unit tests:** schema creation, chunker, notes/graphics detectors.
* **Integration:** end-to-end ingest + ask on golden PDFs (≥95 answerable@k=10).
* **CITATION enforcement:** any missing cite → fail.
* **Coverage:** ≥ 100% on core libs.
* **Latency targets:** p95 ≤ 5 s ask on <20 MB docs.

---

## 10. Security & Config

* `.env` holds API keys / DB DSN.
* Cache excludes secrets.
* Logs structured JSON under `runs/` with request IDs.
* No PII persistence.

---

## 11. Extensibility Roadmap

| Phase              | Feature                                        | Description                              |
| ------------------ | ---------------------------------------------- | ---------------------------------------- |
| **MVP (complete)** | Text + Notes + Graphics extraction; CLI TF-IDF | Local ingestion prototype                |
| **Phase 2**        | Postgres + pgvector + Hybrid Search            | Replace pickle index                     |
| **Phase 3**        | Multi-agent A2A runtime                        | Researcher↔Expert loop with PoML prompts |
| **Phase 4**        | Vision & OCR Agents                            | Scanned PDFs + captioning + math OCR     |
| **Phase 5**        | Web/UI layer + User Agent                      | Interactive QA dashboard                 |
| **Phase 6**        | Multi-doc corpus + cross-doc reasoning         | Global knowledge queries                 |

---

## 12. Success Metrics

* **Precision** ≥ 0.9 for factual QA on gold docs.
* **Recall** ≥ 0.95 within top-10 chunks.
* **Zero hallucinations** (tested via citation completeness).
* **Latency:** ≤ 10 s full pipeline on <50 MB PDF.
* **Cost:** ≤ $0.01 per doc ingestion @ OpenAI 4o-mini.

---

## 13. Directory Layout (final)

```
pdfqanda/
  src/pdfqanda/
    ingest/        # outline, layout, tables, notes, graphics, segmenter
    retrieval/     # researcher, hybrid search, text-to-sql
    agents/        # expert, researcher, helper agents
    util/          # db, cache, embeddings, citations
    cli.py
    config.py
  tests/
  .cache/
  prompts/         # POML templates
  runs/
  FINAL_STATE.md
```

---

## 14. Outcome

At completion, **pdfqanda** becomes a **self-contained agentic RAG platform** for structured PDF understanding:

* Reads any PDF → structured DB with full fidelity.
* Answers questions with citations across text, tables, and figures.
* Operates locally with deterministic caching and reproducible outputs.
* Fully extensible to vision, OCR, and multi-document reasoning.
