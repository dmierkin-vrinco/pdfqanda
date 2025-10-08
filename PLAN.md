# Implementation Plan for FINAL_STATE

## Overview
The current codebase provides a working CLI-oriented ingestion and retrieval
pipeline but keeps most functionality in a handful of monolithic modules. To
reach the multi-agent, modular architecture described in `FINAL_STATE.md`, we
need to progressively refactor the package layout, codify separation between
layers, and prepare surfaces for upcoming agents and caches.

## Phase 1 — Package the Core Layers (this PR)
- [x] Create dedicated packages for `ingest`, `retrieval`, and `agents` that
  mirror the architecture table in `FINAL_STATE.md`.
- [x] Move existing ingestion and retrieval logic into their respective
  packages while preserving public APIs and backwards compatibility.
- [x] Centralize exports through explicit `__all__` declarations so CLI/tests
  can transition gradually without breaking imports.

## Phase 2 — Persistence & Cache Enhancements (this PR)
- [x] Split vector index logic into an interface that can back Chroma/NumPy
  implementations and update the database layer to manage index lifecycle.
- [x] Expand `.cache/` layout to include embeddings/tables namespaces with
  deterministic hashing per FINAL_STATE guidelines.
- [x] Add schema migration utilities and refresh `schema.sql` to cover tables,
  graphics, and notes entities.

## Phase 3 — Agentic Retrieval Loop (future)
- [ ] Flesh out the Researcher and Expert agents inside `pdfqanda.agents`
  using POML prompts, adding coordination scaffolding and structured outputs.
- [ ] Integrate citation guards that fail fast when an answer lacks coverage,
  exposing explicit error codes to the CLI.
- [ ] Introduce planning hooks for UserAgent/GUI layers once the CLI is fully
  deterministic.

Each phase builds toward the final architecture while keeping the system
functional after every increment. This PR completes Phase 1 by establishing the
package structure that later phases will rely on.
