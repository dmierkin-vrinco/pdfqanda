-- Canonical SQLite schema for pdfqanda knowledge base.

CREATE TABLE IF NOT EXISTS kb_documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    meta TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kb_sections (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
    parent_id TEXT REFERENCES kb_sections(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    level INTEGER NOT NULL,
    start_page INTEGER NOT NULL,
    end_page INTEGER NOT NULL,
    path TEXT,
    meta TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS kb_markdowns (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
    section_id TEXT REFERENCES kb_sections(id) ON DELETE SET NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    char_start INTEGER,
    char_end INTEGER,
    start_page INTEGER,
    end_page INTEGER,
    emb TEXT NOT NULL,
    tsv TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sqlite_markdowns_doc
    ON kb_markdowns(document_id);

CREATE TABLE IF NOT EXISTS kb_tables (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
    section_id TEXT REFERENCES kb_sections(id) ON DELETE SET NULL,
    caption TEXT,
    data JSON NOT NULL,
    meta JSON DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS kb_graphics (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
    section_id TEXT REFERENCES kb_sections(id) ON DELETE SET NULL,
    caption TEXT,
    image_path TEXT NOT NULL,
    meta JSON DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS kb_notes (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
    section_id TEXT REFERENCES kb_sections(id) ON DELETE SET NULL,
    content TEXT NOT NULL,
    referenced_page INTEGER,
    meta JSON DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    id TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
