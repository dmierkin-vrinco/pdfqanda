-- Canonical database schema for pdfqanda knowledge base.
-- Schema: kb

-- Optional extensions (Postgres only). These commands are wrapped so they can be
-- executed on other engines without failing.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname = 'pg_catalog') THEN
        EXECUTE 'CREATE EXTENSION IF NOT EXISTS vector';
        EXECUTE 'ALTER DATABASE ' || quote_ident(current_database()) || ' SET vector.hnsw.max_dimensions = 4096';
        EXECUTE 'CREATE EXTENSION IF NOT EXISTS pg_trgm';
    END IF;
EXCEPTION
    WHEN undefined_table THEN NULL;
    WHEN undefined_function THEN NULL;
    WHEN insufficient_privilege THEN NULL;
END
$$;

SET vector.hnsw.max_dimensions = 4096;

CREATE SCHEMA IF NOT EXISTS kb;
CREATE SCHEMA IF NOT EXISTS pdf_tables;

-- Documents table stores one row per ingested PDF document.
CREATE TABLE IF NOT EXISTS kb.documents (
    id UUID PRIMARY KEY,
    title TEXT NOT NULL,
    sha256 CHAR(64) NOT NULL UNIQUE,
    meta JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Sections capture the logical outline hierarchy detected from the document.
CREATE TABLE IF NOT EXISTS kb.sections (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES kb.documents(id) ON DELETE CASCADE,
    parent_id UUID REFERENCES kb.sections(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    level INTEGER NOT NULL,
    start_page INTEGER NOT NULL,
    end_page INTEGER NOT NULL,
    path TEXT,
    meta JSONB DEFAULT '{}'::jsonb
);

-- Markdown segments hold semantically segmented text content.
CREATE TABLE IF NOT EXISTS kb.markdowns (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES kb.documents(id) ON DELETE CASCADE,
    section_id UUID REFERENCES kb.sections(id) ON DELETE SET NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    char_start INTEGER,
    char_end INTEGER,
    start_page INTEGER,
    end_page INTEGER,
    emb VECTOR(3072) NOT NULL,
    tsv tsvector NOT NULL
);

-- Notes such as footnotes or annotations.
CREATE TABLE IF NOT EXISTS kb.notes (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES kb.documents(id) ON DELETE CASCADE,
    section_id UUID REFERENCES kb.sections(id) ON DELETE SET NULL,
    kind TEXT NOT NULL,
    ref_anchor TEXT,
    content TEXT NOT NULL,
    page INTEGER,
    bbox JSONB,
    meta JSONB DEFAULT '{}'::jsonb
);

-- Graphics metadata referencing stored image artifacts.
CREATE TABLE IF NOT EXISTS kb.graphics (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES kb.documents(id) ON DELETE CASCADE,
    section_id UUID REFERENCES kb.sections(id) ON DELETE SET NULL,
    caption TEXT,
    nearby_text TEXT,
    path TEXT NOT NULL,
    sha256 CHAR(64) NOT NULL,
    page INTEGER,
    bbox JSONB,
    meta JSONB DEFAULT '{}'::jsonb
);

-- Table metadata links extracted tables to dynamically generated SQL tables.
CREATE TABLE IF NOT EXISTS kb.tables_metadata (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES kb.documents(id) ON DELETE CASCADE,
    section_id UUID REFERENCES kb.sections(id) ON DELETE SET NULL,
    table_name TEXT NOT NULL,
    caption TEXT,
    columns_json JSONB,
    units_json JSONB,
    meta JSONB DEFAULT '{}'::jsonb
);

-- Vector and FTS indexes.
DO $$
BEGIN
    BEGIN
        EXECUTE 'CREATE INDEX IF NOT EXISTS idx_markdowns_emb ON kb.markdowns USING hnsw (emb vector_cosine_ops)';
    EXCEPTION
        WHEN others THEN
            BEGIN
                EXECUTE 'CREATE INDEX IF NOT EXISTS idx_markdowns_emb ON kb.markdowns USING ivfflat (emb vector_cosine_ops)';
            EXCEPTION
                WHEN others THEN
                    NULL;
            END;
    END;
END
$$;

CREATE INDEX IF NOT EXISTS idx_markdowns_tsv ON kb.markdowns USING gin (tsv);

-- Convenience view for retrieving sections with markdown counts.
CREATE OR REPLACE VIEW kb.section_stats AS
SELECT s.*, COUNT(m.id) AS markdown_count
FROM kb.sections s
LEFT JOIN kb.markdowns m ON m.section_id = s.id
GROUP BY s.id;
