-- Canonical Postgres schema for pdfqanda knowledge base
CREATE SCHEMA IF NOT EXISTS kb;
CREATE SCHEMA IF NOT EXISTS pdf_tables;

CREATE TABLE IF NOT EXISTS kb.documents (
    id UUID PRIMARY KEY,
    title TEXT NOT NULL,
    sha256 CHAR(64) NOT NULL UNIQUE,
    meta JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

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

CREATE TABLE IF NOT EXISTS kb.markdowns (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES kb.documents(id) ON DELETE CASCADE,
    section_id UUID REFERENCES kb.sections(id) ON DELETE SET NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    start_page INTEGER,
    end_page INTEGER,
    emb VECTOR(3072),
    tsv tsvector
);

CREATE INDEX IF NOT EXISTS idx_markdowns_emb
    ON kb.markdowns USING hnsw (emb vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_markdowns_tsv
    ON kb.markdowns USING gin (tsv);

CREATE OR REPLACE FUNCTION kb.markdowns_tsv_trigger()
RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('english', coalesce(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tsv_update
BEFORE INSERT OR UPDATE ON kb.markdowns
FOR EACH ROW EXECUTE FUNCTION kb.markdowns_tsv_trigger();
