-- src/lifeguide/store/schema.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS entries (
    entry_id  TEXT PRIMARY KEY,
    data      JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS entries_data_gin ON entries USING GIN (data);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT PRIMARY KEY,
    entry_id   TEXT NOT NULL REFERENCES entries(entry_id) ON DELETE CASCADE,
    field_name TEXT NOT NULL,
    text       TEXT NOT NULL,
    dense      vector(1024),
    sparse     sparsevec(250002),
    metadata   JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS chunks_dense_hnsw ON chunks USING hnsw (dense vector_cosine_ops);
-- pgvector 0.8：ivfflat 不支持 sparsevec opclass，稀疏索引用 hnsw + sparsevec_l2_ops
CREATE INDEX IF NOT EXISTS chunks_sparse_hnsw ON chunks USING hnsw (sparse sparsevec_l2_ops);
