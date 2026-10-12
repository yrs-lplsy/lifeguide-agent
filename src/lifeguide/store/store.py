# src/lifeguide/store/store.py
import json, os
from pathlib import Path
import psycopg
from pgvector.psycopg import register_vector
from pgvector.sparsevec import SparseVector
from lifeguide.parser.models import Entry


class ChildChunk:
    def __init__(self, chunk_id, entry_id, field_name, text, dense, sparse, metadata):
        self.chunk_id, self.entry_id, self.field_name = chunk_id, entry_id, field_name
        self.text, self.dense, self.sparse, self.metadata = text, dense, sparse, metadata


class EntryStore:
    def __init__(self, database_url: str, embed_dim: int | None = None,
                 sparse_dim: int | None = None):
        self.url = database_url
        self.dim = embed_dim or int(os.getenv("EMBED_DIM", "1024"))
        # pgvector 要求相互比较的 sparsevec 声明维度一致（"different sparsevec
        # dimensions"），因此插入与查询必须用同一个固定维度；bge-m3 词表 250002。
        self.sparse_dim = sparse_dim or int(os.getenv("SPARSE_DIM", "250002"))
        self.conn = psycopg.connect(database_url, autocommit=True)
        register_vector(self.conn)

    def init_schema(self):
        schema = Path(__file__).parent.joinpath("schema.sql").read_text(encoding="utf-8")
        schema = schema.replace("vector(1024)", f"vector({self.dim})")
        schema = schema.replace("sparsevec(250002)", f"sparsevec({self.sparse_dim})")
        with self.conn.cursor() as cur:
            cur.execute(schema)

    def reset(self):
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE chunks, entries CASCADE")

    def _sv(self, d: dict[int, float]) -> SparseVector | None:
        return SparseVector(d, self.sparse_dim) if d else None

    def upsert_entries(self, entries: list[Entry]):
        with self.conn.cursor() as cur:
            for e in entries:
                cur.execute(
                    "INSERT INTO entries (entry_id, data) VALUES (%s, %s::jsonb) "
                    "ON CONFLICT (entry_id) DO UPDATE SET data = EXCLUDED.data",
                    (e.entry_id, e.model_dump_json()))

    def upsert_chunks(self, chunks: list[ChildChunk]):
        with self.conn.cursor() as cur:
            for c in chunks:
                sv = self._sv(c.sparse)
                cur.execute(
                    "INSERT INTO chunks (chunk_id, entry_id, field_name, text, dense, sparse, metadata) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT (chunk_id) DO UPDATE SET "
                    "dense=EXCLUDED.dense, sparse=EXCLUDED.sparse, text=EXCLUDED.text, metadata=EXCLUDED.metadata",
                    (c.chunk_id, c.entry_id, c.field_name, c.text,
                     c.dense if c.dense else None, sv,
                     json.dumps(c.metadata, ensure_ascii=False)))

    def _hits(self, cur) -> list[tuple[str, float, str]]:
        return [(r[0], float(r[1]), r[2]) for r in cur.fetchall()]

    def search_dense(self, qvec: list[float], k: int = 20):
        with self.conn.cursor() as cur:
            cur.execute("SELECT chunk_id, 1 - (dense <=> %s::vector) AS score, entry_id "
                        "FROM chunks WHERE dense IS NOT NULL ORDER BY dense <=> %s::vector LIMIT %s",
                        (qvec, qvec, k))
            return self._hits(cur)

    def search_sparse(self, qsparse: dict[int, float], k: int = 20):
        if not qsparse:
            return []
        sv = self._sv(qsparse)
        # bge-m3 词法权重按内积设计；pgvector <#> 为负内积，升序排 = 内积降序 =
        # 最优在前。score 取负回内积原值，保持"越大越好"口径（L2 会偏袒小范数
        # 短 chunk，实测与内积 top-20 重叠仅 0-1/20）。
        with self.conn.cursor() as cur:
            cur.execute("SELECT chunk_id, -(sparse <#> %s::sparsevec) AS score, entry_id "
                        "FROM chunks WHERE sparse IS NOT NULL ORDER BY sparse <#> %s::sparsevec LIMIT %s",
                        (sv, sv, k))
            return self._hits(cur)

    def chunk_meta(self, chunk_id: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT metadata FROM chunks WHERE chunk_id = %s", (chunk_id,))
            row = cur.fetchone()
            return row[0] if row else None

    def chunk_text(self, chunk_id: str) -> str | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT text FROM chunks WHERE chunk_id = %s", (chunk_id,))
            row = cur.fetchone()
            return row[0] if row else None

    def get_entry(self, entry_id: str) -> Entry | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT data FROM entries WHERE entry_id = %s", (entry_id,))
            row = cur.fetchone()
            # psycopg3 把 jsonb 直接加载为 dict，故用 model_validate
            return Entry.model_validate(row[0]) if row else None

    def chunks_by_entry(self, entry_id: str) -> list[ChildChunk]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT chunk_id, entry_id, field_name, text, dense, sparse, metadata "
                        "FROM chunks WHERE entry_id = %s ORDER BY chunk_id", (entry_id,))
            out = []
            for cid, eid, fname, text, dense, sparse, meta in cur.fetchall():
                out.append(ChildChunk(cid, eid, fname, text,
                                      dense.to_list() if dense is not None else [],
                                      dict(zip(sparse.indices(), sparse.values()))
                                      if sparse is not None else {},
                                      meta or {}))
            return out

    def all_chunks(self) -> list[tuple[str, str]]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id")
            return cur.fetchall()
