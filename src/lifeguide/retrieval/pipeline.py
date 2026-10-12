# src/lifeguide/retrieval/pipeline.py
from dataclasses import dataclass, field
from lifeguide.parser.models import Entry
from lifeguide.retrieval.rrf import rrf

@dataclass
class SearchConfig:
    use_bm25: bool = True
    use_sparse: bool = True
    use_rerank: bool = True          # Task 9 的 reranker 注入后生效
    evidence_min: str | None = None  # "A" / "B" / "C"（含以上）
    top_k: int = 5
    recall_k: int = 20

@dataclass
class Hit:
    chunk_id: str
    entry_id: str
    score: float
    text: str
    evidence_level: str | None = None
    benefit: str | None = None       # cost_tags.收益（大/中/小）
    cost_tag: str | None = None      # cost_tags.钱（0/少/中/大）

@dataclass
class SearchResult:
    hits: list[Hit] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)
    debug: dict = field(default_factory=dict)

class SearchPipeline:
    _EV_ORDER = {"A": 3, "B": 2, "C": 1, None: 0}

    def __init__(self, store, embedder, bm25_index, reranker=None):
        self.store, self.embedder, self.bm25 = store, embedder, bm25_index
        self.reranker = reranker

    def search(self, query: str, config: SearchConfig = SearchConfig()) -> SearchResult:
        dense_q, sparse_q = self.embedder.embed([query])[0]
        rankings: list[list[str]] = [ [c for c, _, _ in self.store.search_dense(dense_q, config.recall_k)] ]
        if config.use_sparse:
            rankings.append([c for c, _, _ in self.store.search_sparse(sparse_q, config.recall_k)])
        if config.use_bm25:
            rankings.append([c for c, _ in self.bm25.search(query, config.recall_k)])
        fused = rrf(rankings)
        candidates = fused[: max(config.recall_k, config.top_k * 4)]
        hits = self._to_hits(candidates)
        if config.evidence_min:
            floor = self._EV_ORDER[config.evidence_min]
            hits = [h for h in hits if self._EV_ORDER.get(h.evidence_level, 0) >= floor]
        if config.use_rerank and self.reranker:
            hits = self.reranker.rerank(query, hits)[: config.top_k]
        else:
            hits = hits[: config.top_k]
        return SearchResult(hits=hits, entries=[e for e in (self.store.get_entry(h.entry_id) for h in hits) if e],
                            debug={"rankings": rankings, "fused": fused[:20]})

    def _to_hits(self, fused: list[tuple[str, float]]) -> list[Hit]:
        out = []
        for chunk_id, score in fused:
            entry_id = chunk_id.split("#")[0]
            meta = self.store.chunk_meta(chunk_id) or {}
            text = self.store.chunk_text(chunk_id) or ""
            out.append(Hit(chunk_id, entry_id, score, text,
                          meta.get("evidence_level"), meta.get("benefit"), meta.get("cost_tag_money")))
        return out
