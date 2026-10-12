# scripts/search_cli.py — 端到端检索 CLI（人工体验入口）
# 用法: python scripts/search_cli.py "被裁了怎么算补偿" [--ev A] [--no-bm25] [--no-sparse] [--no-rerank] [--topk N]
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifeguide.config import settings_from_env
from lifeguide.store.store import EntryStore
from lifeguide.retrieval.embedder import Embedder
from lifeguide.retrieval.bm25 import BM25Index
from lifeguide.retrieval.pipeline import SearchPipeline, SearchConfig
from lifeguide.retrieval.rerank import Reranker
from lifeguide.retrieval.filters import sort_by_value
from rich.console import Console
from rich.table import Table

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--ev", choices=["A", "B", "C"], default=None)
    ap.add_argument("--no-bm25", action="store_true")
    ap.add_argument("--no-sparse", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    s = settings_from_env()
    store = EntryStore(s.database_url)
    emb = Embedder()
    bm25 = BM25Index(); bm25.build(store.all_chunks())
    pipe = SearchPipeline(store, emb, bm25, reranker=Reranker())
    cfg = SearchConfig(use_bm25=not args.no_bm25, use_sparse=not args.no_sparse,
                       use_rerank=not args.no_rerank, evidence_min=args.ev, top_k=args.topk)
    result = pipe.search(args.query, cfg)
    title_of = {e.entry_id: e.title for e in result.entries}
    table = Table(title=f"查询: {args.query}")
    for col in ("#", "条目", "节-条", "证据", "分数", "命中文本"):
        table.add_column(col, overflow="fold")
    for i, h in enumerate(sort_by_value(result.hits), 1):
        table.add_row(str(i), title_of.get(h.entry_id, "?"), h.entry_id,
                      h.evidence_level or "-", f"{h.score:.4f}", h.text[:80])
    Console().print(table)

if __name__ == "__main__":
    main()
