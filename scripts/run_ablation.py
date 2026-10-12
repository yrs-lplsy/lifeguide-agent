# scripts/run_ablation.py — 四组消融实验入口（Task 14）
# 用法: HF_HUB_OFFLINE=1 python scripts/run_ablation.py
# 产出: docs/m1-ablation-report.md（4×3 指标表 + 各层贡献 + zero_hit 拒答率）
#       docs/m1-ablation-details.csv（逐查询明细：4×176 行 + zero_hit 20 行）
import csv
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lifeguide.config import settings_from_env            # noqa: E402
from lifeguide.store.store import EntryStore              # noqa: E402
from lifeguide.retrieval.embedder import Embedder         # noqa: E402
from lifeguide.retrieval.bm25 import BM25Index            # noqa: E402
from lifeguide.retrieval.pipeline import SearchPipeline   # noqa: E402
from lifeguide.retrieval.rerank import Reranker           # noqa: E402
from lifeguide.eval.ablation import run_ablation, zero_hit_rejection  # noqa: E402
from lifeguide.eval.golden_set import load_golden         # noqa: E402

GOLDEN_PATH = ROOT / "data" / "golden_set.jsonl"
REPORT_PATH = ROOT / "docs" / "m1-ablation-report.md"
DETAILS_PATH = ROOT / "docs" / "m1-ablation-details.csv"

FLAGS = {  # 报告表格里各组的开关标记
    "A_dense":       ("✗", "✗", "✗"),
    "B_dense_bm25":  ("✓", "✗", "✗"),
    "C_3way_rrf":    ("✓", "✓", "✗"),
    "D_3way_rerank": ("✓", "✓", "✓"),
}

def write_details_csv(path: Path, per_query: list[dict], zero_rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "qid", "qtype", "recall@5", "mrr", "ndcg@10",
                    "top1_score", "rejected(<0.3)"])
        for r in per_query:
            w.writerow([r["config"], r["qid"], r["qtype"],
                        f"{r['recall@5']:.4f}", f"{r['mrr']:.4f}", f"{r['ndcg@10']:.4f}", "", ""])
        for r in zero_rows:
            w.writerow(["D_3way_rerank(top_k=1)", r["qid"], "zero_hit", "", "", "",
                        f"{r['top1_score']:.4f}" if r["top1_score"] is not None else "",
                        "yes" if r["rejected"] else "no"])

def write_report(path: Path, results: dict, zh_ok: int, zh_n: int, zh_rate: float,
                 counts: Counter, wall: float):
    L = []
    L.append("# M1 检索消融实验报告（简历量化数字来源）")
    L.append("")
    L.append(f"- 日期：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    L.append(f"- Golden set：`data/golden_set.jsonl` 共 {sum(counts.values())} 条"
             f"（colloquial {counts['colloquial']} / multi_hop {counts['multi_hop']}"
             f" / zero_hit {counts['zero_hit']}；未人工复审，按原样评测）")
    L.append("- 口径：四组统一 `top_k=10, recall_k=20`，指标为非 zero_hit 的 "
             f"{results['A_dense']['n']} 条上的平均值；zero_hit 不计入四组指标，单独测拒答")
    L.append("- 指标：recall@5（top5 命中相关条目比例）/ MRR / nDCG@10（chunk 级命中折算到 entry_id）")
    L.append("")
    L.append("## 四组配置结果")
    L.append("")
    L.append("| 配置 | bm25 | sparse | rerank | n | recall@5 | MRR | nDCG@10 |")
    L.append("|------|------|--------|--------|---|----------|-----|---------|")
    for name, (b, s, r) in FLAGS.items():
        m = results[name]
        L.append(f"| {name} | {b} | {s} | {r} | {m['n']} | "
                 f"{m['recall@5']:.4f} | {m['mrr']:.4f} | {m['ndcg@10']:.4f} |")
    L.append("")
    L.append("## 各层贡献（相邻差值，简历数字：B/A、C/B、D/C）")
    L.append("")
    L.append("| 增量 | recall@5 Δ | MRR Δ | nDCG@10 Δ |")
    L.append("|------|-----------|-------|-----------|")
    for label, hi, lo in [("+BM25（B−A）", "B_dense_bm25", "A_dense"),
                          ("+sparse RRF（C−B）", "C_3way_rrf", "B_dense_bm25"),
                          ("+rerank（D−C）", "D_3way_rerank", "C_3way_rrf")]:
        a, c = results[lo], results[hi]
        L.append(f"| {label} | {c['recall@5'] - a['recall@5']:+.4f} | "
                 f"{c['mrr'] - a['mrr']:+.4f} | {c['ndcg@10'] - a['ndcg@10']:+.4f} |")
    L.append("")
    L.append("## zero_hit 拒答率")
    L.append("")
    L.append(f"D 配置 `top_k=1`，top-1 rerank 分数 < 0.3（或无命中）判为正确拒答："
             f"**{zh_ok}/{zh_n} = {zh_rate:.1%}**")
    L.append("")
    L.append("## 运行信息")
    L.append("")
    L.append(f"- 总耗时 {wall:.0f}s（含模型加载、BM25 建索引与全部查询嵌入）")
    L.append(f"- 逐查询明细：`docs/m1-ablation-details.csv`（{len(FLAGS)}×{results['A_dense']['n']} 行"
             f" + zero_hit {zh_n} 行）")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")

def main():
    t0 = time.time()
    golden = load_golden(GOLDEN_PATH)
    counts = Counter(item["qtype"] for item in golden)
    print(f"golden: {len(golden)} items {dict(counts)}", flush=True)

    # 组装（初始化段复制自 scripts/search_cli.py）
    s = settings_from_env()
    store = EntryStore(s.database_url)
    emb = Embedder()
    bm25 = BM25Index(); bm25.build(store.all_chunks())   # 建一次，四组复用
    pipe = SearchPipeline(store, emb, bm25, reranker=Reranker())

    # 预热嵌入缓存：批量编码全部查询，后续四组配置全部命中缓存
    emb.embed([item["query"] for item in golden])

    per_query: list[dict] = []
    zero_rows: list[dict] = []
    results = run_ablation(pipe, golden, per_query=per_query)
    zh_rate = zero_hit_rejection(pipe, golden, per_query=zero_rows)
    wall = time.time() - t0

    zh_ok = sum(1 for r in zero_rows if r["rejected"])
    write_details_csv(DETAILS_PATH, per_query, zero_rows)
    write_report(REPORT_PATH, results, zh_ok, len(zero_rows), zh_rate, counts, wall)
    print(f"report -> {REPORT_PATH}", flush=True)
    print(f"details -> {DETAILS_PATH} ({len(per_query) + len(zero_rows)} rows)", flush=True)
    print(f"wall {wall:.0f}s", flush=True)

if __name__ == "__main__":
    main()
