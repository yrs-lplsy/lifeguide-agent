# src/lifeguide/eval/ablation.py — 四组消融配置 + 评测运行（Task 14，简历量化数字来源）
import dataclasses
from lifeguide.retrieval.pipeline import SearchConfig
from lifeguide.eval.metrics import recall_at_k, mrr, ndcg_at_k

CONFIGS = {
    "A_dense":        SearchConfig(use_bm25=False, use_sparse=False, use_rerank=False),
    "B_dense_bm25":   SearchConfig(use_bm25=True,  use_sparse=False, use_rerank=False),
    "C_3way_rrf":     SearchConfig(use_bm25=True,  use_sparse=True,  use_rerank=False),
    "D_3way_rerank":  SearchConfig(use_bm25=True,  use_sparse=True,  use_rerank=True),
}

def run_ablation(pipeline, golden: list[dict],
                 per_query: list | None = None) -> dict[str, dict[str, float]]:
    """对非 zero_hit 查询跑四组配置，返回 {config: {recall@5, mrr, ndcg@10, n}}。

    统一用 dataclasses.replace(cfg, top_k=10, recall_k=20) 保证四组同口径；
    per_query 若提供，则追加逐查询明细行（dict: config/qid/qtype/recall@5/mrr/ndcg@10），
    供报告的明细 csv 导出。zero_hit 查询不计入四组指标。
    """
    results = {}
    for name, cfg in CONFIGS.items():
        eval_cfg = dataclasses.replace(cfg, top_k=10, recall_k=20)
        rec5, mrr_v, ndcg, n = 0.0, 0.0, 0.0, 0
        for i, item in enumerate(golden, 1):
            if item["qtype"] == "zero_hit":
                continue
            # chunk 级命中保序去重到 entry 级（同一 entry 多 chunk 会重复，挤占 top-k
            # 窗口且使 nDCG@10 双计超 1）；metrics.py 契约假定输入已去重，勿改 metrics。
            retrieved = list(dict.fromkeys(h.entry_id for h in pipeline.search(item["query"], eval_cfg).hits))
            relevant = set(item["relevant"])
            r5 = recall_at_k(retrieved, relevant, 5)
            mv = mrr(retrieved, relevant)
            n10 = ndcg_at_k(retrieved, relevant, 10)
            rec5 += r5
            mrr_v += mv
            ndcg += n10
            n += 1
            if per_query is not None:
                per_query.append({"config": name, "qid": item["qid"], "qtype": item["qtype"],
                                  "recall@5": r5, "mrr": mv, "ndcg@10": n10})
            if i % 25 == 0:
                print(f"[ablation:{name}] {i}/{len(golden)}", flush=True)
        results[name] = {"recall@5": rec5 / n, "mrr": mrr_v / n, "ndcg@10": ndcg / n, "n": n}
        print(f"[ablation:{name}] done {results[name]}", flush=True)
    return results

def zero_hit_rejection(pipeline, golden, score_floor: float = 0.3,
                       per_query: list | None = None) -> float:
    """zero_hit 拒答率：D 配置（全开默认）top_k=1 下，top-1 rerank 分 < score_floor
    （或无命中）视为正确拒答。per_query 若提供则追加 (qid/top1_score/rejected) 明细。"""
    ok, n = 0, 0
    for item in golden:
        if item["qtype"] != "zero_hit":
            continue
        hits = pipeline.search(item["query"], SearchConfig(top_k=1)).hits
        rejected = not hits or hits[0].score < score_floor
        n += 1
        ok += 1 if rejected else 0
        if per_query is not None:
            per_query.append({"qid": item["qid"],
                              "top1_score": (hits[0].score if hits else None),
                              "rejected": rejected})
    return ok / n if n else 0.0
