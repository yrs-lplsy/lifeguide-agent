# src/lifeguide/retrieval/rerank.py
class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from FlagEmbedding import FlagReranker
        self.model = FlagReranker(model_name, use_fp16=False)

    def rerank(self, query: str, hits, top_k: int | None = None):
        if not hits:
            return []
        pairs = [[query, h.text] for h in hits]
        scores = self.model.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        for h, s in zip(hits, scores):
            h.score = float(s)
        ordered = sorted(hits, key=lambda h: -h.score)
        return ordered[:top_k] if top_k else ordered
