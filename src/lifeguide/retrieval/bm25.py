# src/lifeguide/retrieval/bm25.py
import jieba
from rank_bm25 import BM25Okapi

STOP = set("，。！？；：「」『』（）《》、,.!?;:()[]{} \n\t·…—-的了吗呢吧啊和与及或在是对我有你他她它这那也要不没什么")

def tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip() and t not in STOP]

class BM25Index:
    def __init__(self):
        self._ids: list[str] = []
        self._bm25 = None

    def build(self, corpus: list[tuple[str, str]]):
        self._ids = [cid for cid, _ in corpus]
        tokenized = [tokenize(text) for _, text in corpus]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._ids[i], float(scores[i])) for i in order if scores[i] > 0]
