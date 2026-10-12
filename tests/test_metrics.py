# tests/test_metrics.py
import math
from lifeguide.eval.metrics import recall_at_k, mrr, ndcg_at_k

def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "x"}, 5) == 0.5
    assert recall_at_k(["a", "b", "c"], {"a"}, 1) == 1.0
    assert recall_at_k([], {"a"}, 5) == 0.0

def test_mrr():
    assert mrr(["x", "y", "a"], {"a"}) == 1 / 3
    assert mrr(["a", "b"], {"a"}) == 1.0
    assert mrr(["x"], {"a"}) == 0.0

def test_ndcg_hand_computed():
    retrieved = ["x", "a", "b", "c"]
    relevant = {"a", "c"}
    dcg = 0 / math.log2(2) + 1 / math.log2(3) + 0 / math.log2(4) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert abs(ndcg_at_k(retrieved, relevant, 10) - dcg / idcg) < 1e-9
