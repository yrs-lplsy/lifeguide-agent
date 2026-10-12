# tests/test_rrf.py
from lifeguide.retrieval.rrf import rrf

def test_rrf_basic_math():
    # 路1: [a, b]; 路2: [b, a]; k=60
    # a: 1/61 + 1/62; b: 1/62 + 1/61 → 并列，顺序稳定按 id
    out = rrf([["a", "b"], ["b", "a"]], k=60)
    scores = dict(out)
    assert abs(scores["a"] - (1/61 + 1/62)) < 1e-9
    assert abs(scores["b"] - (1/61 + 1/62)) < 1e-9

def test_rrf_missing_in_one_list():
    out = rrf([["a", "b", "c"], ["c"]], k=60)
    scores = dict(out)
    assert scores["c"] > scores["a"] > scores["b"]

def test_rrf_empty():
    assert rrf([[], []]) == []
