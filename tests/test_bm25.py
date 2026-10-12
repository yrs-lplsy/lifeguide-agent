# tests/test_bm25.py
from lifeguide.retrieval.bm25 import BM25Index, tokenize

def test_tokenize_drops_punct():
    toks = tokenize("被裁了，先算清 N！")
    assert "被裁" in "".join(toks) or any("裁" in t for t in toks)
    assert "，" not in toks and "！" not in toks

def test_bm25_ranks_exact_term_first():
    corpus = [("a", "经济补偿按 N 计算，每满一年一个月工资"),
              ("b", "烟雾报警器能降低火灾死亡概率"),
              ("c", "高血压要常年吃药控制")]
    idx = BM25Index(); idx.build(corpus)
    hits = idx.search("被裁 经济补偿 N", k=3)
    assert hits[0][0] == "a"
    hits2 = idx.search("烟雾报警器", k=3)
    assert hits2[0][0] == "b"
