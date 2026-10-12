# tests/test_rerank_filters.py
from lifeguide.retrieval.pipeline import Hit
from lifeguide.retrieval.rerank import Reranker
from lifeguide.retrieval.filters import sort_by_value, filter_evidence

def _hit(cid, score=0.0, ev=None, ben=None, cost=None):
    return Hit(chunk_id=cid, entry_id=cid.split("#")[0], score=score, text="t",
               evidence_level=ev, benefit=ben, cost_tag=cost)

def test_reranker_reorders_by_model_scores(monkeypatch):
    r = Reranker.__new__(Reranker)  # 免模型加载
    class FakeM:
        def compute_score(self, pairs, **kw):
            return [0.1, 0.9, 0.5]  # 第二个最相关
    r.model = FakeM()
    hits = [_hit("a"), _hit("b"), _hit("c")]
    out = r.rerank("q", hits)
    assert [h.chunk_id for h in out] == ["b", "c", "a"]
    assert out[0].score == 0.9

def test_sort_by_value_orders_evidence_benefit_cost():
    hits = [
        _hit("x1", ev="B", ben="大", cost="少"),
        _hit("x2", ev="A", ben="中", cost="中"),
        _hit("x3", ev="A", ben="大", cost="少"),
        _hit("x4", ev="C", ben="大", cost="0"),
    ]
    assert [h.chunk_id for h in sort_by_value(hits)] == ["x3", "x2", "x1", "x4"]

def test_sort_by_value_cost_tie_duo_sorts_after_shao():
    # 对账 entries.jsonl 实际值域: cost money ∈ {0, 少, 多}
    hits = [
        _hit("y1", ev="A", ben="中", cost="多"),
        _hit("y2", ev="A", ben="中", cost="少"),
    ]
    assert [h.chunk_id for h in sort_by_value(hits)] == ["y2", "y1"]

def test_filter_evidence_min_b():
    hits = [_hit("a", ev="A"), _hit("b", ev="B"), _hit("c", ev="C")]
    assert [h.chunk_id for h in filter_evidence(hits, "B")] == ["a", "b"]
