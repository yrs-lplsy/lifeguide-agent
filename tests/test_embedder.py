# tests/test_embedder.py
from lifeguide.retrieval.embedder import to_sparsevec_str, cache_key

def test_sparsevec_str_roundtrip():
    s = to_sparsevec_str({1: 0.12, 57: 0.3})
    assert s == "{1:0.12,57:0.3}"

def test_cache_key_stable():
    assert cache_key("abc", "m") == cache_key("abc", "m")
    assert cache_key("abc", "m") != cache_key("abd", "m")

def test_embedder_caching(tmp_path, monkeypatch):
    from lifeguide.retrieval.embedder import Embedder
    calls = []
    class FakeModel:
        def encode(self, texts, **kw):
            calls.append(list(texts))
            return type("R", (), {
                "dense_embeds": [[0.1, 0.2]] * len(texts),
                "lexical_weights": [{5: 0.7}] * len(texts),
            })()
    emb = Embedder.__new__(Embedder)  # 跳过模型加载
    emb.model, emb.model_name = FakeModel(), "fake"
    emb.cache_dir = tmp_path
    emb._load_cache()
    out1 = emb.embed(["你好"])
    out2 = emb.embed(["你好"])
    assert out1 == out2 == [([0.1, 0.2], {5: 0.7})]
    assert len(calls) == 1  # 第二次命中缓存
