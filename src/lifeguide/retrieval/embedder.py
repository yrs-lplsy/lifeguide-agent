# src/lifeguide/retrieval/embedder.py
import hashlib
import json
from pathlib import Path


def to_sparsevec_str(sparse: dict[int, float]) -> str:
    items = ",".join(f"{i}:{w}" for i, w in sorted(sparse.items()))
    return "{" + items + "}"


def cache_key(text: str, model_name: str) -> str:
    return hashlib.sha256(f"{model_name}:{text}".encode()).hexdigest()


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", cache_dir: Path | None = None):
        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from FlagEmbedding import BGEM3FlagModel
        self.model_name = model_name
        self.model = BGEM3FlagModel(model_name, use_fp16=False)  # CPU
        from lifeguide.config import settings_from_env
        self.cache_dir = cache_dir or settings_from_env().embed_cache_dir
        self._cache: dict[str, tuple[list[float], dict[int, float]]] = {}
        self._load_cache()

    def _load_cache(self):
        # 在此初始化：支持 Embedder.__new__ 跳过 __init__ 的用法（测试/mock 场景）
        if not hasattr(self, "_cache"):
            self._cache = {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / f"{self.model_name.replace('/', '_')}.jsonl"
        if self._cache_file.exists():
            for line in self._cache_file.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._cache[rec["k"]] = (rec["d"], {int(i): w for i, w in rec["s"].items()})
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    continue  # 单行损坏/截断仅跳过，避免全量构建被一行打断（重建即补）

    def _save_cache(self, key: str, value: tuple[list[float], dict[int, float]]):
        with open(self._cache_file, "a", encoding="utf-8") as f:
            d, s = value
            f.write(json.dumps({"k": key, "d": d, "s": s}, ensure_ascii=False) + "\n")

    def embed(self, texts: list[str]) -> list[tuple[list[float], dict[int, float]]]:
        results: list[tuple[list[float], dict[int, float]] | None] = [None] * len(texts)
        todo, todo_idx = [], []
        for i, t in enumerate(texts):
            k = cache_key(t, self.model_name)
            if k in self._cache:
                results[i] = self._cache[k]
            else:
                todo.append(t)
                todo_idx.append(i)
        if todo:
            out = self.model.encode(todo, return_sparse=True, return_dense=True,
                                    return_colbert_vecs=False)
            # 兼容 FlagEmbedding 各版本返回形态：
            # dict 且键为 dense_vecs（1.4.x）/ dense_embeds（旧 dict 版）；属性对象（旧版/mock）
            if isinstance(out, dict):
                dense_embeds = out.get("dense_vecs")
                if dense_embeds is None:
                    dense_embeds = out["dense_embeds"]
                lexical_weights = out["lexical_weights"]
            else:
                dense_embeds = getattr(out, "dense_embeds", None)
                if dense_embeds is None:
                    dense_embeds = out.dense_vecs
                lexical_weights = out.lexical_weights
            # lexical_weights 键为 token_id（新版为 str），统一转 int 以匹配 EntryStore 契约
            for i, t, dense, sparse in zip(todo_idx, todo, dense_embeds, lexical_weights):
                # float(x)：ndarray 元素为 np.float32，json 无法序列化 numpy 标量
                value = ([float(x) for x in dense], {int(k): float(v) for k, v in sparse.items()})
                results[i] = value
                k = cache_key(t, self.model_name)
                self._cache[k] = value
                self._save_cache(k, value)
        return results
