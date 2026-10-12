# scripts/build_index.py
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifeguide.config import settings_from_env
from lifeguide.indexing import make_child_chunks
from lifeguide.parser.models import Entry
from lifeguide.store.store import EntryStore, ChildChunk
from lifeguide.retrieval.embedder import Embedder

def main():
    s = settings_from_env()
    store = EntryStore(s.database_url)
    store.init_schema(); store.reset()
    entries = [Entry.model_validate_json(l) for l in
               (s.data_dir / "entries.jsonl").read_text(encoding="utf-8").splitlines() if l]
    store.upsert_entries(entries)
    specs = []
    for e in entries:
        specs.extend(make_child_chunks(e))
    texts = [spec["text"] for spec in specs]
    emb = Embedder()
    vectors = emb.embed(texts)
    chunks = [ChildChunk(spec["chunk_id"], spec["entry_id"], spec["field_name"],
                         spec["text"], dense, sparse, spec["metadata"])
              for spec, (dense, sparse) in zip(specs, vectors)]
    store.upsert_chunks(chunks)
    print(f"entries={len(entries)} chunks={len(chunks)} 完成（幂等，可重跑）")

if __name__ == "__main__":
    main()
