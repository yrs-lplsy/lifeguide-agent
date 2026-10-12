# scripts/parse_all.py
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifeguide.config import settings_from_env
from lifeguide.parser.entry_parser import parse_all

def main():
    s = settings_from_env()
    entries, docs = parse_all(s.corpus_dir)
    s.data_dir.mkdir(exist_ok=True)
    with open(s.data_dir / "entries.jsonl", "w", encoding="utf-8") as f:
        for e in entries:
            f.write(e.model_dump_json() + "\n")
    with open(s.data_dir / "docs_chunks.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(d.model_dump_json() + "\n")
    lv = [e.evidence_level or "?" for e in entries]
    print(f"book 条目: {len(entries)}（期望 687）")
    print(f"证据分级: A={lv.count('A')} B={lv.count('B')} C={lv.count('C')} 缺失={lv.count('?')}")
    print(f"docs 块: {len(docs)}")
    if len(entries) != 687:
        sys.exit("对账失败：条目数 != 687")

if __name__ == "__main__":
    main()
