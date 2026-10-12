# scripts/gen_golden.py — 生成 golden set（LLM 反向生成 + 对账 + 产物落盘）
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifeguide.config import settings_from_env
from lifeguide.parser.models import Entry
from lifeguide.eval.golden_set import MULTI_HOP_PAIRS, build_golden, load_golden

def normalize_id(eid: str) -> str:
    """entries.jsonl 的 entry_id 无前导零（如 7-2）；写死的对子用了 07-2 这类补零格式。"""
    try:
        a, b = eid.split("-")
        return f"{int(a)}-{int(b)}"
    except ValueError:
        return eid

def probe_llm(s):
    """按序探测 (base_url, model) 组合，返回第一个能完成一次最小对话的 (client, model, combo)。"""
    from openai import OpenAI
    combos = [(s.llm_base_url, s.llm_model)]
    if "deepseek" in s.llm_base_url:
        combos.append((s.llm_base_url, "deepseek-chat"))
    errors = []
    for base, model in combos:
        try:
            client = OpenAI(base_url=base, api_key=s.llm_api_key, timeout=60)
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping，请回复 pong"}],
                temperature=0, max_tokens=8)
            _ = r.choices[0].message.content
            return client, model, (base, model)
        except Exception as e:
            errors.append(f"{base} + {model}: {type(e).__name__}: {e}")
    return None, None, errors

def reconcile_pairs(entries: list[Entry]):
    """对账 MULTI_HOP_PAIRS：归一化 id → 校验存在；缺失 warning+跳过；若 >5 对失效，
    用同章真实邻居条目（同章最近 item_no）替换以保住 20 对结构。"""
    by_id = {e.entry_id: e for e in entries}
    by_sec: dict[int, list[Entry]] = {}
    for e in entries:
        by_sec.setdefault(e.section_no or 0, []).append(e)

    def nearest_in_chapter(eid: str) -> Entry | None:
        sec, item = eid.split("-")[:2]
        try:
            sec_i, item_i = int(sec), int(item)
        except ValueError:
            return None
        cand = by_sec.get(sec_i)
        return min(cand, key=lambda e: abs((e.item_no or 0) - item_i)) if cand else None

    normalized: list[tuple[str, str]] = []
    substitutions: list[tuple[str, str, str]] = []  # (原id, 替换id, 替换标题)
    missing: list[str] = []
    for a, b in MULTI_HOP_PAIRS:
        pair = []
        for eid in (a, b):
            nid = normalize_id(eid)
            if nid in by_id:
                pair.append(nid)
            else:
                print(f"warning: pair id {eid}（归一化为 {nid}）不在 entries.jsonl，"
                      f"尝试同章最近条目替换")
                sub = nearest_in_chapter(nid)
                if sub:
                    pair.append(sub.entry_id)
                    substitutions.append((eid, sub.entry_id, sub.title))
                else:
                    missing.append(eid)
        if len(pair) == 2 and pair[0] != pair[1]:
            normalized.append((pair[0], pair[1]))
        else:
            print(f"warning: 对子 ({a}, {b}) 无法修复，跳过其问题生成")
    invalid_pairs = len(MULTI_HOP_PAIRS) - len(normalized)
    if invalid_pairs:
        print(f"warning: {invalid_pairs} 个对子失效")
    return normalized, substitutions

def main():
    s = settings_from_env()
    entries = [Entry.model_validate_json(l) for l in
               (s.data_dir / "entries.jsonl").read_text(encoding="utf-8").splitlines() if l]
    print(f"entries: {len(entries)}")

    client, model, combo = probe_llm(s)
    if client is None:
        print("BLOCKED: 所有 LLM 组合探测失败：")
        for e in combo:
            print(f"  - {e}")
        sys.exit(1)
    print(f"LLM probe OK: base={combo[0]} model={combo[1]}")

    pairs, subs = reconcile_pairs(entries)
    print(f"multi_hop 对子（对账后）: {len(pairs)}/{len(MULTI_HOP_PAIRS)}")
    if subs:
        print("对账替换明细：")
        for orig, new, title in subs:
            print(f"  {orig} -> {new}（{title}）")

    docs = []  # build_golden 兼容参数，本任务未用
    golden = build_golden(entries, docs, client, model=model, pairs=pairs)

    out_jsonl = s.data_dir / "golden_set.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for g in golden:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    by_id = {e.entry_id: e for e in entries}
    out_csv = s.data_dir / "golden_review.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "query", "relevant_titles", "qtype", "verdict"])
        for g in golden:
            titles = " | ".join(by_id[r].title for r in g["relevant"] if r in by_id)
            w.writerow([g["qid"], g["query"], titles, g["qtype"], ""])

    counts: dict[str, int] = {}
    for g in golden:
        counts[g["qtype"]] = counts.get(g["qtype"], 0) + 1
    print(f"golden 总数: {len(golden)}")
    print(f"按 qtype: colloquial={counts.get('colloquial', 0)} "
          f"multi_hop={counts.get('multi_hop', 0)} zero_hit={counts.get('zero_hit', 0)}")
    assert load_golden(out_jsonl) == golden, "jsonl 回读不一致"
    print(f"written: {out_jsonl}（{len(golden)} 行）, {out_csv}")

if __name__ == "__main__":
    main()
