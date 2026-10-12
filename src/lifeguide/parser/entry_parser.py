# src/lifeguide/parser/entry_parser.py
import re
from pathlib import Path
from lifeguide.parser.models import CostTags, Entry, Source

ENTRY_RE = re.compile(r"^### (\d+)\. (.+)$", re.M)
TAG_RE = re.compile(r"<!--\s*成本标签:\s*(.+?)\s*-->")
URL_RE = re.compile(r"<(https?://[^>]+)>")
FIELD_NAMES = ["成本", "说人话", "收益", "证据等级", "来源", "备注"]
FIELD_RE = re.compile(r"^- (" + "|".join(FIELD_NAMES) + r")：", re.M)

def parse_cost_tag_line(tag_kv: str) -> dict[str, str]:
    out = {}
    for pair in tag_kv.split():
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k] = v
    return out

def parse_sources(field_text: str) -> list[Source]:
    parts = [p.strip() for p in re.split(r"[；;]", field_text) if p.strip()]
    sources = []
    for p in parts:
        m = URL_RE.search(p)
        clean = URL_RE.sub("", p).strip(" .；;，,")
        sources.append(Source(text=clean, url=m.group(1) if m else None))
    return sources

def _split_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    matches = list(FIELD_RE.finditer(body))
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        fields[name] = body[start:end].strip()
    return fields

def _norm_evidence(raw: str) -> str | None:
    # 语料实测变体（05/10/13/23 章共 5 条）："A（争议）"、"B（指南强推荐，但底层证据等级低）"。
    # 取首字母规范化，括号注释原文保留在 full_text；其余异常值置 None。
    v = raw.strip()
    if v in ("A", "B", "C"):
        return v
    return v[:1] if v[:1] in ("A", "B", "C") else None

def parse_book_file(path: Path) -> list[Entry]:
    text = path.read_text(encoding="utf-8")
    rel = f"book/{path.name}"
    header = re.match(r"# (\d+)\. ", text)
    section_no = int(header.group(1)) if header else _infer_section(path.name)
    topic = re.sub(r"^\d+-", "", path.stem)
    entries: list[Entry] = []
    for m in ENTRY_RE.finditer(text):
        item_no, title = int(m.group(1)), m.group(2).strip()
        start = m.end()
        nxt = ENTRY_RE.search(text, start)
        body = text[start: nxt.start() if nxt else len(text)]
        tag_m = TAG_RE.search(body)
        cost_tags = None
        if tag_m:
            kv = parse_cost_tag_line(tag_m.group(1))
            cost_tags = CostTags(money=kv.get("钱"), time=kv.get("时间"),
                                 willpower=kv.get("毅力"), benefit=kv.get("收益"),
                                 scope=kv.get("口径"))
            body = body.replace(tag_m.group(0), "", 1)
        f = _split_fields(body)
        entries.append(Entry(
            entry_id=Entry.make_id(section_no, item_no), source_file=rel,
            section_no=section_no, item_no=item_no, title=title,
            cost=f.get("成本"), plain=f.get("说人话"), benefit=f.get("收益"),
            note=f.get("备注"),
            evidence_level=_norm_evidence(f.get("证据等级", "")),
            sources=parse_sources(f.get("来源", "")),
            cost_tags=cost_tags, topics=[topic],
            full_text=f"### {item_no}. {title}\n" + body.strip(),
        ))
    return entries

def _infer_section(stem: str) -> int:
    m = re.match(r"(\d+)", stem)
    return int(m.group(1)) if m else 0
