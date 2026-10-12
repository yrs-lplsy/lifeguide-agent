# src/lifeguide/indexing.py
import re
from lifeguide.parser.models import Entry

MAX_FIELD_CHARS = 800

def _split_long(text: str, limit: int = MAX_FIELD_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, buf = [], ""
    for sent in re.split(r"(?<=[。！？；])", text):
        if buf and len(buf) + len(sent) > limit:
            parts.append(buf); buf = sent
        else:
            buf += sent
    if buf:
        parts.append(buf)
    return parts

def make_child_chunks(entry: Entry) -> list[dict]:
    ct = entry.cost_tags
    metadata = {"evidence_level": entry.evidence_level,
                "benefit": ct.benefit if ct else None,
                "cost_tag_money": ct.money if ct else None,
                "section_no": entry.section_no, "item_no": entry.item_no,
                "title": entry.title}
    chunks = []
    for field_name in ("title", "cost", "plain", "benefit", "note"):
        value = getattr(entry, field_name)
        if not value:
            continue
        for i, piece in enumerate(_split_long(value), start=1):
            suffix = f"-{i}" if len(_split_long(value)) > 1 else ""
            chunks.append({"chunk_id": f"{entry.entry_id}#{field_name}{suffix}",
                           "entry_id": entry.entry_id, "field_name": field_name,
                           "text": piece, "metadata": metadata})
    return chunks
