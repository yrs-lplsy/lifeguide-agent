# tests/test_models.py
import pytest
from lifeguide.parser.models import Entry, Source, CostTags, DocChunk

def test_entry_roundtrip_and_id():
    e = Entry(entry_id=Entry.make_id(19, 4), source_file="book/19-在职离职和工伤.md",
              section_no=19, item_no=4, title="被裁先算清 N",
              cost="不花钱", plain="干满一年给一个月工资", benefit="劳动合同法…",
              note="月工资按 12 个月平均", evidence_level="A",
              sources=[Source(text="劳动合同法第四十七条", url="http://www.gov.cn/...")],
              cost_tags=CostTags(money="0", time="少", willpower="否", benefit="大", scope="金钱"),
              topics=["在职离职和工伤"], full_text="…")
    assert e.entry_id == "19-4"
    assert e.model_dump()["cost_tags"]["scope"] == "金钱"

def test_entry_optional_fields_default_none():
    d = DocChunk(doc_id="docs-被裁了之后先做什么-2", source_file="docs/被裁了之后先做什么.md",
                 heading="当天：签字之前", text="公司递来「个人原因主动辞职」的单子，不签…")
    assert d.doc_id.startswith("docs-")

def test_evidence_level_rejects_invalid():
    with pytest.raises(Exception):
        Entry(entry_id="1-1", source_file="f", section_no=1, item_no=1, title="t",
              cost=None, plain=None, benefit=None, note=None, evidence_level="X",
              sources=[], cost_tags=None, topics=[], full_text="t")
