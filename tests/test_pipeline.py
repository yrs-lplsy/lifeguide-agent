# tests/test_pipeline.py
import pytest
from lifeguide.parser.models import Entry, Source, CostTags
from lifeguide.indexing import make_child_chunks

LONG = "每满一年支付一个月工资。" * 120  # ~1500 字符

@pytest.fixture
def entry():
    return Entry(entry_id="19-4", source_file="f", section_no=19, item_no=4,
                 title="被裁先算清 N", cost="不花钱", plain="干满一年给一个月工资",
                 benefit=LONG, note="月工资按 12 个月平均", evidence_level="A",
                 sources=[Source(text="劳动合同法", url="http://x")],
                 cost_tags=CostTags(money="0", time="少", willpower="否", benefit="大", scope="金钱"),
                 topics=["在职离职和工伤"], full_text="…")

def test_make_child_chunks_fields(entry):
    chunks = make_child_chunks(entry)
    fields = {c["field_name"] for c in chunks}
    assert {"cost", "plain", "benefit", "note"} <= fields
    assert all(c["chunk_id"].startswith("19-4#") for c in chunks)
    assert all(c["metadata"]["evidence_level"] == "A" for c in chunks)

def test_long_field_split(entry):
    ben = [c for c in make_child_chunks(entry) if c["field_name"] == "benefit"]
    assert len(ben) >= 2 and all(len(c["text"]) <= 820 for c in ben)
