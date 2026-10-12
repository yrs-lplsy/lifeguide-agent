# tests/test_entry_parser.py
from pathlib import Path
import pytest
from lifeguide.parser.entry_parser import parse_book_file, parse_cost_tag_line, parse_sources

@pytest.fixture
def ch19(settings):
    p = settings.corpus_dir / "book/19-在职离职和工伤.md"
    if not p.exists():
        pytest.skip("corpus missing")
    return parse_book_file(p)

def test_chapter19_count_and_ids(ch19):
    assert len(ch19) == 19
    assert [e.item_no for e in ch19] == list(range(1, 20))
    assert ch19[3].entry_id == "19-4" and ch19[3].section_no == 19

def test_cost_tag_parsing():
    ct = parse_cost_tag_line("钱=0 时间=少 毅力=些 收益=大 口径=金钱")
    assert ct == {"钱": "0", "时间": "少", "毅力": "些", "收益": "大", "口径": "金钱"}

def test_sources_split_and_url(ch19):
    e = ch19[0]  # 19-1 加班费：两个来源，各带 URL
    assert len(e.sources) >= 2
    assert all(s.url and s.url.startswith("http") for s in e.sources)
    assert "劳动法" in e.sources[0].text

def test_fields_content(ch19):
    e = ch19[3]  # 19-4 被裁先算清 N
    assert e.evidence_level == "A"
    assert e.cost_tags is not None and e.cost_tags.scope == "金钱"
    assert "每满一年支付一个月工资" in (e.benefit or "")
    assert e.title.startswith("被裁先算清")

def test_multiline_field_keeps_full_text(ch19):
    e = ch19[1]  # 19-2 年休假，备注含多行条件
    assert "自愿放弃" in (e.note or "") and "21.75" in (e.benefit or "")

def test_evidence_level_variant_and_missing_fields(tmp_path):
    # 语料实测变体（05/10/13/23 章共 5 条）：证据等级带括号注释，如 "A（争议）"、"B（指南强推荐，但底层证据等级低）"。
    # 规范化为首字母，注释保留在 full_text；缺字段仍置 None/[] 容错。
    p = tmp_path / "99-测试.md"
    p.write_text(
        "# 99. 测试\n\n"
        "### 1. 带注释\n\n"
        "- 成本：x\n- 说人话：y\n- 收益：z\n"
        "- 证据等级：A（争议）\n- 来源：甲 <https://a.example/c>\n- 备注：n\n\n"
        "### 2. 缺字段\n\n"
        "- 成本：x\n- 收益：z\n",
        encoding="utf-8")
    es = parse_book_file(p)
    assert len(es) == 2
    assert es[0].evidence_level == "A" and "（争议）" in es[0].full_text
    assert es[0].sources[0].url == "https://a.example/c"
    assert es[1].evidence_level is None and es[1].sources == [] and es[1].plain is None
