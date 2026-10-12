# tests/test_golden_set.py
import json
from lifeguide.eval.golden_set import stratified_sample, ZERO_HIT_QUERIES, MULTI_HOP_PAIRS

def test_stratified_covers_all_sections():
    from lifeguide.parser.models import Entry
    entries = [Entry(entry_id=f"{s}-{i}", source_file="f", section_no=s, item_no=i,
                     title=f"t{s}-{i}", cost=None, plain=None, benefit=None, note=None,
                     evidence_level="A", sources=[], cost_tags=None, topics=[],
                     full_text="x") for s in range(1, 35) for i in range(1, 3)]
    picked = stratified_sample(entries, per_section=1, total=70)
    assert len({e.section_no for e in picked}) == 34
    assert len(picked) == 34  # 34 章 × 1 条 < 70 上限

def test_zero_hit_and_pairs_defined():
    assert len(ZERO_HIT_QUERIES) == 20
    assert all(len(pair) == 2 and pair[0] != pair[1] for pair in MULTI_HOP_PAIRS)
    assert len(MULTI_HOP_PAIRS) >= 20
