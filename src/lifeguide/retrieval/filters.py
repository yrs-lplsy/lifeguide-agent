# src/lifeguide/retrieval/filters.py
_EV = {"A": 3, "B": 2, "C": 1, None: 0}
_BEN = {"大": 3, "中": 2, "小": 1, None: 0}
_COST = {"0": 3, "少": 2, "中": 1, "大": 0, "多": 0, None: 1}  # 成本越小越优先（语料实际值域: 0/少/多；中/大 为容错键）

def filter_evidence(hits, minimum: str):
    floor = _EV[minimum]
    return [h for h in hits if _EV.get(h.evidence_level, 0) >= floor]

def sort_by_value(hits):
    return sorted(hits, key=lambda h: (-_EV.get(h.evidence_level, 0),
                                       -_BEN.get(h.benefit, 0),
                                       -_COST.get(h.cost_tag, 0)))
