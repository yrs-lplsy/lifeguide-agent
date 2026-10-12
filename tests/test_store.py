# tests/test_store.py
import pytest
from lifeguide.store.store import EntryStore, ChildChunk

@pytest.fixture
def store(settings):
    st = EntryStore(settings.database_url)
    st.init_schema()
    st.reset()  # 清空两表，测试隔离
    return st

def _fake_dense(dim=8):
    import random
    random.seed(7)
    return [random.random() for _ in range(dim)]

@pytest.mark.integration
def test_roundtrip_and_dense_search(store):
    from lifeguide.parser.models import Entry, Source
    # dense 维度跟随 store.dim（EMBED_DIM=1024 时库列是 vector(1024)，写死 8 维会被
    # pgvector 拒绝：expected 1024 dimensions, not 8）
    dense = _fake_dense(store.dim)
    e = Entry(entry_id="19-4", source_file="f", section_no=19, item_no=4, title="被裁先算清 N",
              cost=None, plain="干满一年给一个月工资", benefit=None, note=None,
              evidence_level="A", sources=[Source(text="劳动合同法", url="http://x")],
              cost_tags=None, topics=["在职离职和工伤"], full_text="…")
    store.upsert_entries([e])
    chunk = ChildChunk(chunk_id="19-4#plain", entry_id="19-4", field_name="plain",
                       text=e.plain, dense=dense, sparse={1: 0.5, 9: 0.2},
                       metadata={"evidence_level": "A"})
    store.upsert_chunks([chunk])
    got = store.get_entry("19-4")
    assert got is not None and got.evidence_level == "A"
    hits = store.search_dense(dense, k=5)
    assert hits[0][0] == "19-4#plain"
    shits = store.search_sparse({1: 0.9}, k=5)
    assert shits[0][0] == "19-4#plain"
