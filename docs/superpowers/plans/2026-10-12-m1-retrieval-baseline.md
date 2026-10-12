# M1 检索基线（条目解析器 + 混合检索 + 评测）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 HowToLiveBetter 语料（34 章 687 条 + docs 长文）解析为结构化条目，建 pgvector 三级索引，交付「三路召回 + RRF + rerank + 元数据过滤」混合检索管线，并用 ~200 条 golden set 产出四组消融评测报告。

**Architecture:** 纯离线 Python 管线（无 API 服务，API 层属 M2）。数据流：`book/*.md + docs/*.md → 解析器 → entries.jsonl → 父子 chunk → bge-m3 嵌入 → PostgreSQL(pgvector)`；查询流：`query → [BM25(jieba+rank_bm25) ∥ dense(bge-m3) ∥ sparse(bge-m3 lexical)] → RRF → bge-reranker → 元数据过滤/排序`。评测流独立于查询流，四组消融通过开关组合复用同一管线。

**Tech Stack:** Python 3.11+、pydantic v2、SQLAlchemy 2.0 + psycopg[binary]、pgvector（`pgvector/pgvector:pg16` Docker 镜像）、FlagEmbedding（bge-m3 + bge-reranker-v2-m3，CPU）、jieba、rank_bm25、openai SDK（仅 golden set 生成用，OpenAI 兼容接口）、pytest。

## Global Constraints

- 语料目录 `HowToLiveBetter/` **只读**，任何任务不得修改其中文件。
- 解析对账硬指标：`book/` 条目总数必须精确等于 **687**（实测 grep `^### ` 计数为 687，无容差）；单条字段缺失置 `null` 不报错。
- bge-m3 稠密向量 **1024 维**；稀疏向量用其 lexical weights（token_id → weight）。
- 与 spec 的实现层偏差（已确认合理）：BM25 路用 `jieba + rank_bm25` 进程内实现，**不用** PG tsvector（PG 原生全文检索中文按空格切词不可用，zhparser 扩展需额外编译；千级 chunk 进程内 BM25 无性能压力且是标准 BM25 公式）。
- 元数据中文值**不翻译**：`钱=少`、`收益=大`、`口径=死亡率` 原样入库（前端/检索映射层处理）。
- 环境变量（`lifeguide/config.py` 统一读取，`.env` 不进 git）：`DATABASE_URL`（默认 `postgresql://lifeguide:lifeguide@localhost:5433/lifeguide`）、`LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`、`HF_ENDPOINT`（默认 `https://hf-mirror.com`，国内镜像）、`EMBED_CACHE_DIR`（默认 `~/.cache/lifeguide/embed`）。
- **LLM Key 分界（spec 7.3 BYOK）**：M1 是离线开发阶段，`LLM_API_KEY` 仅为**开发者自己的 Key**（golden set 生成用，Task 13）；面向用户的「自带 Key + Base URL 配置 + AES-GCM 加密存储」属 M2 在线服务范围，M1 不实现。
- 提交约定：每个任务一次 commit，消息格式 `feat|fix|test|chore(m1): <描述>`；主分支直接提交（单人项目，M2 起再引 worktree 流程）。
- 集成测试标记 `@pytest.mark.integration`（需 Docker PG 起着），单测默认不依赖 PG/模型/网络。

## File Structure

```
lifeguide/（= /home/yrs/JD/Agentic/，git 根，已初始化）
├── pyproject.toml              # Task 1
├── docker-compose.yml          # Task 1（仅 pg 服务 + 可选 redis 占位注释）
├── .env.example                # Task 1
├── src/lifeguide/
│   ├── __init__.py
│   ├── config.py               # Task 1：env 读取，dataclass 配置
│   ├── parser/
│   │   ├── __init__.py
│   │   ├── models.py           # Task 2：Entry/Source/CostTags/DocChunk pydantic
│   │   └── entry_parser.py     # Task 3-4：解析 + 全量对账
│   ├── store/
│   │   ├── __init__.py
│   │   ├── schema.sql          # Task 5：entries/chunks 表 DDL
│   │   └── store.py            # Task 5：EntryStore（建表/写条目/写chunk/向量检索/稀疏检索）
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embedder.py         # Task 6：bge-m3 封装（dense+sparse+磁盘缓存）
│   │   ├── bm25.py             # Task 7：jieba+rank_bm25 封装
│   │   ├── rrf.py              # Task 8：RRF 纯函数
│   │   ├── rerank.py           # Task 9：bge-reranker 封装
│   │   ├── filters.py          # Task 9：证据等级过滤 + 性价比排序
│   │   └── pipeline.py         # Task 8/9：SearchPipeline（消融开关）
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py          # Task 12：recall_at_k/mrr/ndcg_at_k
│   │   ├── golden_set.py       # Task 13：LLM 反向生成 + 校验 + 三类覆盖
│   │   └── ablation.py         # Task 14：四组消融 + markdown 报告
├── data/                       # entries.jsonl、golden_set.jsonl 进 git
├── scripts/
│   ├── parse_all.py            # Task 4 CLI
│   ├── build_index.py          # Task 10 CLI
│   ├── search_cli.py           # Task 11 CLI
│   ├── gen_golden.py           # Task 13 CLI
│   └── run_ablation.py         # Task 14 CLI
└── tests/
    ├── conftest.py             # integration 标记 + fixture
    ├── test_models.py
    ├── test_entry_parser.py
    ├── test_store.py           # integration
    ├── test_embedder.py
    ├── test_bm25.py
    ├── test_rrf.py
    ├── test_rerank_filters.py
    ├── test_pipeline.py        # integration
    ├── test_metrics.py
    └── test_golden_set.py
```

---

### Task 1: 项目脚手架与 PG 环境

**Files:**
- Create: `pyproject.toml`、`docker-compose.yml`、`.env.example`、`src/lifeguide/__init__.py`、`src/lifeguide/config.py`、`tests/conftest.py`、`.gitignore`（追加）

**Interfaces:**
- Produces: `lifeguide.config.Settings`（`settings_from_env() -> Settings`，字段 `database_url: str`、`llm_base_url: str`、`llm_api_key: str`、`llm_model: str`、`hf_endpoint: str`、`embed_cache_dir: Path`、`corpus_dir: Path`（默认 `<repo>/HowToLiveBetter`）、`data_dir: Path`（默认 `<repo>/data`））；pytest 选项 `--integration`

- [ ] **Step 1: 写 pyproject.toml**

```toml
[project]
name = "lifeguide"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.7",
    "sqlalchemy>=2.0",
    "psycopg[binary]>=3.1",
    "pgvector>=0.3",
    "FlagEmbedding>=1.3",
    "jieba>=0.42",
    "rank_bm25>=0.2.2",
    "openai>=1.30",
    "python-dotenv>=1.0",
    "rich>=13.7",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0"]

[tool.pytest.ini_options]
markers = ["integration: needs docker PG up"]
testpaths = ["tests"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/lifeguide"]
```

- [ ] **Step 2: 写 docker-compose.yml（仅 M1 所需 pg；redis 留注释占位）**

```yaml
services:
  pg:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: lifeguide
      POSTGRES_PASSWORD: lifeguide
      POSTGRES_DB: lifeguide
    ports:
      - "5433:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U lifeguide"]
      interval: 5s
      retries: 10
  # redis:            # M2 启用（缓存/限流/短期记忆）
  #   image: redis:7-alpine
volumes:
  pgdata:
```

- [ ] **Step 3: 写 config.py / .env.example / conftest.py**

```python
# src/lifeguide/config.py
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    database_url: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    hf_endpoint: str
    embed_cache_dir: Path
    corpus_dir: Path
    data_dir: Path

def settings_from_env() -> Settings:
    repo = Path(__file__).resolve().parents[2]
    return Settings(
        database_url=os.getenv("DATABASE_URL", "postgresql://lifeguide:lifeguide@localhost:5433/lifeguide"),
        llm_base_url=os.getenv("LLM_BASE_URL", ""),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", ""),
        hf_endpoint=os.getenv("HF_ENDPOINT", "https://hf-mirror.com"),
        embed_cache_dir=Path(os.getenv("EMBED_CACHE_DIR", Path.home() / ".cache/lifeguide/embed")),
        corpus_dir=Path(os.getenv("CORPUS_DIR", repo / "HowToLiveBetter")),
        data_dir=Path(os.getenv("DATA_DIR", repo / "data")),
    )
```

```bash
# .env.example
DATABASE_URL=postgresql://lifeguide:lifeguide@localhost:5433/lifeguide
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_API_KEY=your-glm-key
LLM_MODEL=glm-4-flash
HF_ENDPOINT=https://hf-mirror.com
```

```python
# tests/conftest.py
import pytest
from lifeguide.config import settings_from_env

def pytest_addoption(parser):
    parser.addoption("--integration", action="store_true", default=False,
                     help="run integration tests (needs docker PG)")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--integration"):
        return
    skip = pytest.mark.skip(reason="need --integration and docker PG")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)

@pytest.fixture
def settings():
    return settings_from_env()
```

`.gitignore` 追加一行：`.env`。

- [ ] **Step 4: 安装并验证**

Run: `cd /home/yrs/JD/Agentic && pip install -e ".[dev]" && pytest && docker compose up -d pg && sleep 5 && docker compose ps`
Expected: pytest 无测试 no tests ran（exit 5 视为通过）；pg 状态 healthy。

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml docker-compose.yml .env.example src/ tests/conftest.py .gitignore
git commit -m "chore(m1): 项目脚手架、PG(pgvector) compose、配置层"
```

---

### Task 2: 条目数据模型

**Files:**
- Create: `src/lifeguide/parser/models.py`、`tests/test_models.py`

**Interfaces:**
- Produces: `Source(text: str, url: str | None)`；`CostTags(money/time/willpower/benefit/scope: str | None)`（中文值原样，如 `money="0"|"少"|"中"|"大"`）；`Entry(entry_id: str, source_file: str, section_no: int | None, item_no: int | None, title: str, cost/plain/benefit/note: str | None, evidence_level: Literal["A","B","C"] | None, sources: list[Source], cost_tags: CostTags | None, topics: list[str], full_text: str)`；类方法 `Entry.make_id(section_no, item_no) -> str`（`"19-4"`）与 `DocChunk(doc_id: str, source_file: str, heading: str, text: str)`。

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 跑测试确认失败**  Run: `pytest tests/test_models.py -v`  Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 models.py**

```python
# src/lifeguide/parser/models.py
from typing import Literal, Optional
from pydantic import BaseModel

class Source(BaseModel):
    text: str
    url: Optional[str] = None

class CostTags(BaseModel):
    money: Optional[str] = None      # 钱
    time: Optional[str] = None       # 时间
    willpower: Optional[str] = None  # 毅力
    benefit: Optional[str] = None    # 收益
    scope: Optional[str] = None      # 口径

class Entry(BaseModel):
    entry_id: str
    source_file: str
    section_no: Optional[int] = None
    item_no: Optional[int] = None
    title: str
    cost: Optional[str] = None        # - 成本：
    plain: Optional[str] = None       # - 说人话：
    benefit: Optional[str] = None     # - 收益：
    note: Optional[str] = None        # - 备注：
    evidence_level: Optional[Literal["A", "B", "C"]] = None
    sources: list[Source] = []
    cost_tags: Optional[CostTags] = None
    topics: list[str] = []
    full_text: str = ""

    @staticmethod
    def make_id(section_no: int, item_no: int) -> str:
        return f"{section_no}-{item_no}"

class DocChunk(BaseModel):
    doc_id: str
    source_file: str
    heading: str
    text: str
```

- [ ] **Step 4: 跑测试确认通过**  Run: `pytest tests/test_models.py -v`  Expected: 3 passed
- [ ] **Step 5: Commit**  `git add src/lifeguide/parser/models.py tests/test_models.py && git commit -m "feat(m1): 条目/来源/成本标签 pydantic 模型"`

---

### Task 3: 条目解析器（单文件金样本）

**Files:**
- Create: `src/lifeguide/parser/entry_parser.py`、`tests/test_entry_parser.py`

**Interfaces:**
- Consumes: Task 2 的 `Entry/Source/CostTags`
- Produces: `parse_book_file(path: Path) -> list[Entry]`；`parse_cost_tag_line(line: str) -> CostTags`；`parse_sources(field_text: str) -> list[Source]`；内部正则常量 `ENTRY_RE`、`FIELD_RE`、`TAG_RE`、`URL_RE`

**语料格式事实**（实测，写死进测试）：条目头 `### N. 标题`；标签行紧随其后 `<!-- 成本标签: 钱=0 时间=少 毅力=些 收益=大 口径=金钱 -->`；字段行以 `- 成本：/- 说人话：/- 收益：/- 证据等级：/- 来源：/- 备注：` 开头（全角冒号），字段值可跨多行直到下一个字段行或条目头；来源中 URL 以 `<https://…>` 包裹、多个来源以 `；` 分隔；687 条六字段实测零缺失（缺字段仍置 None 容错）。

- [ ] **Step 1: 写失败测试（真实语料做 fixture）**

```python
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
```

- [ ] **Step 2: 跑测试确认失败**  Run: `pytest tests/test_entry_parser.py -v`  Expected: FAIL（ModuleNotFoundError）
- [ ] **Step 3: 实现 entry_parser.py**

```python
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
            evidence_level=f.get("证据等级", "").strip() or None,
            sources=parse_sources(f.get("来源", "")),
            cost_tags=cost_tags, topics=[topic],
            full_text=f"### {item_no}. {title}\n" + body.strip(),
        ))
    return entries

def _infer_section(stem: str) -> int:
    m = re.match(r"(\d+)", stem)
    return int(m.group(1)) if m else 0
```

- [ ] **Step 4: 跑测试确认通过**  Run: `pytest tests/test_entry_parser.py -v`  Expected: 5 passed
- [ ] **Step 5: Commit**  `git add src/lifeguide/parser/entry_parser.py tests/test_entry_parser.py && git commit -m "feat(m1): book 条目解析器（字段切分/成本标签/来源URL）"`

---

### Task 4: 全量解析 CLI 与 687 对账

**Files:**
- Create: `src/lifeguide/parser/entry_parser.py` 内追加 `parse_docs_file`、`parse_all`；`scripts/parse_all.py`；`data/entries.jsonl`（产物）；`data/docs_chunks.jsonl`（产物）

**Interfaces:**
- Consumes: Task 3 的 `parse_book_file`
- Produces: `parse_docs_file(path: Path) -> list[DocChunk]`（按 `## ` 二级标题切块，`doc_id="docs-{stem}-{i}"`）；`parse_all(corpus_dir: Path) -> tuple[list[Entry], list[DocChunk]]`；CLI：`python scripts/parse_all.py` 输出对账报告并写 `data/entries.jsonl`、`data/docs_chunks.jsonl`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_entry_parser.py 追加
from lifeguide.parser.entry_parser import parse_all, parse_docs_file

def test_parse_all_reconciles_687(settings):
    entries, docs = parse_all(settings.corpus_dir)
    assert len(entries) == 687, f"book 条目数 {len(entries)} != 687"
    assert all(e.evidence_level in ("A", "B", "C") for e in entries)
    lv = [e.evidence_level for e in entries]
    assert lv.count("A") + lv.count("B") + lv.count("C") == 687
    ids = [e.entry_id for e in entries]
    assert len(set(ids)) == 687
    assert len(docs) >= 20  # docs/ 长文按 ## 切块

def test_docs_chunk_ids(settings):
    chunks = parse_docs_file(settings.corpus_dir / "docs/被裁了之后先做什么.md")
    assert chunks[0].doc_id == "docs-被裁了之后先做什么-1"
    assert chunks[0].heading == "当天：签字之前"
```

- [ ] **Step 2: 跑测试确认失败**  Run: `pytest tests/test_entry_parser.py -v`  Expected: 新增 2 条 FAIL
- [ ] **Step 3: 实现追加函数与 CLI**

```python
# entry_parser.py 追加
from lifeguide.parser.models import DocChunk

DOC_HEAD_RE = re.compile(r"^## (.+)$", re.M)

def parse_docs_file(path: Path) -> list[DocChunk]:
    text = path.read_text(encoding="utf-8")
    rel = f"docs/{path.name}"
    chunks: list[DocChunk] = []
    for i, m in enumerate(DOC_HEAD_RE.finditer(text), start=1):
        start = m.end()
        nxt = DOC_HEAD_RE.search(text, start)
        body = text[start: nxt.start() if nxt else len(text)].strip()
        chunks.append(DocChunk(doc_id=f"docs-{path.stem}-{i}", source_file=rel,
                               heading=m.group(1).strip(), text=body))
    return chunks

def parse_all(corpus_dir: Path) -> tuple[list[Entry], list[DocChunk]]:
    entries: list[Entry] = []
    for p in sorted((corpus_dir / "book").glob("*.md")):
        entries.extend(parse_book_file(p))
    docs: list[DocChunk] = []
    for p in sorted((corpus_dir / "docs").glob("*.md")):
        if p.parent.name == "核实记录":
            continue
        docs.extend(parse_docs_file(p))
    return entries, docs
```

```python
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
```

- [ ] **Step 4: 跑测试 + CLI**

Run: `pytest tests/test_entry_parser.py -v && python scripts/parse_all.py`
Expected: 全部 passed；CLI 打印 `book 条目: 687（期望 687）`，A/B/C 计数合计 687。

- [ ] **Step 5: Commit**  `git add -A && git commit -m "feat(m1): 全量解析+687对账+entries.jsonl 产物"`

---

### Task 5: PG 存储层（pgvector 稠密/稀疏）

**Files:**
- Create: `src/lifeguide/store/schema.sql`、`src/lifeguide/store/store.py`、`tests/test_store.py`

**Interfaces:**
- Consumes: Task 2 模型
- Produces: `EntryStore(database_url: str)`，方法：`init_schema()`；`upsert_entries(entries: list[Entry])`；`upsert_chunks(chunks: list[ChildChunk])`（`ChildChunk` 在本文件定义：`chunk_id: str, entry_id: str, field_name: str, text: str, dense: list[float], sparse: dict[int, float], metadata: dict`）；`search_dense(qvec, k) -> list[tuple[chunk_id, score, entry_id]]`；`search_sparse(qsparse: dict[int,float], k) -> list[tuple[chunk_id, score, entry_id]]`；`get_entry(entry_id) -> Entry | None`；`chunks_by_entry(entry_id) -> list[ChildChunk]`；`all_chunks() -> list[ChildChunk]`（BM25 用）

- [ ] **Step 1: 写 schema.sql**

```sql
-- src/lifeguide/store/schema.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS entries (
    entry_id  TEXT PRIMARY KEY,
    data      JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS entries_data_gin ON entries USING GIN (data);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT PRIMARY KEY,
    entry_id   TEXT NOT NULL REFERENCES entries(entry_id) ON DELETE CASCADE,
    field_name TEXT NOT NULL,
    text       TEXT NOT NULL,
    dense      vector(1024),
    sparse     sparsevec,
    metadata   JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS chunks_dense_hnsw ON chunks USING hnsw (dense vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_sparse_ivfflat ON chunks USING ivfflat (sparse sparsevec_l2_ops) WITH (lists = 100);
```

- [ ] **Step 2: 写失败测试（integration）**

```python
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
    e = Entry(entry_id="19-4", source_file="f", section_no=19, item_no=4, title="被裁先算清 N",
              cost=None, plain="干满一年给一个月工资", benefit=None, note=None,
              evidence_level="A", sources=[Source(text="劳动合同法", url="http://x")],
              cost_tags=None, topics=["在职离职和工伤"], full_text="…")
    store.upsert_entries([e])
    chunk = ChildChunk(chunk_id="19-4#plain", entry_id="19-4", field_name="plain",
                       text=e.plain, dense=_fake_dense(), sparse={1: 0.5, 9: 0.2},
                       metadata={"evidence_level": "A"})
    store.upsert_chunks([chunk])
    got = store.get_entry("19-4")
    assert got is not None and got.evidence_level == "A"
    hits = store.search_dense(_fake_dense(), k=5)
    assert hits[0][0] == "19-4#plain"
    shits = store.search_sparse({1: 0.9}, k=5)
    assert shits[0][0] == "19-4#plain"
```

注意：`dense` 生产为 1024 维，测试用 8 维需 schema 兼容——实现时 schema 以环境变量 `EMBED_DIM`（默认 1024）建列，测试 env 设 8。`EntryStore.__init__` 读 `EMBED_DIM` 并对 `vector({dim})` 做 `.format(dim=dim)`。

- [ ] **Step 3: 跑测试确认失败**  Run: `pytest tests/test_store.py -v --integration`  Expected: FAIL（ModuleNotFoundError）
- [ ] **Step 4: 实现 store.py**

```python
# src/lifeguide/store/store.py
import json, os
from pathlib import Path
import psycopg
from pgvector.psycopg import register_vector, SparseVector
from lifeguide.parser.models import Entry

class ChildChunk:
    def __init__(self, chunk_id, entry_id, field_name, text, dense, sparse, metadata):
        self.chunk_id, self.entry_id, self.field_name = chunk_id, entry_id, field_name
        self.text, self.dense, self.sparse, self.metadata = text, dense, sparse, metadata

class EntryStore:
    def __init__(self, database_url: str, embed_dim: int | None = None):
        self.url = database_url
        self.dim = embed_dim or int(os.getenv("EMBED_DIM", "1024"))
        self.conn = psycopg.connect(database_url, autocommit=True)
        register_vector(self.conn)

    def init_schema(self):
        schema = Path(__file__).parent.joinpath("schema.sql").read_text(encoding="utf-8")
        schema = schema.replace("vector(1024)", f"vector({self.dim})")
        with self.conn.cursor() as cur:
            cur.execute(schema)

    def reset(self):
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE chunks, entries CASCADE")

    def upsert_entries(self, entries: list[Entry]):
        with self.conn.cursor() as cur:
            for e in entries:
                cur.execute(
                    "INSERT INTO entries (entry_id, data) VALUES (%s, %s::jsonb) "
                    "ON CONFLICT (entry_id) DO UPDATE SET data = EXCLUDED.data",
                    (e.entry_id, e.model_dump_json()))

    def upsert_chunks(self, chunks: list[ChildChunk]):
        with self.conn.cursor() as cur:
            for c in chunks:
                sv = SparseVector(c.sparse) if c.sparse else None
                cur.execute(
                    "INSERT INTO chunks (chunk_id, entry_id, field_name, text, dense, sparse, metadata) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT (chunk_id) DO UPDATE SET "
                    "dense=EXCLUDED.dense, sparse=EXCLUDED.sparse, text=EXCLUDED.text, metadata=EXCLUDED.metadata",
                    (c.chunk_id, c.entry_id, c.field_name, c.text,
                     c.dense, sv, json.dumps(c.metadata, ensure_ascii=False)))

    def _hits(self, cur) -> list[tuple[str, float, str]]:
        return [(r[0], float(r[1]), r[2]) for r in cur.fetchall()]

    def search_dense(self, qvec: list[float], k: int = 20):
        with self.conn.cursor() as cur:
            cur.execute("SELECT chunk_id, 1 - (dense <=> %s::vector) AS score, entry_id "
                        "FROM chunks WHERE dense IS NOT NULL ORDER BY dense <=> %s::vector LIMIT %s",
                        (qvec, qvec, k))
            return self._hits(cur)

    def search_sparse(self, qsparse: dict[int, float], k: int = 20):
        with self.conn.cursor() as cur:
            sv = SparseVector(qsparse)
            cur.execute("SELECT chunk_id, sparse <-> %s::sparsevec AS score, entry_id "
                        "FROM chunks WHERE sparse IS NOT NULL ORDER BY sparse <-> %s::sparsevec LIMIT %s",
                        (sv, sv, k))
            return self._hits(cur)

    def chunk_meta(self, chunk_id: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT metadata FROM chunks WHERE chunk_id = %s", (chunk_id,))
            row = cur.fetchone()
            return row[0] if row else None

    def chunk_text(self, chunk_id: str) -> str | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT text FROM chunks WHERE chunk_id = %s", (chunk_id,))
            row = cur.fetchone()
            return row[0] if row else None

    def get_entry(self, entry_id: str) -> Entry | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT data FROM entries WHERE entry_id = %s", (entry_id,))
            row = cur.fetchone()
            return Entry.model_validate_json(row[0]) if row else None

    def chunks_by_entry(self, entry_id: str) -> list[ChildChunk]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT chunk_id, entry_id, field_name, text, dense, sparse, metadata "
                        "FROM chunks WHERE entry_id = %s ORDER BY chunk_id", (entry_id,))
            out = []
            for cid, eid, fname, text, dense, sparse, meta in cur.fetchall():
                out.append(ChildChunk(cid, eid, fname, text,
                                      list(dense) if dense is not None else [],
                                      sparse.as_dict() if sparse is not None else {},
                                      meta or {}))
            return out

    def all_chunks(self) -> list[tuple[str, str]]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id")
            return cur.fetchall()
```

（实现时注意 `chunks_by_entry` 的列序解包与 sparse 空值，以测试通过为准微调。）

- [ ] **Step 5: 跑测试确认通过**  Run: `EMBED_DIM=8 pytest tests/test_store.py -v --integration`  Expected: 1 passed
- [ ] **Step 6: Commit**  `git add src/lifeguide/store tests/test_store.py && git commit -m "feat(m1): pgvector 存储层（entries/chunks/稠密HNSW/稀疏ivfflat）"`

---

### Task 6: bge-m3 嵌入封装（稠密+稀疏+缓存）

**Files:**
- Create: `src/lifeguide/retrieval/embedder.py`、`tests/test_embedder.py`

**Interfaces:**
- Produces: `Embedder(model_name: str = "BAAI/bge-m3")`，方法：`embed(texts: list[str]) -> list[tuple[list[float], dict[int, float]]]`（每文本返回 `(dense, sparse)`）；`to_sparsevec_str(sparse: dict[int, float]) -> str`（pgvector 字符串 `{1:0.12,57:0.3}`）；类级磁盘缓存（key=sha256(text)+model_name，存 `embed_cache_dir` 下 jsonl）

- [ ] **Step 1: 写失败测试（不加载模型，mock 逻辑）**

```python
# tests/test_embedder.py
from lifeguide.retrieval.embedder import to_sparsevec_str, cache_key

def test_sparsevec_str_roundtrip():
    s = to_sparsevec_str({1: 0.12, 57: 0.3})
    assert s == "{1:0.12,57:0.3}"

def test_cache_key_stable():
    assert cache_key("abc", "m") == cache_key("abc", "m")
    assert cache_key("abc", "m") != cache_key("abd", "m")

def test_embedder_caching(tmp_path, monkeypatch):
    from lifeguide.retrieval.embedder import Embedder
    calls = []
    class FakeModel:
        def encode(self, texts, **kw):
            calls.append(list(texts))
            return type("R", (), {
                "dense_embeds": [[0.1, 0.2]] * len(texts),
                "lexical_weights": [{5: 0.7}] * len(texts),
            })()
    emb = Embedder.__new__(Embedder)  # 跳过模型加载
    emb.model, emb.model_name = FakeModel(), "fake"
    emb.cache_dir = tmp_path
    emb._load_cache()
    out1 = emb.embed(["你好"])
    out2 = emb.embed(["你好"])
    assert out1 == out2 == [([0.1, 0.2], {5: 0.7})]
    assert len(calls) == 1  # 第二次命中缓存
```

- [ ] **Step 2: 跑测试确认失败**  Run: `pytest tests/test_embedder.py -v`  Expected: FAIL
- [ ] **Step 3: 实现 embedder.py**

```python
# src/lifeguide/retrieval/embedder.py
import hashlib, json
from pathlib import Path

def to_sparsevec_str(sparse: dict[int, float]) -> str:
    items = ",".join(f"{i}:{w}" for i, w in sorted(sparse.items()))
    return "{" + items + "}"

def cache_key(text: str, model_name: str) -> str:
    return hashlib.sha256(f"{model_name}:{text}".encode()).hexdigest()

class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", cache_dir: Path | None = None):
        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from FlagEmbedding import BGEM3FlagModel
        self.model_name = model_name
        self.model = BGEM3FlagModel(model_name, use_fp16=False)  # CPU
        from lifeguide.config import settings_from_env
        self.cache_dir = cache_dir or settings_from_env().embed_cache_dir
        self._cache: dict[str, tuple[list[float], dict[int, float]]] = {}
        self._load_cache()

    def _load_cache(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / f"{self.model_name.replace('/', '_')}.jsonl"
        if self._cache_file.exists():
            for line in self._cache_file.read_text(encoding="utf-8").splitlines():
                if line:
                    rec = json.loads(line)
                    self._cache[rec["k"]] = (rec["d"], {int(i): w for i, w in rec["s"].items()})

    def _save_cache(self, key: str, value):
        with open(self._cache_file, "a", encoding="utf-8") as f:
            d, s = value
            f.write(json.dumps({"k": key, "d": d, "s": s}, ensure_ascii=False) + "\n")

    def embed(self, texts: list[str]) -> list[tuple[list[float], dict[int, float]]]:
        results: list[tuple[list[float], dict[int, float]] | None] = [None] * len(texts)
        todo, todo_idx = [], []
        for i, t in enumerate(texts):
            k = cache_key(t, self.model_name)
            if k in self._cache:
                results[i] = self._cache[k]
            else:
                todo.append(t); todo_idx.append(i)
        if todo:
            out = self.model.encode(todo, return_sparse=True, return_dense=True,
                                    return_colbert_vecs=False)
            for i, t, dense, sparse in zip(todo_idx, todo, out["dense_embeds"],
                                           out["lexical_weights"]):
                value = (dense.tolist(), {int(k): float(v) for k, v in sparse.items()})
                results[i] = value
                self._cache[cache_key(t, self.model_name)] = value
                self._save_cache(cache_key(t, self.model_name), value)
        return results
```

- [ ] **Step 4: 跑测试确认通过**  Run: `pytest tests/test_embedder.py -v`  Expected: 3 passed
- [ ] **Step 5:（集成冒烟，手动一次）**  Run: `python -c "from lifeguide.retrieval.embedder import Embedder; e=Embedder(); d,s=e.embed(['被裁了怎么办'])[0]; print(len(d), len(s))"`  Expected: `1024 <稀疏项数>`（首次运行下载模型 ~2.3GB）
- [ ] **Step 6: Commit**  `git add src/lifeguide/retrieval/embedder.py tests/test_embedder.py && git commit -m "feat(m1): bge-m3 嵌入封装（稠密+稀疏+磁盘缓存）"`

---

### Task 7: BM25 检索器（jieba 分词）

**Files:**
- Create: `src/lifeguide/retrieval/bm25.py`、`tests/test_bm25.py`

**Interfaces:**
- Produces: `tokenize(text: str) -> list[str]`（jieba lcut，去停用词与单字符标点）；`BM25Index`：`build(corpus: list[tuple[str, str]])`（`(chunk_id, text)`）、`search(query: str, k: int = 20) -> list[tuple[str, float]]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_bm25.py
from lifeguide.retrieval.bm25 import BM25Index, tokenize

def test_tokenize_drops_punct():
    toks = tokenize("被裁了，先算清 N！")
    assert "被裁" in "".join(toks) or any("裁" in t for t in toks)
    assert "，" not in toks and "！" not in toks

def test_bm25_ranks_exact_term_first():
    corpus = [("a", "经济补偿按 N 计算，每满一年一个月工资"),
              ("b", "烟雾报警器能降低火灾死亡概率"),
              ("c", "高血压要常年吃药控制")]
    idx = BM25Index(); idx.build(corpus)
    hits = idx.search("被裁 经济补偿 N", k=3)
    assert hits[0][0] == "a"
    hits2 = idx.search("烟雾报警器", k=3)
    assert hits2[0][0] == "b"
```

- [ ] **Step 2: 跑测试确认失败**  Run: `pytest tests/test_bm25.py -v`  Expected: FAIL
- [ ] **Step 3: 实现 bm25.py**

```python
# src/lifeguide/retrieval/bm25.py
import jieba
from rank_bm25 import BM25Okapi

STOP = set("，。！？；：「」『』（）《》、,.!?;:()[]{} \n\t·…—-的了吗呢吧啊和与及或在是对我有你他她它这那也要不没什么")

def tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip() and t not in STOP]

class BM25Index:
    def __init__(self):
        self._ids: list[str] = []
        self._bm25 = None

    def build(self, corpus: list[tuple[str, str]]):
        self._ids = [cid for cid, _ in corpus]
        tokenized = [tokenize(text) for _, text in corpus]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._ids[i], float(scores[i])) for i in order if scores[i] > 0]
```

（`BM25Index` 为进程内索引，`build()` 从 `store.all_chunks()` 全量构建，语料千级无性能问题。）

- [ ] **Step 4: 跑测试确认通过**  Run: `pytest tests/test_bm25.py -v`  Expected: 2 passed
- [ ] **Step 5: Commit**  `git add src/lifeguide/retrieval/bm25.py tests/test_bm25.py && git commit -m "feat(m1): jieba+BM25Okapi 检索器"`

---

### Task 8: RRF 融合与检索管道骨架

**Files:**
- Create: `src/lifeguide/retrieval/rrf.py`、`src/lifeguide/retrieval/pipeline.py`、`tests/test_rrf.py`

**Interfaces:**
- Consumes: Task 5 `EntryStore`、Task 6 `Embedder`、Task 7 `BM25Index`
- Produces: `rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]`（输入多路有序 chunk_id 列表，输出融合排序，**纯函数**）；`SearchConfig(use_bm25: bool = True, use_sparse: bool = True, use_rerank: bool = True, evidence_min: str | None = None, top_k: int = 5, recall_k: int = 20)`；`SearchPipeline(store, embedder, bm25_index, reranker=None)`，方法 `search(query: str, config: SearchConfig) -> SearchResult`；`SearchResult(chunk_hits: list[Hit], entries: list[Entry])`，`Hit(chunk_id: str, entry_id: str, score: float, text: str, evidence_level: str | None)`。消融四组 = SearchConfig 开关组合。

- [ ] **Step 1: 写失败测试（RRF 手算金值）**

```python
# tests/test_rrf.py
from lifeguide.retrieval.rrf import rrf

def test_rrf_basic_math():
    # 路1: [a, b]; 路2: [b, a]; k=60
    # a: 1/61 + 1/62; b: 1/62 + 1/61 → 并列，顺序稳定按 id
    out = rrf([["a", "b"], ["b", "a"]], k=60)
    scores = dict(out)
    assert abs(scores["a"] - (1/61 + 1/62)) < 1e-9
    assert abs(scores["b"] - (1/61 + 1/62)) < 1e-9

def test_rrf_missing_in_one_list():
    out = rrf([["a", "b", "c"], ["c"]], k=60)
    scores = dict(out)
    assert scores["c"] > scores["a"] > scores["b"]

def test_rrf_empty():
    assert rrf([[], []]) == []
```

- [ ] **Step 2: 跑测试确认失败**  Run: `pytest tests/test_rrf.py -v`  Expected: FAIL
- [ ] **Step 3: 实现 rrf.py**

```python
# src/lifeguide/retrieval/rrf.py

def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))
```

- [ ] **Step 4: 跑测试确认通过**  Run: `pytest tests/test_rrf.py -v`  Expected: 3 passed
- [ ] **Step 5: 实现管道骨架 pipeline.py**

```python
# src/lifeguide/retrieval/pipeline.py
from dataclasses import dataclass, field
from lifeguide.parser.models import Entry
from lifeguide.retrieval.rrf import rrf

@dataclass
class SearchConfig:
    use_bm25: bool = True
    use_sparse: bool = True
    use_rerank: bool = True          # Task 9 的 reranker 注入后生效
    evidence_min: str | None = None  # "A" / "B" / "C"（含以上）
    top_k: int = 5
    recall_k: int = 20

@dataclass
class Hit:
    chunk_id: str
    entry_id: str
    score: float
    text: str
    evidence_level: str | None = None
    benefit: str | None = None       # cost_tags.收益（大/中/小）
    cost_tag: str | None = None      # cost_tags.钱（0/少/中/大）

@dataclass
class SearchResult:
    hits: list[Hit] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)
    debug: dict = field(default_factory=dict)

class SearchPipeline:
    _EV_ORDER = {"A": 3, "B": 2, "C": 1, None: 0}

    def __init__(self, store, embedder, bm25_index, reranker=None):
        self.store, self.embedder, self.bm25 = store, embedder, bm25_index
        self.reranker = reranker

    def search(self, query: str, config: SearchConfig = SearchConfig()) -> SearchResult:
        dense_q, sparse_q = self.embedder.embed([query])[0]
        rankings: list[list[str]] = [ [c for c, _, _ in self.store.search_dense(dense_q, config.recall_k)] ]
        if config.use_sparse:
            rankings.append([c for c, _, _ in self.store.search_sparse(sparse_q, config.recall_k)])
        if config.use_bm25:
            rankings.append([c for c, _ in self.bm25.search(query, config.recall_k)])
        fused = rrf(rankings)
        candidates = fused[: max(config.recall_k, config.top_k * 4)]
        hits = self._to_hits(candidates)
        if config.evidence_min:
            floor = self._EV_ORDER[config.evidence_min]
            hits = [h for h in hits if self._EV_ORDER.get(h.evidence_level, 0) >= floor]
        if config.use_rerank and self.reranker:
            hits = self.reranker.rerank(query, hits)[: config.top_k]
        else:
            hits = hits[: config.top_k]
        return SearchResult(hits=hits, entries=[e for e in (self.store.get_entry(h.entry_id) for h in hits) if e],
                            debug={"rankings": rankings, "fused": fused[:20]})

    def _to_hits(self, fused: list[tuple[str, float]]) -> list[Hit]:
        out = []
        for chunk_id, score in fused:
            entry_id = chunk_id.split("#")[0]
            meta = self.store.chunk_meta(chunk_id) or {}
            text = self.store.chunk_text(chunk_id) or ""
            out.append(Hit(chunk_id, entry_id, score, text,
                          meta.get("evidence_level"), meta.get("benefit"), meta.get("cost_tag_money")))
        return out
```

（`store.chunk_meta / store.chunk_text` 已在 Task 5 的 store.py 中定义。）

- [ ] **Step 6: 跑全部单测**  Run: `pytest -v`  Expected: 全部 passed
- [ ] **Step 7: Commit**  `git add src/lifeguide/retrieval/rrf.py src/lifeguide/retrieval/pipeline.py tests/test_rrf.py src/lifeguide/store/store.py && git commit -m "feat(m1): RRF 纯函数 + SearchPipeline 骨架（消融开关）"`

---

### Task 9: Reranker 与元数据过滤排序

**Files:**
- Create: `src/lifeguide/retrieval/rerank.py`、`src/lifeguide/retrieval/filters.py`、`tests/test_rerank_filters.py`

**Interfaces:**
- Consumes: Task 8 的 `Hit`
- Produces: `Reranker(model_name: str = "BAAI/bge-reranker-v2-m3")`，`rerank(query: str, hits: list[Hit], top_k: int | None = None) -> list[Hit]`（按相关性分数重排并写回 `Hit.score`）；`sort_by_value(hits: list[Hit]) -> list[Hit]`（性价比排序：证据等级 A>B>C > 收益 大>中>小 > 成本 0<少<中<大——成本越小越前）；`filter_evidence(hits, minimum: str) -> list[Hit]`

- [ ] **Step 1: 写失败测试（reranker 用 mock，排序用真值）**

```python
# tests/test_rerank_filters.py
from lifeguide.retrieval.pipeline import Hit
from lifeguide.retrieval.rerank import Reranker
from lifeguide.retrieval.filters import sort_by_value, filter_evidence

def _hit(cid, score=0.0, ev=None, ben=None, cost=None):
    return Hit(chunk_id=cid, entry_id=cid.split("#")[0], score=score, text="t",
               evidence_level=ev, benefit=ben, cost_tag=cost)

def test_reranker_reorders_by_model_scores(monkeypatch):
    r = Reranker.__new__(Reranker)  # 免模型加载
    class FakeM:
        def compute_score(self, pairs, **kw):
            return [0.1, 0.9, 0.5]  # 第二个最相关
    r.model = FakeM()
    hits = [_hit("a"), _hit("b"), _hit("c")]
    out = r.rerank("q", hits)
    assert [h.chunk_id for h in out] == ["b", "c", "a"]
    assert out[0].score == 0.9

def test_sort_by_value_orders_evidence_benefit_cost():
    hits = [
        _hit("x1", ev="B", ben="大", cost="少"),
        _hit("x2", ev="A", ben="中", cost="中"),
        _hit("x3", ev="A", ben="大", cost="少"),
        _hit("x4", ev="C", ben="大", cost="0"),
    ]
    assert [h.chunk_id for h in sort_by_value(hits)] == ["x3", "x2", "x1", "x4"]

def test_filter_evidence_min_b():
    hits = [_hit("a", ev="A"), _hit("b", ev="B"), _hit("c", ev="C")]
    assert [h.chunk_id for h in filter_evidence(hits, "B")] == ["a", "b"]
```

（`Hit` 的 `benefit/cost_tag` 字段已在 Task 8 定义，由 `_to_hits` 从 metadata 填充：`metadata["benefit"]`、`metadata["cost_tag_money"]`。）

- [ ] **Step 2: 跑测试确认失败**  Run: `pytest tests/test_rerank_filters.py -v`  Expected: FAIL
- [ ] **Step 3: 实现 rerank.py / filters.py**

```python
# src/lifeguide/retrieval/rerank.py
class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from FlagEmbedding import FlagReranker
        self.model = FlagReranker(model_name, use_fp16=False)

    def rerank(self, query: str, hits, top_k: int | None = None):
        if not hits:
            return []
        pairs = [[query, h.text] for h in hits]
        scores = self.model.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        for h, s in zip(hits, scores):
            h.score = float(s)
        ordered = sorted(hits, key=lambda h: -h.score)
        return ordered[:top_k] if top_k else ordered
```

```python
# src/lifeguide/retrieval/filters.py
_EV = {"A": 3, "B": 2, "C": 1, None: 0}
_BEN = {"大": 3, "中": 2, "小": 1, None: 0}
_COST = {"0": 3, "少": 2, "中": 1, "大": 0, None: 1}  # 成本越小越优先

def filter_evidence(hits, minimum: str):
    floor = _EV[minimum]
    return [h for h in hits if _EV.get(h.evidence_level, 0) >= floor]

def sort_by_value(hits):
    return sorted(hits, key=lambda h: (-_EV.get(h.evidence_level, 0),
                                       -_BEN.get(h.benefit, 0),
                                       -_COST.get(h.cost_tag, 0)))
```

- [ ] **Step 4: 同步给 pipeline.Hit 补字段并在 `_to_hits` 填充；跑测试**  Run: `pytest -v`  Expected: 全部 passed
- [ ] **Step 5: Commit**  `git add src/lifeguide/retrieval/rerank.py src/lifeguide/retrieval/filters.py src/lifeguide/retrieval/pipeline.py tests/test_rerank_filters.py && git commit -m "feat(m1): bge-reranker 精排 + 证据/收益/成本元数据排序"`

---

### Task 10: 索引构建管线（父子 chunk + 入库）

**Files:**
- Create: `scripts/build_index.py`、`tests/test_pipeline.py`（integration，复用真模型太慢→用 8 维假向量端到端）

**Interfaces:**
- Consumes: Task 4 `entries.jsonl`/`docs_chunks.jsonl`、Task 5 `EntryStore`、Task 6 `Embedder`
- Produces: `make_child_chunks(entry: Entry) -> list[dict]`（字段级子块；单字段 >800 字符按句号切分拼接至 ≤800；chunk_id=`"{entry_id}#{field_name}-{i}"`；metadata 携带 `evidence_level/benefit/cost_tag_money/section_no/item_no/title`）；CLI `python scripts/build_index.py`（读 jsonl → chunk → embed → upsert，幂等可重跑，打印计数）

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 跑测试确认失败**  Run: `pytest tests/test_pipeline.py -v`  Expected: FAIL（ModuleNotFoundError: lifeguide.indexing）

- [ ] **Step 3: 实现 src/lifeguide/indexing.py + scripts/build_index.py**

```python
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
```

```python
# scripts/build_index.py
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifeguide.config import settings_from_env
from lifeguide.indexing import make_child_chunks
from lifeguide.parser.models import Entry
from lifeguide.store.store import EntryStore, ChildChunk
from lifeguide.retrieval.embedder import Embedder

def main():
    s = settings_from_env()
    store = EntryStore(s.database_url)
    store.init_schema(); store.reset()
    entries = [Entry.model_validate_json(l) for l in
               (s.data_dir / "entries.jsonl").read_text(encoding="utf-8").splitlines() if l]
    store.upsert_entries(entries)
    specs = []
    for e in entries:
        specs.extend(make_child_chunks(e))
    texts = [spec["text"] for spec in specs]
    emb = Embedder()
    vectors = emb.embed(texts)
    chunks = [ChildChunk(spec["chunk_id"], spec["entry_id"], spec["field_name"],
                         spec["text"], dense, sparse, spec["metadata"])
              for spec, (dense, sparse) in zip(specs, vectors)]
    store.upsert_chunks(chunks)
    print(f"entries={len(entries)} chunks={len(chunks)} 完成（幂等，可重跑）")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试 + 真实构建**  Run: `pytest tests/test_pipeline.py -v && python scripts/build_index.py`  Expected: 测试 2 passed；CLI 首次跑约 10-30 分钟（687 条 CPU 编码），打印 entries=687 chunks≈3000+
- [ ] **Step 5: Commit**  `git add src/lifeguide/indexing.py scripts/build_index.py tests/test_pipeline.py && git commit -m "feat(m1): 索引构建管线（字段级父子chunk+嵌入+入库）"`

---

### Task 11: 端到端检索 CLI（人工体验入口）

**Files:**
- Create: `scripts/search_cli.py`

**Interfaces:**
- Consumes: Task 8/9 `SearchPipeline`、`Reranker`、`sort_by_value`
- Produces: `python scripts/search_cli.py "被裁了怎么算补偿" [--ev A] [--no-bm25] [--no-sparse] [--no-rerank]`，rich 表格输出：排名/条目标题/节-条号/证据徽章/分数/命中字段文本前 80 字

- [ ] **Step 1: 实现 CLI**

```python
# scripts/search_cli.py
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifeguide.config import settings_from_env
from lifeguide.store.store import EntryStore
from lifeguide.retrieval.embedder import Embedder
from lifeguide.retrieval.bm25 import BM25Index
from lifeguide.retrieval.pipeline import SearchPipeline, SearchConfig
from lifeguide.retrieval.rerank import Reranker
from lifeguide.retrieval.filters import sort_by_value
from rich.console import Console
from rich.table import Table

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--ev", choices=["A", "B", "C"], default=None)
    ap.add_argument("--no-bm25", action="store_true")
    ap.add_argument("--no-sparse", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    s = settings_from_env()
    store = EntryStore(s.database_url)
    emb = Embedder()
    bm25 = BM25Index(); bm25.build(store.all_chunks())
    pipe = SearchPipeline(store, emb, bm25, reranker=Reranker())
    cfg = SearchConfig(use_bm25=not args.no_bm25, use_sparse=not args.no_sparse,
                       use_rerank=not args.no_rerank, evidence_min=args.ev, top_k=args.topk)
    result = pipe.search(args.query, cfg)
    title_of = {e.entry_id: e.title for e in result.entries}
    table = Table(title=f"查询: {args.query}")
    for col in ("#", "条目", "节-条", "证据", "分数", "命中文本"):
        table.add_column(col, overflow="fold")
    for i, h in enumerate(sort_by_value(result.hits), 1):
        table.add_row(str(i), title_of.get(h.entry_id, "?"), h.entry_id,
                      h.evidence_level or "-", f"{h.score:.4f}", h.text[:80])
    Console().print(table)

if __name__ == "__main__":
    main()
```

（表格"条目"列用 `result.entries` 构建的 entry_id→title 映射填充。）

- [ ] **Step 2: 手工验收**

Run: `python scripts/search_cli.py "被裁了公司不给补偿怎么办"` 和 `python scripts/search_cli.py "替朋友担保签不签"` 和 `python scripts/search_cli.py "怎么给家里装报警器" --ev A`
Expected: 三条查询的 top5 都应命中直觉正确条目（19 节相关 / 08 节担保相关 / 01 节第 3 条烟雾报警器）。记录命中情况到 `docs/m1-manual-checks.md`。

- [ ] **Step 3: Commit**  `git add scripts/search_cli.py docs/m1-manual-checks.md && git commit -m "feat(m1): 检索 CLI（消融开关+证据过滤+rich 输出）"`

---

### Task 12: 评测指标（Recall@k / MRR / nDCG@k）

**Files:**
- Create: `src/lifeguide/eval/metrics.py`、`tests/test_metrics.py`

**Interfaces:**
- Produces: `recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float`（retrieved 为 entry_id 有序列表）；`mrr(retrieved: list[str], relevant: set[str]) -> float`；`ndcg_at_k(retrieved: list[str], relevant: set[str], k: int = 10) -> float`（二值相关，DCG=Σ1/log2(i+1)）

- [ ] **Step 1: 写失败测试（手算金值）**

```python
# tests/test_metrics.py
import math
from lifeguide.eval.metrics import recall_at_k, mrr, ndcg_at_k

def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "x"}, 5) == 0.5
    assert recall_at_k(["a", "b", "c"], {"a"}, 1) == 1.0
    assert recall_at_k([], {"a"}, 5) == 0.0

def test_mrr():
    assert mrr(["x", "y", "a"], {"a"}) == 1 / 3
    assert mrr(["a", "b"], {"a"}) == 1.0
    assert mrr(["x"], {"a"}) == 0.0

def test_ndcg_hand_computed():
    retrieved = ["x", "a", "b", "c"]
    relevant = {"a", "c"}
    dcg = 0 / math.log2(2) + 1 / math.log2(3) + 0 / math.log2(4) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert abs(ndcg_at_k(retrieved, relevant, 10) - dcg / idcg) < 1e-9
```

- [ ] **Step 2: 跑测试确认失败**  Run: `pytest tests/test_metrics.py -v`  Expected: FAIL
- [ ] **Step 3: 实现 metrics.py**

```python
# src/lifeguide/eval/metrics.py
import math

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)

def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, item in enumerate(retrieved[:k], start=1)
              if item in relevant)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0
```

- [ ] **Step 4: 跑测试确认通过**  Run: `pytest tests/test_metrics.py -v`  Expected: 3 passed
- [ ] **Step 5: Commit**  `git add src/lifeguide/eval/metrics.py tests/test_metrics.py && git commit -m "feat(m1): 检索评测指标 recall/mrr/ndcg"`

---

### Task 13: Golden Set 构造（LLM 反向生成 + 人工校验）

**Files:**
- Create: `src/lifeguide/eval/golden_set.py`、`scripts/gen_golden.py`、`data/golden_set.jsonl`（产物）、`data/golden_review.csv`（人工校验用）

**Interfaces:**
- Consumes: `data/entries.jsonl`、LLM API（`LLM_BASE_URL/LLM_API_KEY/LLM_MODEL`）
- Produces: golden 行格式 `{"qid": "g001", "query": str, "relevant": ["19-4", ...], "qtype": "colloquial|multi_hop|zero_hit", "source_entries": [...]}`；函数 `gen_questions_for_entry(entry: Entry, client, n: int = 2) -> list[str]`；`build_golden(entries, docs) -> list[dict]`（抽样规则见步骤 3）；`load_golden(path) -> list[dict]`

**构成（目标 ~200）**：colloquial 140（70 个条目 ×2 问，分层抽样按章，每章至少 1 条目）+ multi_hop 40（20 对跨章相关条目 ×2 问，对子人工指定见步骤 3）+ zero_hit 20（人工写死，见步骤 3）。

- [ ] **Step 1: 写失败测试（纯逻辑：抽样覆盖与格式）**

```python
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
```

- [ ] **Step 2: 跑测试确认失败**  Run: `pytest tests/test_golden_set.py -v`  Expected: FAIL
- [ ] **Step 3: 实现 golden_set.py（含写死数据）**

```python
# src/lifeguide/eval/golden_set.py
import json, random
from pathlib import Path
from lifeguide.config import settings_from_env
from lifeguide.parser.models import Entry

ZERO_HIT_QUERIES = [  # 书中明确无对应条目的主题，人工写死
    "怎么写简历找工作", "怎么选驾校考驾照", "买什么车划算", "怎么考公务员怎么准备行测",
    "家里装修什么风格好看", "怎么练出马甲线", "养猫要注意什么", "怎么学英语口语",
    "拍婚纱照选哪家", "怎么给孩子起名字", "哪个手机牌子好用", "冬天暖气不热找谁",
    "怎么注册自媒体账号涨粉", "健身房私教课值不值得买", "看什么书提升情商",
    "无人机在哪里能飞", "怎么办理港澳通行证旅游签注", "WiFi 路由器怎么摆放信号好",
    "二手车过户流程", "怎么举报噪音扰民",
]

MULTI_HOP_PAIRS = [  # (entry_a, entry_b) 跨章相关对，人工指定
    ("19-4", "07-2"),   # 被裁补偿 + 被欠薪投诉
    ("16-1", "24-1"),   # 慢性病 + 看病
    ("10-3", "08-19"),  # 结婚 + 法律红线
    ("13-1", "01-3"),   # 紧急 + 烟雾报警器
    ("15-2", "09-2"),   # 租房 + 法律
    ("27-1", "20-1"),   # 怀孕 + 新生儿
    ("05-10", "12-1"),  # 省钱 + 创业
    ("11-3", "14-1"),   # 程序员红线 + 账号安全
    ("17-2", "29-1"),   # 老人 + 重大打击
    ("31-2", "32-1"),   # 十八岁出路 + 留学
    ("33-1", "07-1"),   # 残疾 + 没钱
    ("34-2", "02-1"),   # 常备药 + 慢慢死
    ("18-1", "30-2"),   # 养孩子 + 上学孩子
    ("21-3", "08-10"),  # 出国 + 法律
    ("06-2", "05-1"),   # 反面清单 + 浪费钱
    ("22-1", "03-1"),   # 放松 + 精力
    ("26-1", "12-2"),   # 做网站 + 创业
    ("28-1", "01-7"),   # 外形 + 早死
    ("25-1", "19-8"),   # 身后事 + 离职
    ("23-1", "04-1"),   # 学技能 + 时间
]

PROMPT = """你将看到《高性价比人生指南》中的一条建议。请生成 {n} 个普通用户会问的口语化问题，
要求：1) 不出现标题里的原词；2) 描述具体处境而非抽象概念；3) 中文。
条目标题：{title}
说人话：{plain}
输出 JSON：{{"questions": ["...", "..."]}}"""

def stratified_sample(entries: list[Entry], per_section: int, total: int) -> list[Entry]:
    by_sec: dict[int, list[Entry]] = {}
    for e in entries:
        by_sec.setdefault(e.section_no or 0, []).append(e)
    picked: list[Entry] = []
    for sec in sorted(by_sec):
        random.Random(42).shuffle(by_sec[sec])
        picked.extend(by_sec[sec][:per_section])
    return picked[:total]

def gen_questions_for_entry(entry: Entry, client, n: int = 2) -> list[str]:
    resp = client.chat.completions.create(
        model=settings_from_env().llm_model,
        messages=[{"role": "user", "content": PROMPT.format(n=n, title=entry.title,
                                                            plain=(entry.plain or "")[:500])}],
        temperature=0.7)
    try:
        data = json.loads(resp.choices[0].message.content)
        return data["questions"][:n]
    except Exception:
        return []

def build_golden(entries: list[Entry], docs, client) -> list[dict]:
    by_id = {e.entry_id: e for e in entries}
    golden: list[dict] = []
    qid = 0
    for e in stratified_sample(entries, per_section=2, total=70):
        for q in gen_questions_for_entry(e, client, n=2):
            qid += 1
            golden.append({"qid": f"g{qid:03d}", "query": q, "relevant": [e.entry_id],
                           "qtype": "colloquial", "source_entries": [e.entry_id]})
    for a, b in MULTI_HOP_PAIRS:
        ea, eb = by_id.get(a), by_id.get(b)
        if not (ea and eb):
            continue
        for q in gen_questions_for_entry(ea, client, n=2):
            qid += 1
            golden.append({"qid": f"g{qid:03d}", "query": q, "relevant": [a, b],
                           "qtype": "multi_hop", "source_entries": [a, b]})
    for q in ZERO_HIT_QUERIES:
        qid += 1
        golden.append({"qid": f"g{qid:03d}", "query": q, "relevant": [],
                       "qtype": "zero_hit", "source_entries": []})
    return golden

def load_golden(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
```

（`MULTI_HOP_PAIRS` 条号需在生成后与真实条目对账：写 `scripts/gen_golden.py` 时先校验所有 id 存在，缺失的打印 warning 并跳过——以真实数据微调为准。）

- [ ] **Step 4: 跑测试确认通过 + 生成**

Run: `pytest tests/test_golden_set.py -v && python scripts/gen_golden.py`
（gen_golden.py：读 entries.jsonl → build_golden → 写 `data/golden_set.jsonl` + `data/golden_review.csv`（qid/query/relevant 标题/留空 verdict 列），打印各类计数。）
Expected: 测试 passed；生成 ~140 colloquial + ~40 multi_hop + 20 zero_hit ≈ 200 条。

- [ ] **Step 5: 人工校验**：打开 `data/golden_review.csv` 逐条标注 keep/drop（预计 30-45 分钟），drop 的从 jsonl 删除。此步必须人工完成，不派 subagent。
- [ ] **Step 6: Commit**  `git add src/lifeguide/eval/golden_set.py scripts/gen_golden.py tests/test_golden_set.py data/golden_set.jsonl data/golden_review.csv && git commit -m "feat(m1): golden set 三类覆盖（LLM反向生成+人工校验）"`

---

### Task 14: 四组消融实验与报告

**Files:**
- Create: `src/lifeguide/eval/ablation.py`、`scripts/run_ablation.py`、`docs/m1-ablation-report.md`（产物）

**Interfaces:**
- Consumes: Task 8 `SearchPipeline/SearchConfig`、Task 11 的组装方式、Task 12 指标、Task 13 golden set
- Produces: `run_ablation(pipeline, golden) -> dict[str, dict[str, float]]`；报告 markdown：四组 × 三指标表格 + zero_hit 拒答率 + 每组逐查询明细 csv（`docs/m1-ablation-details.csv`）

**四组配置**：`A_dense`：`use_bm25=False, use_sparse=False, use_rerank=False`；`B_dense_bm25`：`use_sparse=False, use_rerank=False`；`C_3way_rrf`：`use_rerank=False`；`D_3way_rerank`：全开（默认）。

- [ ] **Step 1: 实现 ablation.py**

```python
# src/lifeguide/eval/ablation.py
import dataclasses
from lifeguide.retrieval.pipeline import SearchConfig
from lifeguide.eval.metrics import recall_at_k, mrr, ndcg_at_k

CONFIGS = {
    "A_dense":        SearchConfig(use_bm25=False, use_sparse=False, use_rerank=False),
    "B_dense_bm25":   SearchConfig(use_bm25=True,  use_sparse=False, use_rerank=False),
    "C_3way_rrf":     SearchConfig(use_bm25=True,  use_sparse=True,  use_rerank=False),
    "D_3way_rerank":  SearchConfig(use_bm25=True,  use_sparse=True,  use_rerank=True),
}

def run_ablation(pipeline, golden: list[dict]) -> dict[str, dict[str, float]]:
    results = {}
    for name, cfg in CONFIGS.items():
        eval_cfg = dataclasses.replace(cfg, top_k=10, recall_k=20)
        rec5, mrr_v, ndcg, n = 0.0, 0.0, 0.0, 0
        for item in golden:
            if item["qtype"] == "zero_hit":
                continue
            retrieved = [h.entry_id for h in pipeline.search(item["query"], eval_cfg).hits]
            relevant = set(item["relevant"])
            rec5 += recall_at_k(retrieved, relevant, 5)
            mrr_v += mrr(retrieved, relevant)
            ndcg += ndcg_at_k(retrieved, relevant, 10)
            n += 1
        results[name] = {"recall@5": rec5 / n, "mrr": mrr_v / n, "ndcg@10": ndcg / n, "n": n}
    return results

def zero_hit_rejection(pipeline, golden, score_floor: float = 0.3) -> float:
    ok, n = 0, 0
    for item in golden:
        if item["qtype"] != "zero_hit":
            continue
        hits = pipeline.search(item["query"], SearchConfig(top_k=1)).hits
        n += 1
        if not hits or hits[0].score < score_floor:
            ok += 1
    return ok / n if n else 0.0
```

- [ ] **Step 2: 实现 scripts/run_ablation.py**（组装 store/embedder/bm25/reranker/pipeline → run_ablation + zero_hit_rejection → 写 markdown 报告含表格与逐查询 csv，组装代码复制 Task 11 CLI 的初始化段）

- [ ] **Step 3: 运行并产出报告**

Run: `python scripts/run_ablation.py`
Expected: `docs/m1-ablation-report.md` 含四组指标表（**这就是简历数字的来源**：B/A、C/B、D/C 的差值即各层贡献）+ zero_hit 拒答率。

- [ ] **Step 4: 数字合理性检查**：若 D 组 recall@5 < 0.75，按「错误案例 → 归因（改写/分词/切分）→ 针对性修 → 重跑」循环，最多 2 轮，修不动的问题记录进报告"已知局限"。此步人工判断，不派 subagent。
- [ ] **Step 5: Commit**  `git add src/lifeguide/eval/ablation.py scripts/run_ablation.py docs/m1-ablation-report.md docs/m1-ablation-details.csv && git commit -m "feat(m1): 四组消融评测报告（简历量化数字来源）"`

---

### Task 15: CI 集成

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: 全部测试；语料在 repo 内（HowToLiveBetter/ 已在 gitignore——CI 需要：改用 actions/checkout 到子目录或 gitignore 保留语料。**决策：语料以 `git clone --depth 1` 在 CI 内拉取**，与 spec「语料只读」一致）

- [ ] **Step 1: 写 workflow**

```yaml
# .github/workflows/ci.yml
name: ci
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - name: fetch corpus
        run: git clone --depth 1 https://github.com/eternity4719/HowToLiveBetter.git
      - name: install
        run: pip install -e ".[dev]"
      - name: unit tests
        run: pytest -v
      - name: parse reconciliation (687)
        run: python scripts/parse_all.py
```

（integration 测试与消融不在 CI：需 PG/模型/大下载。M3 再加 self-hosted runner 方案。）

- [ ] **Step 2: 推送验证**  Run: `git push 后看 Actions`  Expected: CI 绿（若暂无远端，`act` 本地跑或留待推 GitHub 时验证，在 commit message 注明）
- [ ] **Step 3: Commit**  `git add .github/workflows/ci.yml && git commit -m "ci(m1): 单测+687解析对账进 GitHub Actions"`

---

## 交付验收（M1 完成定义）

> **Scope 注记（M1 收口时裁定）**：Goal 行中的「docs 长文」部分推迟至 M2——M1 的索引、golden set 与全部评测数字均基于 book 687 条目（数据自洽）；docs 长文（94 块，如「被裁了之后先做什么」聚合清单）的真正消费者是 M2 自查清单场景的批量主题检索，且其 chunk_id 无 `#` 分隔符，需在 M2 连同索引方案与 entry_id 解析一起设计后再入索引。`data/docs_chunks.jsonl` 产物保留备用。

1. `pytest -v` 全绿（单测不依赖网络/模型/PG）；`pytest -v --integration`（PG 起）全绿
2. `python scripts/parse_all.py` 输出 `book 条目: 687（期望 687）`
3. `python scripts/build_index.py` 完成，chunks ≥ 3000
4. `python scripts/search_cli.py` 三条人工查询命中合理（记录于 docs/m1-manual-checks.md）
5. `docs/m1-ablation-report.md` 有四组完整指标 + zero_hit 拒答率
6. CI 绿
