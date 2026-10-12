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
