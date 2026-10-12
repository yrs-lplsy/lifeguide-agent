# src/lifeguide/eval/golden_set.py
import json, random, time
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

def _extract_json(text: str):
    """模型可能把 JSON 包在 markdown 围栏里，先剥掉再 json.loads。"""
    t = text.strip()
    if t.startswith("```"):
        # 去掉 ```json ... ``` 围栏
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return json.loads(t)

def stratified_sample(entries: list[Entry], per_section: int, total: int) -> list[Entry]:
    by_sec: dict[int, list[Entry]] = {}
    for e in entries:
        by_sec.setdefault(e.section_no or 0, []).append(e)
    picked: list[Entry] = []
    for sec in sorted(by_sec):
        random.Random(42).shuffle(by_sec[sec])
        picked.extend(by_sec[sec][:per_section])
    return picked[:total]

def gen_questions_for_entry(entry: Entry, client, n: int = 2,
                            model: str | None = None) -> list[str]:
    model = model or settings_from_env().llm_model
    for _attempt in range(2):  # 单次瞬时失败重试一次
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT.format(
                n=n, title=entry.title, plain=(entry.plain or "")[:500])}],
            temperature=0.7)
        try:
            data = _extract_json(resp.choices[0].message.content)
            return [q for q in data["questions"] if isinstance(q, str)][:n]
        except Exception:
            continue
    return []

def build_golden(entries: list[Entry], docs, client, model: str | None = None,
                 pairs: list[tuple[str, str]] | None = None) -> list[dict]:
    by_id = {e.entry_id: e for e in entries}
    golden: list[dict] = []
    qid = 0
    for e in stratified_sample(entries, per_section=2, total=70):
        for q in gen_questions_for_entry(e, client, n=2, model=model):
            qid += 1
            golden.append({"qid": f"g{qid:03d}", "query": q, "relevant": [e.entry_id],
                           "qtype": "colloquial", "source_entries": [e.entry_id]})
        time.sleep(0.3)  # 对 LLM API 礼貌限速
    for a, b in (pairs if pairs is not None else MULTI_HOP_PAIRS):
        ea, eb = by_id.get(a), by_id.get(b)
        if not (ea and eb):
            continue
        for q in gen_questions_for_entry(ea, client, n=2, model=model):
            qid += 1
            golden.append({"qid": f"g{qid:03d}", "query": q, "relevant": [a, b],
                           "qtype": "multi_hop", "source_entries": [a, b]})
        time.sleep(0.3)
    for q in ZERO_HIT_QUERIES:
        qid += 1
        golden.append({"qid": f"g{qid:03d}", "query": q, "relevant": [],
                       "qtype": "zero_hit", "source_entries": []})
    return golden

def load_golden(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
