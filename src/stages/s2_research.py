from __future__ import annotations

import json

from src.context import Ctx
from src.errors import StageBlocked
from src.util.jobs import read_json, write_json
from src.util.llmjson import extract_json

OUTPUT = "research.json"
INPUT = "topic.json"
LEVELS = ("documented", "claim", "speculation")

SYSTEM = (
    "你是「探索飞碟与外星人」频道的事实核查研究员。针对给定 UFO 案例列出关键事实，"
    "并为每条标注证据等级：documented(有档案/记录) / claim(目击或证词) / speculation(推测)。"
    "只输出一个 JSON 对象，不要额外文字。"
)


def build_prompt(topic: dict) -> str:
    return (
        "为下面的案例做事实核查，输出 JSON，字段 facts：数组，每个元素 "
        "{claim(中文,该说法), evidence(documented|claim|speculation), source(来源,中文)}。"
        "尽量给出可查证的 documented 事实。\n"
        f"案例：{json.dumps(topic, ensure_ascii=False)}"
    )


def summarize(facts: list[dict]) -> dict:
    out = {lv: 0 for lv in LEVELS}
    for f in facts:
        lv = f.get("evidence")
        if lv in out:
            out[lv] += 1
    return out


def run(ctx: Ctx) -> None:
    topic = read_json(ctx.job, INPUT)
    research = extract_json(ctx.llm.generate(SYSTEM, build_prompt(topic)))
    facts = research.get("facts", [])
    if not isinstance(facts, list):        # 畸形 LLM 输出 → 视为空 → fail closed
        facts = []
    research["facts"] = facts
    summary = summarize(facts)
    research["evidence_summary"] = summary
    documented = summary["documented"]
    min_doc = ctx.config.get("guardrails.min_documented_facts", 3)
    blocked = documented == 0
    research["blocked"] = blocked
    if not blocked and documented < min_doc:
        ctx.warn(f"{ctx.slug}: documented 事实 {documented} 条 < 阈值 {min_doc}")
        warns = research.get("warnings")
        if not isinstance(warns, list):
            warns = []
            research["warnings"] = warns
        warns.append(f"documented={documented} < {min_doc}")
    write_json(ctx.job, OUTPUT, research)
    if blocked:
        raise StageBlocked(f"{ctx.slug}: 无 documented 事实，证据不足，停止本案例")
