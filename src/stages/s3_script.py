from __future__ import annotations

import json

from src.context import Ctx
from src.errors import StageBlocked
from src.util.jobs import read_json, write_json
from src.util.llmjson import extract_json

OUTPUT = "script.json"
INPUT = "research.json"
CPM = 300  # 中文旁白约 300 字/分钟

SYSTEM = (
    "你是「探索飞碟与外星人」频道的中文脚本作者，写 8–20 分钟的悬疑解说。"
    "严格区分证据等级：documented 当事实讲；claim 用「目击者称/据称」；"
    "speculation 用「有推测认为」。只输出一个 JSON 对象，不要额外文字。"
)


def build_prompt(research: dict) -> str:
    return (
        "根据以下核查过的事实写一篇原创中文悬疑解说脚本，输出 JSON，字段："
        "title_options(3个中文标题字符串数组)，"
        "segments(数组，每段 {id(整数), text(中文旁白), visual_query(英文关键词), est_sec(整数秒)})。"
        "结构：钩子→背景→事件经过→疑点→各方解释→开放式收尾+订阅引导。\n"
        "visual_query 规则：2–4 个英文关键词的名词短语，必须是图库能搜到的具体实物/地点/场景"
        "（好例：'desert New Mexico'、'weather balloon'、'night sky stars'、'military aircraft 1947'）；"
        "禁止写分镜、运镜、剪辑或剧情描述（坏例：'Dramatic intro with flashing headlines, quick cut to...'）。\n"
        f"事实：{json.dumps(research, ensure_ascii=False)}"
    )


def run(ctx: Ctx) -> None:
    research = read_json(ctx.job, INPUT)
    if research.get("blocked"):
        raise StageBlocked(f"{ctx.slug}: research.blocked=true，不生成脚本")
    script = extract_json(ctx.llm.generate(SYSTEM, build_prompt(research)))
    segments = script.get("segments", [])
    total_words = sum(len(s.get("text", "")) for s in segments)
    script["total_words"] = total_words
    script["est_minutes"] = round(total_words / CPM, 1)
    write_json(ctx.job, OUTPUT, script)
