"""P7 T4：阶段③——把 d2_gpu 产出的英文转录(transcript.json)逐段译成
口语化中文配音稿(zh_script.json)，保留原有段数与时间轴，供下游 d4_tts/d5_subs
按句起点配音、烧字幕。

关键坑（已踩过，别重蹈）：src.util.llmjson.extract_json 只从第一个 `{` 找到
最后一个 `}` 解析——它解析的是一个 JSON **对象**，不是顶层数组。所以绝不能让
LLM 直接回一个 `[...]` 数组，必须让它包一层对象，如 `{"items": [...]}`，
本文件的 SYSTEM/build_prompt 与 _translate_batch 都遵循这一点。
"""
from __future__ import annotations

import json

from src.context import Ctx
from src.util.jobs import read_json, write_json
from src.util.llmjson import extract_json

OUTPUT = "zh_script.json"
INPUT = "transcript.json"
CHUNK_SIZE = 80  # 每批喂给 LLM 的段数上限，避免长纪录片一次性撑爆输出/被截断

SYSTEM = (
    "你是纪录片字幕翻译，把英文解说逐句译成口语化、流畅、适合配音朗读的中文。"
    "自然口语，不要生硬直译，不要加解释或备注，不要合并或拆分句子。"
    "只输出一个 JSON 对象，不要额外文字、不要 markdown 代码块。"
)


def build_prompt(items: list[dict]) -> str:
    payload = json.dumps({"items": items}, ensure_ascii=False)
    return (
        "把下面每条英文解说译成中文配音稿，输出一个 JSON 对象（不是数组），"
        '格式必须是 {"items": [{"id": 整数, "zh": "中文翻译"}, ...]}，'
        "id 必须和输入的 id 一一对应，不要遗漏、不要新增、不要打乱顺序。\n"
        f"输入：{payload}"
    )


def _translate_batch(ctx: Ctx, batch: list[tuple[int, str]]) -> dict[int, str]:
    items = [{"id": i, "en": text} for i, text in batch]
    raw = ctx.llm.generate(SYSTEM, build_prompt(items))
    parsed = extract_json(raw)
    zh_map: dict[int, str] = {}
    for item in parsed.get("items", []):
        try:
            idx = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        zh_map[idx] = item.get("zh", "")
    return zh_map


def run(ctx: Ctx) -> None:
    data = read_json(ctx.job, INPUT)
    segments = data.get("segments", [])
    duration = data.get("duration", 0)

    if not segments:
        write_json(ctx.job, OUTPUT, {"duration": duration, "segments": []})
        ctx.warn(f"{ctx.slug}: d3_translate 转录 0 段，跳过翻译")
        return

    indexed = list(enumerate(segments))
    zh_map: dict[int, str] = {}
    for start in range(0, len(indexed), CHUNK_SIZE):
        chunk = indexed[start : start + CHUNK_SIZE]
        batch = [(i, seg.get("text", "")) for i, seg in chunk]
        zh_map.update(_translate_batch(ctx, batch))

    out_segments = []
    missing = 0
    for i, seg in indexed:
        zh = zh_map.get(i)
        if zh is None:
            zh = seg.get("text", "")  # 回退：保留英文原文，不留空、不错位、不崩
            missing += 1
        out_segments.append(
            {
                "id": i,
                "start": seg.get("start"),
                "end": seg.get("end"),
                "en": seg.get("text", ""),
                "zh": zh,
            }
        )

    if missing:
        ctx.warn(
            f"{ctx.slug}: d3_translate 有 {missing}/{len(segments)} 段缺少翻译"
            "（LLM 返回条数不足），已回退用英文原文顶上"
        )

    write_json(ctx.job, OUTPUT, {"duration": duration, "segments": out_segments})
