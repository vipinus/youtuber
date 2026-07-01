from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.context import Ctx
from src.util.jobs import write_json
from src.util.llmjson import extract_json

OUTPUT = "topic.json"

SYSTEM = (
    "你是中文 UFO/未解之谜频道「探索飞碟与外星人」的资料研究员。"
    "只输出一个 JSON 对象，不要任何额外解释或 Markdown 围栏外的文字。"
)


def load_seed(slug: str, seeds_path: str | Path) -> dict:
    cases = yaml.safe_load(Path(seeds_path).read_text(encoding="utf-8")) or []
    for c in cases:
        if c.get("slug") == slug:
            return c
    raise KeyError(f"种子案例未找到: {slug}（in {seeds_path}）")


def build_prompt(seed: dict) -> str:
    return (
        "根据以下案例线索，补全并标准化一个选题元信息对象。\n"
        f"案例线索：{json.dumps(seed, ensure_ascii=False)}\n\n"
        "输出 JSON，字段：title_working(中文标题), year(整数), location(中文), "
        "hook(一句话钩子,中文), known_controversies(中文字符串数组), "
        "seed_sources(字符串数组)。"
    )


def run(ctx: Ctx) -> None:
    seeds_path = ctx.config.get("paths.seeds", "seeds/cases.yaml")
    seed = load_seed(ctx.slug, seeds_path)
    raw = ctx.llm.generate(SYSTEM, build_prompt(seed))
    topic = extract_json(raw)
    topic.setdefault("slug", ctx.slug)
    write_json(ctx.job, OUTPUT, topic)
