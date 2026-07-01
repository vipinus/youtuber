from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from src.config import Config, load_config
from src.context import Ctx
from src.errors import StageBlocked
from src.llm import LLMProvider, get_llm
from src.stages import s1_topic, s2_research, s3_script, s4_tts, s5_subs, s6_media, s7_assemble
from src.util.jobs import exists

STAGES: list[tuple[str, Callable[[Ctx], None], str]] = [
    ("s1_topic", s1_topic.run, s1_topic.OUTPUT),
    ("s2_research", s2_research.run, s2_research.OUTPUT),
    ("s3_script", s3_script.run, s3_script.OUTPUT),
    ("s4_tts", s4_tts.run, s4_tts.OUTPUT),
    ("s5_subs", s5_subs.run, s5_subs.OUTPUT),
    ("s6_media", s6_media.run, s6_media.OUTPUT),
    ("s7_assemble", s7_assemble.run, s7_assemble.OUTPUT),
]


def run_pipeline(
    slug: str,
    config: Config,
    root: Path,
    llm: LLMProvider,
    frm: str | None = None,
    to: str | None = None,
    force: bool = False,
) -> None:
    names = [s[0] for s in STAGES]
    i0 = names.index(frm) if frm else 0
    i1 = names.index(to) if to else len(STAGES) - 1
    ctx = Ctx(slug=slug, config=config, root=root, llm=llm)
    for name, fn, output in STAGES[i0 : i1 + 1]:
        if not force and exists(ctx.job, output):
            print(f"[skip] {name} ({output} 已存在)")
            continue
        print(f"[run ] {name}")
        try:
            fn(ctx)
        except StageBlocked as e:
            print(f"[blocked] {name}: {e}")
            break
        print(f"[done] {name} -> {output}")
    if ctx.warnings:
        print(f"[warnings] 本次共 {len(ctx.warnings)} 条")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="探索飞碟与外星人 出片管线")
    ap.add_argument("--case", required=True, help="种子案例 slug")
    ap.add_argument("--from", dest="frm", default=None, help="起始阶段名")
    ap.add_argument("--to", default=None, help="结束阶段名")
    ap.add_argument("--force", action="store_true", help="忽略已存在产物，重跑")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    root = Path(config.get("paths.jobs_root", "jobs"))
    llm = get_llm(config)
    run_pipeline(args.case, config, root, llm, args.frm, args.to, args.force)


if __name__ == "__main__":
    main()
