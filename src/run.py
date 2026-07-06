from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from src.config import Config, load_config
from src.context import Ctx
from src.errors import StageBlocked
from src.llm import LLMProvider, get_llm
from src.stages import (
    d1_source,
    d2_gpu,
    d3_translate,
    d4_tts,
    d5_subs,
    d7_assemble,
    d8_meta,
    d9b_bilibili,
    d9c_publish_en,
    d9d_crosslink,
    d10_archive,
    s1_topic,
    s2_research,
    s3_script,
    s4_tts,
    s5_subs,
    s6_media,
    s7_assemble,
    s8_meta,
    s9_publish,
)
from src.util.jobs import exists

# 原始 s1-s9「选题→出片→发布」管线（保持不动，别破坏既有行为/测试）。
STAGES: list[tuple[str, Callable[[Ctx], None], str]] = [
    ("s1_topic", s1_topic.run, s1_topic.OUTPUT),
    ("s2_research", s2_research.run, s2_research.OUTPUT),
    ("s3_script", s3_script.run, s3_script.OUTPUT),
    ("s4_tts", s4_tts.run, s4_tts.OUTPUT),
    ("s5_subs", s5_subs.run, s5_subs.OUTPUT),
    ("s6_media", s6_media.run, s6_media.OUTPUT),
    ("s7_assemble", s7_assemble.run, s7_assemble.OUTPUT),
    ("s8_meta", s8_meta.run, s8_meta.OUTPUT),
    ("s9_publish", s9_publish.run, s9_publish.OUTPUT),
]

# P7「整片翻译」管线：选公有领域纪录片 → zion GPU 转录/去人声 → 中文翻译 →
# 中文配音/字幕 → 合成成片 → AI 缩略图+简介 → 发布 → 归档到 NAS。用户已 pivot
# 到这条链作为主产品（见 config.yaml 的 `pipeline: doc`），但 CLI/config 都能
# 显式切回 `original`。d9_publish 复用既有 s9_publish（读
# meta.json+video.mp4+thumb.jpg，护栏#3 auto_upload/visibility 不变）——
# d8_meta 产出的字段/文件约定与它完全兼容，不用改 s9_publish 本身。P7 T19：
# d9b_bilibili 在 YouTube 发布之后追加发布到 B 站（biliup CLI，best-effort，
# 失败/未登录/未开启都只留状态标记不中止 run）；d9c_publish_en（P7 T20）紧随
# 其后，把英文原声原片 source.mp4（未配音）追加发布到第二个 YouTube 账号/频道
# （档案/引流定位），同样 best-effort，独立 token（secrets/token_en.json）。
# 两者都放在 d9_publish 和 d10_archive 之间，好让 d10_archive 把
# publish_bilibili.json / publish_en.json 也一并归档。d10_archive 把发布完的
# job 整份搬去 NAS 并裁剪本地大媒体，失败只留标记不中止 run（视频已经发布，
# 归档是 best-effort）。
STAGES_P7: list[tuple[str, Callable[[Ctx], None], str]] = [
    ("d1_source", d1_source.run, d1_source.OUTPUT),
    ("d2_gpu", d2_gpu.run, d2_gpu.OUTPUT),
    ("d3_translate", d3_translate.run, d3_translate.OUTPUT),
    ("d4_tts", d4_tts.run, d4_tts.OUTPUT),
    ("d5_subs", d5_subs.run, d5_subs.OUTPUT),
    ("d7_assemble", d7_assemble.run, d7_assemble.OUTPUT),
    ("d8_meta", d8_meta.run, d8_meta.OUTPUT),
    # 用户定(2026-07-06):英文原版先发,中文配音版后发。d9c(best-effort,英文)
    # 在前;d9_publish(主发布,中文)在后。互链 d9d 独立最后一步,顺序无关。
    ("d9c_publish_en", d9c_publish_en.run, d9c_publish_en.OUTPUT),
    ("d9_publish", s9_publish.run, s9_publish.OUTPUT),
    ("d9b_bilibili", d9b_bilibili.run, d9b_bilibili.OUTPUT),
    ("d9d_crosslink", d9d_crosslink.run, d9d_crosslink.OUTPUT),
    ("d10_archive", d10_archive.run, d10_archive.OUTPUT),
]


def _stages_for(pipeline: str) -> list[tuple[str, Callable[[Ctx], None], str]]:
    """按管线名选 STAGES 列表。用全局名（而非在模块加载时快照进 dict）查找，
    这样 `monkeypatch.setattr(run_mod, "STAGES", ...)`（既有测试的写法）依旧
    能生效——dict 快照会在加载时就绑死旧的 list 对象，monkeypatch 换不掉它。"""
    if pipeline == "original":
        return STAGES
    if pipeline == "doc":
        return STAGES_P7
    raise ValueError(f"未知 pipeline: {pipeline!r}（可选 original|doc）")


def run_pipeline(
    slug: str,
    config: Config,
    root: Path,
    llm: LLMProvider,
    frm: str | None = None,
    to: str | None = None,
    force: bool = False,
    pipeline: str | None = None,
) -> None:
    stages = _stages_for(pipeline or config.get("pipeline", "original"))
    names = [s[0] for s in stages]
    i0 = names.index(frm) if frm else 0
    i1 = names.index(to) if to else len(stages) - 1
    ctx = Ctx(slug=slug, config=config, root=root, llm=llm)
    for name, fn, output in stages[i0 : i1 + 1]:
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
    ap.add_argument(
        "--pipeline", choices=["original", "doc"], default=None,
        help="选择管线：original(s1-s9 选题→出片→发布) | doc(P7 d1-d7 整片翻译)。"
             "不传则用 config.yaml 的 `pipeline:` 键，两者都没有则默认 original。",
    )
    args = ap.parse_args(argv)

    config = load_config(args.config)
    root = Path(config.get("paths.jobs_root", "jobs"))
    llm = get_llm(config)
    run_pipeline(args.case, config, root, llm, args.frm, args.to, args.force,
                 pipeline=args.pipeline)


if __name__ == "__main__":
    main()
