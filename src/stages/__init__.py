"""管线阶段模块集合。

两条管线共用这一个包：
  - 原始 s1-s9：选题 -> 调研 -> 脚本 -> 配音 -> 字幕 -> 配图 -> 合成 -> 元数据 -> 发布。
  - P7 d1-d7（整片翻译）：选片 -> zion GPU 转录/去人声 -> 中文翻译 -> 中文配音 ->
    中文字幕 -> 合成成片。

导出各阶段子模块名，供 `src.run` 的 STAGES/STAGES_P7 装配、以及测试按
`from src.stages import d1_source` 之类的写法直接引用。
"""
from __future__ import annotations

from src.stages import (
    d1_source,
    d2_gpu,
    d3_translate,
    d4_tts,
    d5_subs,
    d7_assemble,
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

__all__ = [
    "d1_source",
    "d2_gpu",
    "d3_translate",
    "d4_tts",
    "d5_subs",
    "d7_assemble",
    "s1_topic",
    "s2_research",
    "s3_script",
    "s4_tts",
    "s5_subs",
    "s6_media",
    "s7_assemble",
    "s8_meta",
    "s9_publish",
]
