"""P7 T5：阶段④——把 d3_translate 产出的中文配音稿(zh_script.json)逐段合成
中文旁白音频(narration/nNNN.mp3)，记录每段的真实时长(tts_dur)和视频锚点起点
(start)，供下游 d7_assemble 按段起点摆放旁白、与(去人声后的)背景混音。

复用 src/tts.py 的 get_tts（FakeTTS 离线可测 / EdgeTTS 真跑），复用
src/util/ffprobe.duration_sec 量真实音频时长（含引擎尾静音，修时间轴漂移，
与 s4_tts 同一手法）。zh_script 的 zh 字段偶尔会因翻译缺失回退成英文——这里
不区分中英文，只要非空就照常配音；只有空串才跳过。
"""
from __future__ import annotations

from src.context import Ctx
from src.tts import get_tts
from src.util.ffprobe import duration_sec
from src.util.jobs import read_json, write_json

OUTPUT = "narration.json"
INPUT = "zh_script.json"


def run(ctx: Ctx) -> None:
    data = read_json(ctx.job, INPUT)
    segments = data.get("segments", [])

    engine = get_tts(ctx.config)
    narration_dir = ctx.job / "narration"
    narration_dir.mkdir(parents=True, exist_ok=True)

    out_segments: list[dict] = []
    skipped = 0
    for seg in segments:
        zh = (seg.get("zh") or "").strip()
        if not zh:
            skipped += 1
            continue
        seg_id = int(seg.get("id", 0))
        audio_bytes, _words = engine.synth(zh)
        name = f"n{seg_id:03d}.mp3"
        path = narration_dir / name
        path.write_bytes(audio_bytes)
        dur = round(duration_sec(path), 3)      # 真实音频时长（含尾静音），供 d7 定位/避免重叠
        out_segments.append({
            "id": seg_id,
            "start": seg.get("start"),           # 视频起点锚点：d7 按此摆放旁白
            "file": f"narration/{name}",
            "tts_dur": dur,
            "zh": zh,
        })

    if skipped:
        ctx.warn(
            f"{ctx.slug}: d4_tts 有 {skipped}/{len(segments)} 段 zh 为空，已跳过配音"
        )
    if not out_segments:
        ctx.warn(f"{ctx.slug}: d4_tts 没有可用配音段（zh_script 为空或全部 zh 缺失）")

    write_json(ctx.job, OUTPUT, {"segments": out_segments})
