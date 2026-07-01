from __future__ import annotations

from src.context import Ctx
from src.tts import get_tts
from src.util.ffprobe import duration_sec
from src.util.jobs import read_json, write_json

OUTPUT = "timing.json"
INPUT = "script.json"


def run(ctx: Ctx) -> None:
    script = read_json(ctx.job, INPUT)
    engine = get_tts(ctx.config)
    audio_dir = ctx.job / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    seg_out: list[dict] = []
    cursor = 0.0
    for seg in script.get("segments", []):
        text = seg.get("text", "")
        data, words = engine.synth(text)
        name = f"seg_{int(seg.get('id', 0)):03d}.mp3"
        (audio_dir / name).write_bytes(data)
        dur = round(duration_sec(audio_dir / name), 3)   # 真实音频时长（含尾静音），修时间轴漂移
        gwords = [
            {
                "text": w["text"],
                "start_sec": round(cursor + w["offset_sec"], 3),
                "dur_sec": w["dur_sec"],
            }
            for w in words
        ]
        seg_out.append({
            "id": seg.get("id"),
            "audio": name,
            "start_sec": round(cursor, 3),
            "dur_sec": round(dur, 3),
            "words": gwords,
        })
        cursor += dur

    write_json(ctx.job, OUTPUT, {"segments": seg_out, "total_sec": round(cursor, 3)})
