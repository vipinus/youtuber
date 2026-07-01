from __future__ import annotations

import subprocess
from pathlib import Path

from src.context import Ctx
from src.util.jobs import read_json

OUTPUT = "video.mp4"
INPUT = "timing.json"


def build_cmd(job: Path, segments: list[dict], subs: Path,
              fps: int, w: int, h: int) -> list[str]:
    n = len(segments)
    cmd: list[str] = ["ffmpeg", "-y"]
    for seg in segments:                                  # 图片输入：每段按 dur 展示
        img = job / "media" / f"seg_{int(seg['id']):03d}.jpg"
        cmd += ["-loop", "1", "-t", f"{seg['dur_sec']}", "-i", str(img)]
    for seg in segments:                                  # 音频输入
        cmd += ["-i", str(job / "audio" / seg["audio"])]

    parts: list[str] = []
    for i in range(n):                                    # 每图 铺满 1080p
        parts.append(
            f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},setsar=1,fps={fps}[v{i}]"
        )
    parts.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vcat]")
    for i in range(n):                                    # 音频归一采样率
        parts.append(f"[{n + i}:a]aresample=44100[a{i}]")
    parts.append("".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[acat]")
    subs_esc = str(subs).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    parts.append(f"[vcat]subtitles='{subs_esc}'[vout]")

    cmd += [
        "-filter_complex", ";".join(parts),
        "-map", "[vout]", "-map", "[acat]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-r", str(fps), str(job / OUTPUT),
    ]
    return cmd


def run(ctx: Ctx) -> None:
    timing = read_json(ctx.job, INPUT)
    segments = timing.get("segments", [])
    fps = ctx.config.get("video.fps", 30)
    cmd = build_cmd(ctx.job, segments, ctx.job / "subs.ass", fps, 1920, 1080)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 合成失败 (exit {proc.returncode}):\n{(proc.stderr or '')[-2000:]}"
        )
