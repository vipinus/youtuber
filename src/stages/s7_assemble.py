from __future__ import annotations

import subprocess
from pathlib import Path

from src.context import Ctx
from src.util.jobs import exists, read_json

OUTPUT = "video.mp4"
INPUT = "timing.json"
CREDITS = "credits.json"

VIDEO_EXTS = {".mp4", ".ogv", ".m4v", ".webm"}          # s6/media.py 视频候选扩展名


def resolve_media(job: Path, segments: list[dict]) -> dict[int, str]:
    """按 id 解析每段真实素材文件名（图或视频）。

    读 credits.json 的 items[].file（以 id 为键）；credits.json 缺失、或某段不在
    其中，一律兜底为 seg_{id:03d}.jpg（向后兼容 s6 的兜底命名）。"""
    by_id: dict[int, str] = {}
    if exists(job, CREDITS):
        try:
            credits = read_json(job, CREDITS)
        except Exception:
            credits = {}
        for item in credits.get("items", []):
            if "id" in item and "file" in item:
                by_id[int(item["id"])] = str(item["file"])

    resolved: dict[int, str] = {}
    for seg in segments:
        sid = int(seg["id"])
        resolved[sid] = by_id.get(sid, f"seg_{sid:03d}.jpg")
    return resolved


def build_cmd(job: Path, segments: list[dict], media_files: dict[int, str], subs: Path,
              fps: int, w: int, h: int) -> list[str]:
    n = len(segments)
    cmd: list[str] = ["ffmpeg", "-y"]
    for seg in segments:                                  # 每段视觉素材：图或视频输入
        sid = int(seg["id"])
        name = media_files.get(sid, f"seg_{sid:03d}.jpg")
        path = job / "media" / name
        dur = seg["dur_sec"]
        if Path(name).suffix.lower() in VIDEO_EXTS:
            # -stream_loop -1 循环片段，-t 截到刚好 dur 秒（短则循环补足，长则截断）
            cmd += ["-stream_loop", "-1", "-t", f"{dur}", "-i", str(path)]
        else:
            cmd += ["-loop", "1", "-t", f"{dur}", "-i", str(path)]
    for seg in segments:                                  # 音频输入
        cmd += ["-i", str(job / "audio" / seg["audio"])]

    parts: list[str] = []
    for i in range(n):                                    # 每段 铺满 1080p，干净时间戳
        parts.append(
            f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},setsar=1,fps={fps},setpts=PTS-STARTPTS[v{i}]"
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
    media_files = resolve_media(ctx.job, segments)
    cmd = build_cmd(ctx.job, segments, media_files, ctx.job / "subs.ass", fps, 1920, 1080)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 合成失败 (exit {proc.returncode}):\n{(proc.stderr or '')[-2000:]}"
        )
