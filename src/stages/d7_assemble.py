"""P7 T6：阶段⑦——把 source.mp4(源片视频) + no_vocals.mp3(demucs 去人声后的背景床)
+ narration.json(逐段中文旁白，按 start 时间锚点摆放) + subs.ass(中文字幕) 合成成片
video.mp4。扩展 s7_assemble，思路取自已跑通的
.superpowers/sdd/reference-build_sample.py 第 4 节。

两遍 ffmpeg（比单条塞 100+ 路 -i 的巨型 filter_complex 更稳，样片已验证）：
  1) 旁白轨：每段配音 adelay 到其 start，amix 叠成一路，apad/atrim 补齐/裁到
     源片时长 → narration.wav。
  2) 成片：源片视频原样(不做无谓放大)烧字幕；背景先 volume=bg_gain 压成床，
     再用 sidechaincompress 让旁白路过时自动压低背景(说话时床音量降、间隙
     恢复)——比样片里恒定 volume=0.30 的背景垫更聪明。ducked 背景与旁白
     amix 后 alimiter 防削波，与烧好字幕的视频一起编码。

0 段旁白(narration.json 缺失或 segments 为空)：跳过第①遍，第②遍直接用
背景(压 bg_gain，无旁白可 duck)+字幕合成，warn 不中断——保证任何 job 都能
出片。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from src.context import Ctx
from src.util.ffprobe import duration_sec
from src.util.jobs import exists, read_json

OUTPUT = "video.mp4"
NARRATION = "narration.json"
SOURCE = "source.mp4"
BACKGROUND = "no_vocals.mp3"
SUBS = "subs.ass"
NARRATION_TRACK = "narration.wav"          # 中间产物：第①遍拼好的旁白轨


NARRATION_GAP_SEC = 0.15      # 相邻旁白间留的喘息间隙
MAX_TEMPO = 1.4               # atempo 压缩比封顶：再快听感就"赶"了；超限段保留少量重叠并告警


def tempo_factors(segments: list[dict], video_dur: float) -> list[float]:
    """逐段算 atempo 压缩比：配音若长过「到下一条旁白开播」的窗口就压回去(留间隙)，
    否则 1.0 保持原速。真跑发现 -18% 慢速让 85% 段溢出槽位、叠到下一条上——
    字幕锚定原时间轴，压音频不压字幕，同步不受影响。"""
    factors: list[float] = []
    for i, seg in enumerate(segments):
        start = float(seg.get("start", 0.0))
        dur = float(seg.get("tts_dur", 0.0) or 0.0)
        next_start = float(segments[i + 1]["start"]) if i + 1 < len(segments) else video_dur
        window = max(next_start - start - NARRATION_GAP_SEC, 0.3)
        f = dur / window if dur > 0 else 1.0
        factors.append(min(max(f, 1.0), MAX_TEMPO) if f > 1.02 else 1.0)
    return factors


def build_narration_filter(segments: list[dict], video_dur: float) -> str:
    """按每段 start 生成 (必要时 atempo 压时 +) adelay，再 amix 叠轨，
    最后 apad/atrim 补齐/裁到源片时长。"""
    n = len(segments)
    factors = tempo_factors(segments, video_dur)
    parts: list[str] = []
    for idx, seg in enumerate(segments):
        ms = int(round(float(seg.get("start", 0.0)) * 1000))
        tempo = f"atempo={factors[idx]:.4f}," if factors[idx] > 1.0 else ""
        parts.append(f"[{idx}:a]{tempo}adelay={ms}|{ms}[d{idx}]")
    parts.append("".join(f"[d{idx}]" for idx in range(n)) + f"amix=inputs={n}:normalize=0[nar]")
    parts.append(f"[nar]apad=whole_dur={video_dur},atrim=0:{video_dur}[narp]")
    return ";".join(parts)


def build_narration_cmd(job: Path, segments: list[dict], video_dur: float, out_path: Path) -> list[str]:
    cmd: list[str] = ["ffmpeg", "-y"]
    for seg in segments:
        cmd += ["-i", str(job / seg.get("file", ""))]
    cmd += ["-filter_complex", build_narration_filter(segments, video_dur), "-map", "[narp]", str(out_path)]
    return cmd


def build_assemble_cmd(
    source: Path, bg: Path, narration: Path | None, subs: Path, out_path: Path, bg_gain: float,
) -> list[str]:
    """第②遍：源片视频烧字幕 + 背景床(narration 非空时 sidechaincompress 被旁白 duck，
    再与旁白 amix；narration=None 时只压低背景床，跳过 duck/amix，因为没有旁白可侧链)。"""
    subs_esc = str(subs).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    cmd = ["ffmpeg", "-y", "-i", str(source), "-i", str(bg)]
    if narration is not None:
        cmd += ["-i", str(narration)]
        fc = (
            f"[1:a]volume={bg_gain}[bg];"
            f"[bg][2:a]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=300[ducked];"
            f"[ducked][2:a]amix=inputs=2:normalize=0,alimiter=limit=0.95[aout];"
            f"[0:v]subtitles='{subs_esc}'[vout]"
        )
    else:
        fc = (
            f"[1:a]volume={bg_gain},alimiter=limit=0.95[aout];"
            f"[0:v]subtitles='{subs_esc}'[vout]"
        )
    cmd += [
        "-filter_complex", fc,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", str(out_path),
    ]
    return cmd


def run(ctx: Ctx) -> None:
    job = ctx.job
    source = job / SOURCE
    bg = job / BACKGROUND
    subs = job / SUBS
    bg_gain = ctx.config.get("p7.bg_gain", 0.5)

    segments = read_json(job, NARRATION).get("segments", []) if exists(job, NARRATION) else []
    narration_path: Path | None = None

    if segments:
        video_dur = round(duration_sec(source), 3)
        capped = [s.get("id") for s, f in zip(segments, tempo_factors(segments, video_dur))
                  if f >= MAX_TEMPO]
        if capped:
            ctx.warn(f"{ctx.slug}: d7 有 {len(capped)} 段压到 atempo 上限 {MAX_TEMPO} 仍偏长"
                     f"(可能残留轻微重叠): seg {capped[:8]}")
        narration_path = job / NARRATION_TRACK
        cmd1 = build_narration_cmd(job, segments, video_dur, narration_path)
        proc1 = subprocess.run(cmd1, capture_output=True, text=True)
        if proc1.returncode != 0:
            raise RuntimeError(
                f"ffmpeg 旁白轨合成失败 (exit {proc1.returncode}):\n{(proc1.stderr or '')[-2000:]}"
            )
    else:
        ctx.warn(f"{ctx.slug}: d7_assemble 无旁白段(narration.json 缺失或为空)，仅用背景+字幕合成")

    cmd2 = build_assemble_cmd(source, bg, narration_path, subs, job / OUTPUT, bg_gain)
    proc2 = subprocess.run(cmd2, capture_output=True, text=True)
    if proc2.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 合成失败 (exit {proc2.returncode}):\n{(proc2.stderr or '')[-2000:]}"
        )
