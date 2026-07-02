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

P7 T14：`p7.encoder` 编码器开关(默认 `libx264`，笔记本上没有 NVENC)。zion
原生跑管线时设成 `nvenc`（见 config-zion.yaml），第②遍改用
`h264_nvenc -preset p5 -cq 23`——ffmpeg 的 nvenc 编码器要读
`/run/opengl-driver/lib` 下 NixOS 的 NVIDIA 驱动库(已在 zion 上验证)，走
`p7.encoder_lib_path` PREPEND 进 `LD_LIBRARY_PATH`(合并 os.environ，不替换)。
nvenc 编码失败(如驱动没装对/显存被占)不该丢了这条片子——warn 后自动回退
libx264 重试一次，绝不因为 GPU 编码挂了就把整段渲染报废。

背景床来源(`p7.bgm`，默认 replace)：真跑首次上传发现 NASA 公有领域片里常内嵌
片方外购的商业库背景乐，demucs 的 no_vocals 背景床把它原样保留，触发了 4 条
YouTube Content ID 版权认领(不下架但拦变现)。replace 模式改用
`assets/bgm/`(`p7.bgm_dir`) 里的免版税 CC-BY 曲目：按 slug 的 crc32 确定性选一
条(同 slug 永远同曲，换 slug 可能换)，`-stream_loop -1 -t {video_dur}` 循环铺满
源片时长，接同一条 volume=bg_gain + sidechaincompress 侧链——no_vocals.mp3 只是
不再被用作背景床(d2 仍照常产出，留作素材)。选中曲目写进 job 的 bgm.json，供
d8_meta 追加 CC-BY 署名。曲目清单/目录缺失或清单为空 → warn 后回退 original
模式(背景床=no_vocals.mp3)，绝不中止 run。
"""
from __future__ import annotations

import os
import subprocess
import zlib
from pathlib import Path
from typing import Callable

from src.config import Config
from src.context import Ctx
from src.util.ffprobe import duration_sec
from src.util.jobs import exists, read_json, write_json

OUTPUT = "video.mp4"
NARRATION = "narration.json"
SOURCE = "source.mp4"
BACKGROUND = "no_vocals.mp3"
SUBS = "subs.ass"
NARRATION_TRACK = "narration.wav"          # 中间产物：第①遍拼好的旁白轨
BGM_MANIFEST = "bgm.json"                  # p7.bgm_dir 下的曲目清单文件名
JOB_BGM = "bgm.json"                       # job 目录：记录本次选中曲目(供 d8 署名)
DEFAULT_BGM_DIR = "assets/bgm"
DEFAULT_ENCODER_LIB_PATH = "/run/opengl-driver/lib"    # NVENC 需要的 NixOS NVIDIA 驱动库路径(zion 已验证)


NARRATION_GAP_SEC = 0.15      # 相邻旁白间留的喘息间隙
MAX_TEMPO = 1.4               # atempo 压缩比封顶：再快听感就"赶"了；超限段保留少量重叠并告警


def _plan(segments: list[dict], video_dur: float) -> list[tuple[float, float]]:
    """逐段算 (atempo 压缩比, 可用窗口秒)。窗口=到下一条旁白开播为止(留间隙)。"""
    plan: list[tuple[float, float]] = []
    for i, seg in enumerate(segments):
        start = float(seg.get("start", 0.0))
        dur = float(seg.get("tts_dur", 0.0) or 0.0)
        next_start = float(segments[i + 1]["start"]) if i + 1 < len(segments) else video_dur
        window = max(next_start - start - NARRATION_GAP_SEC, 0.3)
        f = dur / window if dur > 0 else 1.0
        plan.append((min(max(f, 1.0), MAX_TEMPO) if f > 1.02 else 1.0, window))
    return plan


def tempo_factors(segments: list[dict], video_dur: float) -> list[float]:
    """逐段算 atempo 压缩比：配音若长过「到下一条旁白开播」的窗口就压回去(留间隙)，
    否则 1.0 保持原速。真跑发现 -18% 慢速让 85% 段溢出槽位、叠到下一条上——
    字幕锚定原时间轴，压音频不压字幕，同步不受影响。"""
    return [f for f, _ in _plan(segments, video_dur)]


def build_narration_filter(segments: list[dict], video_dur: float) -> str:
    """按每段 start 生成 (必要时 atempo 压时 +) adelay，再 amix 叠轨，
    最后 apad/atrim 补齐/裁到源片时长。

    压到 MAX_TEMPO 仍装不进窗口的段：硬截到窗口长 + 0.2s 淡出——宁可丢半句
    尾音，绝不让两段旁白双声同放(用户实际听出叠音)。"""
    n = len(segments)
    plan = _plan(segments, video_dur)
    parts: list[str] = []
    for idx, seg in enumerate(segments):
        ms = int(round(float(seg.get("start", 0.0)) * 1000))
        factor, window = plan[idx]
        dur = float(seg.get("tts_dur", 0.0) or 0.0)
        chain = f"atempo={factor:.4f}," if factor > 1.0 else ""
        if dur > 0 and dur / factor > window + 0.05:   # 压满仍超窗 → 截尾+淡出
            fade_start = max(window - 0.2, 0.0)
            chain += f"atrim=0:{window:.3f},afade=t=out:st={fade_start:.3f}:d=0.2,"
        parts.append(f"[{idx}:a]{chain}adelay={ms}|{ms}[d{idx}]")
    parts.append("".join(f"[d{idx}]" for idx in range(n)) + f"amix=inputs={n}:normalize=0[nar]")
    parts.append(f"[nar]apad=whole_dur={video_dur},atrim=0:{video_dur}[narp]")
    return ";".join(parts)


def build_narration_cmd(job: Path, segments: list[dict], video_dur: float, out_path: Path) -> list[str]:
    cmd: list[str] = ["ffmpeg", "-y"]
    for seg in segments:
        cmd += ["-i", str(job / seg.get("file", ""))]
    cmd += ["-filter_complex", build_narration_filter(segments, video_dur), "-map", "[narp]", str(out_path)]
    return cmd


def pick_bgm_track(bgm_dir: Path, slug: str, warn: Callable[[str], None]) -> dict | None:
    """从 `bgm_dir/bgm.json` 清单里按 slug 的 crc32 确定性挑一条(同 slug 永远同曲，
    换 slug 可能换——不用 Date/random，保证同一 job 重跑选曲不变)。

    清单/目录缺失、清单为空、或选中曲目的文件本身不存在，一律 warn 后返回
    None，调用方回退 original 模式——never crash, never silent-fail。"""
    if not exists(bgm_dir, BGM_MANIFEST):
        warn(f"d7 BGM 清单缺失({bgm_dir / BGM_MANIFEST})，回退用 no_vocals 背景床")
        return None
    try:
        manifest = read_json(bgm_dir, BGM_MANIFEST)
    except Exception as e:
        warn(f"d7 BGM 清单解析失败({type(e).__name__}: {e})，回退用 no_vocals 背景床")
        return None
    tracks = manifest.get("tracks") or []
    if not tracks:
        warn(f"d7 BGM 清单为空({bgm_dir / BGM_MANIFEST})，回退用 no_vocals 背景床")
        return None
    idx = zlib.crc32(slug.encode("utf-8")) % len(tracks)
    entry = tracks[idx]
    path = bgm_dir / entry.get("file", "")
    if not path.exists():
        warn(f"d7 BGM 选中曲目文件缺失({path})，回退用 no_vocals 背景床")
        return None
    return {"path": path, "entry": entry}


def build_assemble_cmd(
    source: Path, bg: Path, narration: Path | None, subs: Path, out_path: Path, bg_gain: float,
    bg_input_opts: list[str] | None = None, encoder: str = "libx264",
) -> list[str]:
    """第②遍：源片视频烧字幕 + 背景床(narration 非空时 sidechaincompress 被旁白 duck，
    再与旁白 amix；narration=None 时只压低背景床，跳过 duck/amix，因为没有旁白可侧链)。

    bg_input_opts：背景输入的额外 ffmpeg input 选项(如 replace 模式的
    `-stream_loop -1 -t {video_dur}` 循环铺满源片时长)，插在背景的 `-i` 之前；
    original 模式(bg=no_vocals.mp3，天然与源片等长)不需要，传 None/空列表。

    encoder：`libx264`(默认，笔记本兼容) | `nvenc`(zion 原生跑时用 GPU 编码，
    见 `p7.encoder`)——nvenc 跑失败时调用方(run())负责 warn + 回退 libx264 重试，
    这里只管拼命令，不做失败判断。"""
    subs_esc = str(subs).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    cmd = ["ffmpeg", "-y", "-i", str(source)] + list(bg_input_opts or []) + ["-i", str(bg)]
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
    video_codec_args = (
        ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23"] if encoder == "nvenc"
        else ["-c:v", "libx264"]
    )
    cmd += [
        "-filter_complex", fc,
        "-map", "[vout]", "-map", "[aout]",
        *video_codec_args, "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", str(out_path),
    ]
    return cmd


def nvenc_env(config: Config, base_env: dict | None = None) -> dict:
    """nvenc 编码要 `/run/opengl-driver/lib` 里的 NixOS NVIDIA 驱动库(zion 上
    验证过)，PREPEND 进 `LD_LIBRARY_PATH`——合并进 base_env(默认 os.environ)，
    不是替换：保留调用方环境里原有的其它变量/已有的 LD_LIBRARY_PATH 路径，
    传给 subprocess.run(env=...)。base_env 参数主要供单测注入固定环境，不依赖
    真实 os.environ 的当前状态。"""
    lib_path = config.get("p7.encoder_lib_path", DEFAULT_ENCODER_LIB_PATH)
    env = dict(base_env if base_env is not None else os.environ)
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib_path}:{existing}" if existing else lib_path
    return env


def run(ctx: Ctx) -> None:
    job = ctx.job
    source = job / SOURCE
    subs = job / SUBS
    bg_gain = ctx.config.get("p7.bg_gain", 0.5)
    bgm_mode = ctx.config.get("p7.bgm", "replace")

    segments = read_json(job, NARRATION).get("segments", []) if exists(job, NARRATION) else []
    narration_path: Path | None = None

    bg = job / BACKGROUND
    bg_input_opts: list[str] = []
    picked = None
    if bgm_mode == "replace":
        bgm_dir = Path(ctx.config.get("p7.bgm_dir", DEFAULT_BGM_DIR))
        picked = pick_bgm_track(bgm_dir, ctx.slug, lambda m: ctx.warn(f"{ctx.slug}: {m}"))
    elif bgm_mode != "original":
        ctx.warn(f"{ctx.slug}: d7 未知 p7.bgm={bgm_mode!r}(应为 replace|original)，回退用 no_vocals 背景床")

    video_dur: float | None = None
    if segments or picked is not None:
        video_dur = round(duration_sec(source), 3)

    if picked is not None:
        bg = picked["path"]
        bg_input_opts = ["-stream_loop", "-1", "-t", f"{video_dur}"]
        write_json(job, JOB_BGM, picked["entry"])

    if segments:
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

    encoder = ctx.config.get("p7.encoder", "libx264")
    cmd2 = build_assemble_cmd(source, bg, narration_path, subs, job / OUTPUT, bg_gain,
                               bg_input_opts, encoder=encoder)
    env2 = nvenc_env(ctx.config) if encoder == "nvenc" else None
    proc2 = subprocess.run(cmd2, capture_output=True, text=True, env=env2)
    if proc2.returncode != 0 and encoder == "nvenc":
        # nvenc 挂了(驱动/显存/环境问题)不该丢了这条片子——warn 后回退 libx264 重试一次，
        # never lose the render。
        ctx.warn(f"{ctx.slug}: d7 nvenc 编码失败(exit {proc2.returncode})，回退 libx264 重试一次: "
                 f"{(proc2.stderr or '')[-500:]}")
        cmd2 = build_assemble_cmd(source, bg, narration_path, subs, job / OUTPUT, bg_gain,
                                   bg_input_opts, encoder="libx264")
        proc2 = subprocess.run(cmd2, capture_output=True, text=True)
    if proc2.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 合成失败 (exit {proc2.returncode}):\n{(proc2.stderr or '')[-2000:]}"
        )
