"""P7 阶段⑪：从长片自动切竖屏高光 Shorts，引流长片 + 涨订阅（2026-07-11 用户定）。

为什么做：长片在零权重新号上算法几乎不给流量；Shorts 是 YouTube 唯一会大量推给
新号的格式，能当漏斗顶端把人导向长片并转订阅。且每部长片本就有震撼素材，切高光
是纯增量、不吃 GPU（ffmpeg + 一次 LLM 调用）。

流程（4 步）：
  1. 选高光段：把 zh_script.json 的带时间戳中文脚本喂 ctx.llm，挑 count 个自成一体、
     有钩子的片段，返回 {clips:[{start,end,hook}]}（每段 15~max_sec 秒）。
  2. 切片 + 竖屏改框：从 video.mp4（成片，已含中文配音+烧字幕+台标）切段，blur-pad
     成 9:16（画面居中，上下用同画面模糊填充，纪录片全画面不丢失）。
  3. 叠钩子大字：顶部 drawtext 钩子标题（Noto CJK，半透明底衬）。
  4. 发布：publish=auto 直发 Shorts（标题带 #Shorts、简介回链长片+关注引导）；
     publish=review 只生成本地文件+台账，等人工审核后再发。

护栏（同 d9b/d9d best-effort）：任一步失败只 warn，绝不中止 run，不影响已发的长片。
幂等：shorts.json 已存在（且非空/未强制）就跳过，重跑不重复切/发。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.context import Ctx
from src.util.jobs import read_json, write_json
from src.util.llmjson import extract_json

OUTPUT = "shorts.json"
VIDEO = "video.mp4"
SCRIPT = "zh_script.json"
ZH_PUB = "publish.json"
META = "meta.json"

_SYSTEM = (
    "你是纪录片短视频编导。给你一部中文纪录片的带时间戳分段脚本，"
    "你要挑出最适合做竖屏高光短视频（YouTube Shorts）的片段。"
    "标准：自成一体（不看前后也懂）、有强钩子（惊人事实/戏剧性转折/震撼画面对应的解说）、"
    "情绪或信息密度高。避开平淡的过渡解说、致谢、片头曲。"
)


def _build_prompt(segments: list[dict], count: int, max_sec: int) -> str:
    lines = [f"[{s['id']}] {s['start']:.0f}-{s['end']:.0f}s {s.get('zh','')}" for s in segments]
    body = "\n".join(lines)
    return (
        f"下面是分段脚本（[序号] 起-止秒 中文）：\n{body}\n\n"
        f"挑出 {count} 个最佳高光片段。每段时长 15~{max_sec} 秒（可合并相邻段落，但总时长不超过 {max_sec} 秒）。"
        f"为每段写一句 ≤14 字的中文钩子标题（吸引点击、口语化、别剧透光）。\n"
        '只输出 JSON：{"clips":[{"start":秒,"end":秒,"hook":"钩子标题"}]}，start/end 用脚本里的秒数。'
    )


def _select_clips(ctx: Ctx, script: dict, count: int, max_sec: int) -> list[dict]:
    segments = script.get("segments", [])
    if not segments:
        return []
    raw = ctx.llm.generate(_SYSTEM, _build_prompt(segments, count, max_sec))
    data = extract_json(raw)
    clips = []
    dur_total = float(script.get("duration", 0)) or (segments[-1].get("end", 0) if segments else 0)
    for c in (data.get("clips") or [])[:count]:
        try:
            start, end = float(c["start"]), float(c["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        end = min(end, start + max_sec)                       # 硬顶时长
        if dur_total and end > dur_total:
            end = dur_total
        if end - start < 8:                                   # 太短没意义
            continue
        clips.append({"start": round(start, 2), "end": round(end, 2),
                      "hook": str(c.get("hook", "")).strip()[:20]})
    return clips


def _make_vertical(src: Path, start: float, dur: float, hook: str, font: str, out: Path) -> None:
    """从 src 切 [start,+dur]，blur-pad 成 1080x1920 竖屏，顶部叠钩子大字。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    hook_file = out.with_suffix(".hook.txt")
    hook_file.write_text(hook or "", encoding="utf-8")
    # 模糊背景填充 + 前景等宽居中 + 顶部钩子(半透明底衬)
    draw = ""
    if hook:
        draw = (f",drawtext=fontfile='{font}':textfile='{hook_file}':fontcolor=white:fontsize=58:"
                f"box=1:boxcolor=black@0.45:boxborderw=18:x=(w-tw)/2:y=150:line_spacing=8")
    vf = (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,gblur=sigma=24[bg];"
        "[0:v]scale=1080:-2[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2{draw}[v]"
    )
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", str(src),
        "-filter_complex", vf, "-map", "[v]", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    hook_file.unlink(missing_ok=True)


def _short_meta(hook: str, base_meta: dict, long_url: str) -> dict:
    title = (hook or base_meta.get("title", "精彩片段"))[:90]
    if "#Shorts" not in title:
        title = f"{title} #Shorts"
    desc = (hook or "") + "\n\n"
    if long_url:
        desc += f"▶ 完整纪录片：{long_url}\n"
    desc += "关注 @发现纪录片，每期解锁一部 NASA 中文配音纪录片。\n#Shorts #纪录片 #太空 #NASA"
    tags = list(base_meta.get("tags", []))[:8] + ["Shorts", "短视频"]
    return {"title": title, "description": desc, "tags": tags}


def run(ctx: Ctx) -> None:
    cfg = ctx.config
    if not cfg.get("p7.shorts.enabled", False):
        return
    existing = ctx.job / OUTPUT
    if existing.exists():
        try:
            if (read_json(ctx.job, OUTPUT) or {}).get("clips"):
                return                                        # 幂等：已切过
        except Exception:  # noqa: BLE001
            pass

    # 优先用无片头的 core(d7b_bookend 加片头会整体后移,zh_script 时间戳对不上加了片头的 video)
    video = ctx.job / "video_core.mp4"
    if not video.exists():
        video = ctx.job / VIDEO
    if not video.exists():
        ctx.warn(f"{ctx.slug}: d11_shorts 无 {VIDEO}，跳过")
        return
    if not (ctx.job / SCRIPT).exists():
        ctx.warn(f"{ctx.slug}: d11_shorts 无 {SCRIPT}，跳过")
        return

    count = int(cfg.get("p7.shorts.count", 1))
    max_sec = int(cfg.get("p7.shorts.max_sec", 55))
    mode = str(cfg.get("p7.shorts.publish", "review"))       # review | auto
    visibility = cfg.get("p7.shorts.visibility", "public")
    font = cfg.get("paths.title_font", "")

    try:
        clips = _select_clips(ctx, read_json(ctx.job, SCRIPT), count, max_sec)
    except Exception as e:  # noqa: BLE001
        ctx.warn(f"{ctx.slug}: d11_shorts 高光选段失败({type(e).__name__}: {str(e)[:60]})，跳过")
        return
    if not clips:
        ctx.warn(f"{ctx.slug}: d11_shorts 没选出高光段，跳过")
        return

    long_url = ""
    if (ctx.job / ZH_PUB).exists():
        long_url = (read_json(ctx.job, ZH_PUB) or {}).get("url", "")
    base_meta = read_json(ctx.job, META) if (ctx.job / META).exists() else {}

    made: list[dict] = []
    for i, c in enumerate(clips, 1):
        out = ctx.job / "shorts" / f"short_{i}.mp4"
        entry = {"idx": i, "start": c["start"], "end": c["end"], "hook": c["hook"],
                 "file": f"shorts/short_{i}.mp4", "published": False}
        try:
            _make_vertical(video, c["start"], c["end"] - c["start"], c["hook"], font, out)
        except subprocess.CalledProcessError as e:
            ctx.warn(f"{ctx.slug}: d11_shorts 切片{i}失败: {str(e.stderr)[:80]}")
            continue
        entry["meta"] = _short_meta(c["hook"], base_meta, long_url)
        made.append(entry)

    if not made:
        ctx.warn(f"{ctx.slug}: d11_shorts 全部切片失败，跳过")
        return

    # 发布（auto）或留待审核（review）
    if mode == "auto":
        from src.youtube import load_creds, YouTubeUploader, FakeUploader
        try:
            up = YouTubeUploader(load_creds(
                cfg.get("youtube.token", "secrets/token.json"),
                cfg.get("youtube.client_secret", "secrets/client_secret.json")))
        except Exception as e:  # noqa: BLE001
            ctx.warn(f"{ctx.slug}: d11_shorts 上传器初始化失败({str(e)[:50]})，转为待审")
            up = None
        for entry in made:
            if up is None:
                break
            try:
                res = up.upload(ctx.job / entry["file"], entry["meta"], None, visibility)
                entry.update(published=True, shortId=res.get("videoId"), url=res.get("url"),
                             visibility=visibility)
            except Exception as e:  # noqa: BLE001
                ctx.warn(f"{ctx.slug}: d11_shorts 发布 short_{entry['idx']} 失败: {str(e)[:70]}")

    write_json(ctx.job, OUTPUT, {"mode": mode, "count": len(made), "clips": made})
