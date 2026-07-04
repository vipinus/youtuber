"""P7 T5：阶段⑤——把 d3_translate 产出的中文配音稿(zh_script.json)渲染成
中文字幕(subs.ass)，供下游 d7_assemble 烧字幕。

与 s5_subs 不同：s5_subs 有 edge-tts 的逐字时间戳(timing.json/words)可用来把
文字位置精确映射到音频时间；d3_translate 只给出「段」级的 [start, end]（源片
原始时间轴，翻译不改变时长），没有逐字时间戳。所以这里按字符数占比把
[start, end] 线性分给该段切出的每条字幕 cue —— 思路与 s5_subs.seg_lines 完全
一致，只是把「词时间戳算出的 span」换成「段自带的 start/end」。

复用 s5_subs 的 _split_cues（断句规则：句末标点必断/次级标点够长才断/标点
不落行首/不拦腰断词）和 render_ass（含 ASS 头 + 样式 + ass_time 时间格式化），
不重新发明这套已经踩过坑、修过用户反馈的断句/渲染逻辑。
"""
from __future__ import annotations

from src.context import Ctx
from src.stages.s5_subs import _split_cues, ass_time, render_ass  # noqa: F401  (ass_time 经 render_ass 复用)
from src.util.jobs import read_json

OUTPUT = "subs.ass"
INPUT = "zh_script.json"


def _seg_cue_lines(
    zh: str, start: float, end: float, max_chars: int
) -> list[tuple[float, float, str]]:
    """把一段 zh 切成字幕 cue（复用 _split_cues），再按字符数占比把该段的
    [start, end] 线性分给每条 cue（与 s5_subs.seg_lines 的 pos/n 比例映射同思路，
    区别只是这里没有逐字时间戳，直接用段级 start/end 做分母）。"""
    cues = _split_cues(zh, max_chars)
    if not cues:
        return []
    span = max(end - start, 0.001)
    n = len(zh)
    lines: list[tuple[float, float, str]] = []
    pos = 0
    for cue in cues:
        s = start + span * (pos / n)
        pos += len(cue)
        e = start + span * (pos / n)
        lines.append((round(s, 3), round(e, 3), cue))
    return lines


def _dub_windows(narr_segments: list[dict], video_dur: float) -> list[tuple[float, float]]:
    """逐段算配音在成片里的真实播放窗口 [start, start+有效时长]。

    用户反馈(2026-07-04, Cosmic Dawn):字幕跟英文原声时间轴、配音跟自己的时长,
    两条线错位——中文说完字幕还挂着,压速段字幕先于人声。修法:字幕改跟配音。
    有效时长 = min(tts_dur/atempo压缩比, 窗口)——与 d7_assemble 的实际摆放逻辑
    完全同源(共用 _plan),包括压满截尾的段,保证字幕消失点=声音结束点。"""
    from src.stages.d7_assemble import _plan
    plan = _plan(narr_segments, video_dur)
    out: list[tuple[float, float]] = []
    for seg, (factor, window) in zip(narr_segments, plan):
        start = float(seg.get("start", 0.0))
        dur = float(seg.get("tts_dur", 0.0) or 0.0)
        eff = min(dur / factor if factor > 0 else dur, window)
        if eff <= 0:
            eff = min(2.0, window)   # 无时长信息的残段:给个短窗,别让字幕消失不掉
        out.append((start, start + eff))
    return out


def run(ctx: Ctx) -> None:
    max_chars = ctx.config.get("subs.max_chars", 20)
    lines: list[tuple[float, float, str]] = []
    skipped = 0

    narr_path = ctx.job / "narration.json"
    if narr_path.exists():
        # 配音时间轴优先:字幕跟声走(见 _dub_windows)。video_dur 只影响末段窗口,
        # 源片探测失败退化为"末段起点+时长+1s",不炸。
        narr = read_json(ctx.job, "narration.json")
        segments = [s for s in narr.get("segments", []) if s.get("file")]
        try:
            from src.util.ffprobe import duration_sec
            video_dur = duration_sec(ctx.job / "source.mp4")
        except Exception:
            last = segments[-1] if segments else {}
            video_dur = float(last.get("start", 0.0)) + float(last.get("tts_dur", 0.0) or 0.0) + 1.0
        for seg, (s, e) in zip(segments, _dub_windows(segments, video_dur)):
            zh = (seg.get("zh") or "").strip()
            if not zh:
                skipped += 1
                continue
            lines += _seg_cue_lines(zh, s, e, max_chars)
        (ctx.job / OUTPUT).write_text(render_ass(lines), encoding="utf-8")
        if skipped:
            ctx.warn(f"{ctx.slug}: d5_subs 有 {skipped}/{len(segments)} 段 zh 为空，已跳过字幕")
        return

    # 老路径(无配音时间轴):按 zh_script 的原片段级 [start,end] 分布
    data = read_json(ctx.job, INPUT)
    segments = data.get("segments", [])
    for seg in segments:
        zh = (seg.get("zh") or "").strip()
        if not zh:
            skipped += 1
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        lines += _seg_cue_lines(zh, start, end, max_chars)

    if not segments:
        ctx.warn(f"{ctx.slug}: d5_subs zh_script 0 段，生成空字幕")
    elif skipped:
        ctx.warn(f"{ctx.slug}: d5_subs 有 {skipped}/{len(segments)} 段 zh 为空，已跳过字幕")

    (ctx.job / OUTPUT).write_text(render_ass(lines), encoding="utf-8")
