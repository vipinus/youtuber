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

    字幕与配音共用 d7 的 place_segments 段落重排结果(2026-07-04 起):句子重排到
    新起点、整段落统一速率、末句可能截尾——字幕出现=开口、消失=收声,完全同源。"""
    from src.stages.d7_assemble import place_segments
    out: list[tuple[float, float]] = []
    for pl in place_segments(narr_segments, video_dur):
        eff = pl["dur"] if pl["dur"] > 0 else 2.0
        out.append((pl["start"], pl["start"] + eff))
    return out


def run(ctx: Ctx) -> None:
    max_chars = ctx.config.get("subs.max_chars", 20)
    lines: list[tuple[float, float, str]] = []
    skipped = 0

    narr_path = ctx.job / "narration.json"
    _narr_has_segs = (narr_path.exists()
                      and any(s.get("file") for s in
                              read_json(ctx.job, "narration.json").get("segments", [])))
    # 字幕版(narration 空,mode=subs_only):跳过配音时间轴,直接走下面 zh_script
    # 原字幕时间轴——字幕跟原片文字卡的出现/消失时刻
    if _narr_has_segs:
        # 配音时间轴优先:字幕跟声走(见 _dub_windows)。video_dur 只影响末段窗口,
        # 源片探测失败退化为"末段起点+时长+1s",不炸。
        narr = read_json(ctx.job, "narration.json")
        from src.stages.d7_assemble import enrich_speaker_fields
        segments = enrich_speaker_fields(
            [s for s in narr.get("segments", []) if s.get("file")], ctx.job)
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
