from __future__ import annotations

from src.context import Ctx
from src.util.jobs import read_json

OUTPUT = "subs.ass"
INPUT = "timing.json"
SENT_PUNCT = "。！？，、；"

_ASS_HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,3,1,2,80,80,70,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_time(sec: float) -> str:
    cs = int(round(sec * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def build_lines(timing: dict, max_chars: int = 16) -> list[tuple[float, float, str]]:
    lines: list[tuple[float, float, str]] = []
    for seg in timing.get("segments", []):
        cur: list[dict] = []
        for w in seg.get("words", []):
            cur.append(w)
            text = "".join(x["text"] for x in cur)
            last = w["text"][-1] if w.get("text") else ""
            if len(text) >= max_chars or last in SENT_PUNCT:
                start = cur[0]["start_sec"]
                end = w["start_sec"] + w["dur_sec"]
                lines.append((start, end, text))
                cur = []
        if cur:
            start = cur[0]["start_sec"]
            end = cur[-1]["start_sec"] + cur[-1]["dur_sec"]
            lines.append((start, end, "".join(x["text"] for x in cur)))
    return lines


def render_ass(lines: list[tuple[float, float, str]]) -> str:
    font = "Noto Sans CJK SC"
    size = 54
    body = "".join(
        f"Dialogue: 0,{ass_time(s)},{ass_time(e)},Default,,0,0,0,,{text}\n"
        for s, e, text in lines
    )
    return _ASS_HEAD.format(font=font, size=size) + body


def seg_lines(text: str, words: list[dict], max_chars: int) -> list[tuple[float, float, str]]:
    lines = []
    if not words:
        return lines
    start = words[0]["start_sec"]
    end = words[-1]["start_sec"] + words[-1]["dur_sec"]
    span = max(end - start, 0.001)
    n = len(text)
    buf = ""
    line_start_i = 0
    for i, ch in enumerate(text):
        buf += ch
        if len(buf) >= max_chars or ch in SENT_PUNCT or i == n - 1:
            s = start + span * (line_start_i / n)
            e = start + span * ((i + 1) / n)
            lines.append((round(s, 3), round(e, 3), buf))
            buf = ""
            line_start_i = i + 1
    return lines


def run(ctx: Ctx) -> None:
    timing = read_json(ctx.job, INPUT)
    max_chars = ctx.config.get("subs.max_chars", 16)

    # 如果 script.json 存在，用新的 seg_lines 方法保留标点；否则用旧的 build_lines 方法
    script_path = ctx.job / "script.json"
    if script_path.exists():
        script = read_json(ctx.job, "script.json")
        tsegs = {s.get("id"): s for s in timing.get("segments", [])}
        all_lines = []
        for seg in script.get("segments", []):
            words = tsegs.get(seg.get("id"), {}).get("words", [])
            all_lines += seg_lines(seg.get("text", ""), words, max_chars)
        lines = all_lines
    else:
        # 向后兼容：如果没有 script.json，使用旧的 build_lines
        lines = build_lines(timing, max_chars)

    (ctx.job / OUTPUT).write_text(render_ass(lines), encoding="utf-8")
