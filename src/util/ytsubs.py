"""YouTube 字幕(json3)→ 转录段(P7,2026-07-04)。

为什么存在:对讲机/无线电对话 whisper large-v3 仍漏抓一多半(Artemis 天地通话
实测 39段/2516字符),而 YouTube 自动字幕抓到 264条/9214字符——Google 的 ASR
用海量 YouTube 音频训练,对自家平台内容是主场。youtube_gov 片源一律优先用
平台字幕当转录真值,whisper 只作性别检测和无字幕时的兜底。

json3 结构:events[].tStartMs/dDurationMs + segs[].utf8。相邻 event 是滚动的
"续行"(时间窗互相重叠),不能直接当独立句用——按句末标点合并成句子,句子的
时间跨度=首行起点→末行终点,重叠自然消解。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_SENT_END = re.compile(r'[.!?。！？]["\')\]]?\s*$')
_MAX_SENT_SEC = 30.0     # 无标点长流保险:超过就硬断句,别让一段吞掉半分钟
_MAX_SENT_CHARS = 220


# 音效/非语音描述词(用户定 2026-07-07:字幕不要音效词)。字幕里的 [music]/[音乐]/
# [applause]/[掌声]/[background noise]/[laughs]…全在方括号 []/【】里;♪♫ 是配乐符;
# >> 是换人标记。圆括号 ()（）是缩写/补充(CNEOS/DART),保留不动。
_SOUND_FX = re.compile(r"\[[^\]]*\]|【[^】]*】|[♪♫]|>>+\s*")


def strip_sound_effects(text: str) -> str:
    """去掉字幕里的音效描述词(方括号类)+配乐符+换人标记,保留圆括号缩写。
    在字幕解析源头剥,不进翻译/配音/显示。"""
    return re.sub(r"\s+", " ", _SOUND_FX.sub(" ", text)).strip()


def _events(path: Path) -> list[tuple[float, float, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: list[tuple[float, float, str]] = []
    for e in data.get("events", []):
        segs = e.get("segs")
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs).replace("\n", " ").strip()
        text = strip_sound_effects(text)     # 音效词在源头剥掉
        if not text:                          # 整条就是音效词(如 "[music]")→ 丢弃
            continue
        t0 = float(e.get("tStartMs", 0)) / 1000.0
        dur = float(e.get("dDurationMs", 0)) / 1000.0
        rows.append((t0, t0 + dur, text))
    return rows


def parse_json3_segments(path: Path) -> list[dict]:
    """json3 → [{"start","end","text"}...],句末标点(或超长保险)断句。"""
    rows = _events(path)
    out: list[dict] = []
    cur_texts: list[str] = []
    cur_start = cur_end = 0.0
    for t0, t1, text in rows:
        if not cur_texts:
            cur_start = t0
        cur_texts.append(text)
        cur_end = max(cur_end, t1)
        joined = " ".join(cur_texts)
        if (_SENT_END.search(text) or cur_end - cur_start > _MAX_SENT_SEC
                or len(joined) > _MAX_SENT_CHARS):
            out.append({"start": round(cur_start, 2), "end": round(cur_end, 2),
                        "text": joined})
            cur_texts, cur_end = [], 0.0
    if cur_texts:
        out.append({"start": round(cur_start, 2), "end": round(cur_end, 2),
                    "text": " ".join(cur_texts)})
    return out


def genders_by_overlap(subs_segs: list[dict], whisper_segs: list[dict]) -> None:
    """把 whisper 转录段(带基频性别)的 gender 按时间重叠映射到字幕段(原地写入)。
    没有重叠命中的段落落 None(下游按默认男声,不瞎猜)。"""
    for s in subs_segs:
        best, best_ov = None, 0.0
        for w in whisper_segs:
            ov = min(s["end"], w.get("end", 0)) - max(s["start"], w.get("start", 0))
            if ov > best_ov and w.get("gender"):
                best, best_ov = w["gender"], ov
        s["gender"] = best
