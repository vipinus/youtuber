"""YouTube json3 字幕解析(P7 2026-07-04:平台字幕当转录真值)。全离线。"""
from __future__ import annotations

import json

from src.util.ytsubs import genders_by_overlap, parse_json3_segments


def _j3(tmp_path, events):
    p = tmp_path / "subs.json3"
    p.write_text(json.dumps({"events": events}), encoding="utf-8")
    return p


def test_parse_merges_rolling_lines_into_sentences(tmp_path):
    p = _j3(tmp_path, [
        {"tStartMs": 6100, "dDurationMs": 3100, "segs": [{"utf8": "Station, this is Mission Control"}]},
        {"tStartMs": 7500, "dDurationMs": 4300, "segs": [{"utf8": "Houston. Please call for a"}]},
        {"tStartMs": 9200, "dDurationMs": 2600, "segs": [{"utf8": "voice check."}]},
        {"tStartMs": 13700, "dDurationMs": 6500, "segs": [{"utf8": "How do you hear?"}]},
    ])
    segs = parse_json3_segments(p)
    # 滚动续行按句末标点合并;"Houston." 处收句(续行含句号),两句以上产出
    assert 2 <= len(segs) <= 3
    assert segs[0]["start"] == 6.1
    assert segs[-1]["text"].endswith("How do you hear?")
    joined = " ".join(s["text"] for s in segs)
    assert "voice check." in joined and "Mission Control" in joined


def test_gender_overlap_mapping(tmp_path):
    subs = [{"start": 0.0, "end": 5.0, "text": "a"}, {"start": 10.0, "end": 12.0, "text": "b"}]
    whisper = [{"start": 0.5, "end": 4.0, "gender": "female"},
               {"start": 30.0, "end": 33.0, "gender": "male"}]
    genders_by_overlap(subs, whisper)
    assert subs[0]["gender"] == "female"     # 重叠命中
    assert subs[1]["gender"] is None         # 无重叠不瞎猜


def test_strip_sound_effects_removes_brackets_keeps_parens():
    """用户定(2026-07-07):去音效词——方括号[]/【】+♪+>>去掉,圆括号缩写保留。"""
    from src.util.ytsubs import strip_sound_effects
    assert strip_sound_effects("[music] Hello [applause] world") == "Hello world"
    assert strip_sound_effects("【音乐】大家好【掌声】") == "大家好"
    assert strip_sound_effects("♪ la la ♪ speech") == "la la speech"
    assert strip_sound_effects(">> Speaker: hi") == "Speaker: hi"
    assert strip_sound_effects("[music]") == ""                      # 纯音效→空
    assert strip_sound_effects("近地天体(CNEOS)监测") == "近地天体(CNEOS)监测"  # 圆括号缩写保留


def test_parse_json3_drops_pure_sfx_lines(tmp_path):
    import json
    p = tmp_path / "s.json3"
    p.write_text(json.dumps({"events": [
        {"tStartMs": 1000, "dDurationMs": 2000, "segs": [{"utf8": "[Music]"}]},
        {"tStartMs": 3000, "dDurationMs": 2000, "segs": [{"utf8": "Real speech here."}]},
    ]}), encoding="utf-8")
    from src.util.ytsubs import parse_json3_segments
    segs = parse_json3_segments(p)
    joined = " ".join(s["text"] for s in segs)
    assert "Music" not in joined and "Real speech here." in joined
