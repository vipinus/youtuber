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
