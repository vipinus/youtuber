from __future__ import annotations

import ast
import inspect

from src.config import Config
from src.context import Ctx
from src.llm import FakeLLM
from src.util.jobs import write_json

import src.stages.d5_subs as d5


def _ctx(tmp_path, cfg_data=None) -> Ctx:
    return Ctx(slug="doc1", config=Config(cfg_data or {}), root=tmp_path, llm=FakeLLM())


def _write_zh_script(ctx: Ctx, segments: list[dict], duration: float = 10.0) -> None:
    write_json(ctx.job, d5.INPUT, {"duration": duration, "segments": segments})


# ---------------------------------------------------------------------------
# happy path — valid ASS with Dialogue lines, single short segment = 1 cue
# spanning the full [start, end] window
# ---------------------------------------------------------------------------

def test_run_writes_valid_ass_with_dialogue_lines(tmp_path):
    ctx = _ctx(tmp_path)
    _write_zh_script(ctx, [
        {"id": 0, "start": 1.0, "end": 3.0, "en": "UFO", "zh": "飞碟目击事件。"},
    ])

    d5.run(ctx)

    txt = (ctx.job / d5.OUTPUT).read_text(encoding="utf-8")
    assert "[Script Info]" in txt
    assert "[Events]" in txt
    assert txt.count("Dialogue:") == 1
    assert "飞碟目击事件。" in txt
    assert ctx.warnings == []


# ---------------------------------------------------------------------------
# long segment -> multiple cues via _split_cues; timings stay within
# [start, end] and are contiguous; no cue starts with punctuation
# ---------------------------------------------------------------------------

def test_long_segment_splits_into_multiple_cues_within_time_span(tmp_path):
    from src.stages.s5_subs import _split_cues

    zh = "然而，军方很快改口。这到底是飞碟，还是气球？"
    expected_cues = _split_cues(zh, max_chars=20)
    assert len(expected_cues) >= 2          # sanity: this text really does split (per s5 tests)

    ctx = _ctx(tmp_path, {"subs": {"max_chars": 20}})
    _write_zh_script(ctx, [
        {"id": 0, "start": 10.0, "end": 20.0, "en": "long sentence", "zh": zh},
    ])

    d5.run(ctx)

    txt = (ctx.job / d5.OUTPUT).read_text(encoding="utf-8")
    dialogue_lines = [ln for ln in txt.splitlines() if ln.startswith("Dialogue:")]
    assert len(dialogue_lines) == len(expected_cues)

    # texts, in order, match _split_cues output exactly (reused, not reinvented)
    texts = [ln.rsplit(",", 1)[-1] for ln in dialogue_lines]
    assert texts == expected_cues
    for t in texts:
        assert t[0] not in "，。！？、；："        # no cue starts with punctuation (s5 guarantee)

    # parse start/end fields (cols: Layer,Start,End,Style,...)
    def _parse(ln: str):
        parts = ln[len("Dialogue: "):].split(",", 3)
        return parts[1], parts[2]

    times = [_parse(ln) for ln in dialogue_lines]
    first_start = times[0][0]
    last_end = times[-1][1]
    assert first_start == "0:00:10.00"          # segment start
    assert last_end == "0:00:20.00"             # segment end
    # cues are contiguous: each cue's end == next cue's start
    for i in range(len(times) - 1):
        assert times[i][1] == times[i + 1][0]


# ---------------------------------------------------------------------------
# empty-zh segment skipped, others unaffected
# ---------------------------------------------------------------------------

def test_empty_zh_segment_is_skipped(tmp_path):
    ctx = _ctx(tmp_path)
    _write_zh_script(ctx, [
        {"id": 0, "start": 0.0, "end": 1.0, "en": "Hello", "zh": "你好"},
        {"id": 1, "start": 1.0, "end": 2.0, "en": "", "zh": ""},
        {"id": 2, "start": 2.0, "end": 3.0, "en": "World", "zh": "世界"},
    ])

    d5.run(ctx)

    txt = (ctx.job / d5.OUTPUT).read_text(encoding="utf-8")
    assert txt.count("Dialogue:") == 2
    assert "你好" in txt and "世界" in txt
    assert any("1/3" in w for w in ctx.warnings)


# ---------------------------------------------------------------------------
# 0 segments -> valid empty ASS (header only), no crash
# ---------------------------------------------------------------------------

def test_zero_segments_writes_header_only_ass_and_warns(tmp_path):
    ctx = _ctx(tmp_path)
    _write_zh_script(ctx, [], duration=0.0)

    d5.run(ctx)          # must not raise

    txt = (ctx.job / d5.OUTPUT).read_text(encoding="utf-8")
    assert "[Script Info]" in txt
    assert "[Events]" in txt
    assert txt.count("Dialogue:") == 0
    assert ctx.warnings


# ---------------------------------------------------------------------------
# reuse guard: d5_subs must import _split_cues/ass_time from s5_subs, not
# reimplement its own copy of the line-breaking or time-formatting logic
# ---------------------------------------------------------------------------

def test_reuses_split_cues_and_ass_time_from_s5_subs():
    src = inspect.getsource(d5)
    assert "from src.stages.s5_subs import" in src
    assert "_split_cues" in src
    assert "ass_time" in src
    # no local re-definition of these functions
    tree = ast.parse(src)
    defined_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "_split_cues" not in defined_names
    assert "ass_time" not in defined_names


# ---------------------------------------------------------------------------
# offline safety: module import does no network at top level
# ---------------------------------------------------------------------------

def test_module_import_does_no_network_at_top_level():
    banned = {"requests", "urllib3", "httpx", "paramiko", "fabric", "google", "edge_tts"}
    tree = ast.parse(inspect.getsource(d5))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] not in banned for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in banned
