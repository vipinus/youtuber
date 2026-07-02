from __future__ import annotations

import ast
import inspect

from src.config import Config
from src.context import Ctx
from src.llm import FakeLLM
from src.util.jobs import read_json, write_json

import src.stages.d4_tts as d4


def _ctx(tmp_path, cfg_data=None) -> Ctx:
    cfg = Config(cfg_data or {"tts": {"provider": "fake"}})
    return Ctx(slug="doc1", config=cfg, root=tmp_path, llm=FakeLLM())


def _write_zh_script(ctx: Ctx, segments: list[dict], duration: float = 10.0) -> None:
    write_json(ctx.job, d4.INPUT, {"duration": duration, "segments": segments})


# ---------------------------------------------------------------------------
# happy path — narration.json shape + real mp3 files on disk
# ---------------------------------------------------------------------------

def test_run_writes_narration_manifest_and_audio_files(tmp_path):
    ctx = _ctx(tmp_path)
    _write_zh_script(ctx, [
        {"id": 0, "start": 0.0, "end": 2.0, "en": "Hello", "zh": "飞碟"},       # 2字 -> 0.6s
        {"id": 1, "start": 5.5, "end": 8.0, "en": "Aliens", "zh": "外星人"},   # 3字 -> 0.9s
    ])

    d4.run(ctx)

    out = read_json(ctx.job, d4.OUTPUT)
    assert len(out["segments"]) == 2
    s0, s1 = out["segments"]

    assert s0["id"] == 0
    assert s0["start"] == 0.0            # 锚点 = 视频起点(段 start)，供 d7 定位
    assert s0["file"] == "narration/n000.mp3"
    assert s0["zh"] == "飞碟"
    assert round(s0["tts_dur"], 1) == 0.6

    assert s1["id"] == 1
    assert s1["start"] == 5.5
    assert s1["file"] == "narration/n001.mp3"
    assert round(s1["tts_dur"], 1) == 0.9

    p0 = ctx.job / "narration" / "n000.mp3"
    p1 = ctx.job / "narration" / "n001.mp3"
    assert p0.exists() and p0.stat().st_size > 0
    assert p1.exists() and p1.stat().st_size > 0

    assert ctx.warnings == []


def test_fallback_english_zh_is_synthesized_gracefully(tmp_path):
    """d3_translate 偶尔会因翻译缺失而回退成英文原文——d4 不应特殊拒绝，
    只要非空就照常配音（"处理优雅"= 当普通文本对待，不崩、不特判）。"""
    ctx = _ctx(tmp_path)
    _write_zh_script(ctx, [
        {"id": 0, "start": 0.0, "end": 1.0, "en": "Hi", "zh": "Hi"},  # 翻译缺失回退英文
    ])

    d4.run(ctx)

    out = read_json(ctx.job, d4.OUTPUT)
    assert len(out["segments"]) == 1
    assert out["segments"][0]["zh"] == "Hi"
    assert (ctx.job / "narration" / "n000.mp3").exists()
    assert ctx.warnings == []


# ---------------------------------------------------------------------------
# P7 T9 — gender-matched voice selection
# ---------------------------------------------------------------------------

def test_gender_selects_matching_voice(tmp_path):
    """female -> tts.voice_female；male/None/缺失 -> 默认 tts.voice(不瞎猜)。
    选中的 voice 要记进 narration.json 每段的 "voice" 字段供调试核对。"""
    ctx = _ctx(tmp_path)
    _write_zh_script(ctx, [
        {"id": 0, "start": 0.0, "end": 1.0, "en": "Hi", "zh": "你好", "gender": "female"},
        {"id": 1, "start": 1.0, "end": 2.0, "en": "Yo", "zh": "喂", "gender": "male"},
        {"id": 2, "start": 2.0, "end": 3.0, "en": "Hey", "zh": "嘿", "gender": None},
        {"id": 3, "start": 3.0, "end": 4.0, "en": "Sup", "zh": "咦"},  # no gender key at all
    ])

    d4.run(ctx)

    out = read_json(ctx.job, d4.OUTPUT)
    voices = {s["id"]: s["voice"] for s in out["segments"]}
    assert voices[0] == "zh-CN-XiaoxiaoNeural"   # female -> female voice
    assert voices[1] == "zh-CN-YunyangNeural"    # male -> default voice
    assert voices[2] == "zh-CN-YunyangNeural"    # unknown (null) -> default voice
    assert voices[3] == "zh-CN-YunyangNeural"    # missing gender key -> default voice
    assert ctx.warnings == []
    for s in out["segments"]:
        assert (ctx.job / s["file"]).exists()


def test_gender_voice_config_overrides_are_honored(tmp_path):
    """自定义 tts.voice / tts.voice_female 要被实际用上，不是硬编码默认值。"""
    ctx = _ctx(tmp_path, {"tts": {
        "provider": "fake", "voice": "zh-CN-CustomMale", "voice_female": "zh-CN-CustomFemale",
    }})
    _write_zh_script(ctx, [
        {"id": 0, "start": 0.0, "end": 1.0, "en": "Hi", "zh": "你好", "gender": "female"},
        {"id": 1, "start": 1.0, "end": 2.0, "en": "Yo", "zh": "喂", "gender": "male"},
    ])

    d4.run(ctx)

    out = read_json(ctx.job, d4.OUTPUT)
    voices = {s["id"]: s["voice"] for s in out["segments"]}
    assert voices[0] == "zh-CN-CustomFemale"
    assert voices[1] == "zh-CN-CustomMale"


# ---------------------------------------------------------------------------
# empty-zh segment — skipped + warned, other segments unaffected
# ---------------------------------------------------------------------------

def test_empty_zh_segment_is_skipped_and_warned(tmp_path):
    ctx = _ctx(tmp_path)
    _write_zh_script(ctx, [
        {"id": 0, "start": 0.0, "end": 1.0, "en": "Hello", "zh": "你好"},
        {"id": 1, "start": 1.0, "end": 2.0, "en": "", "zh": ""},        # 空 zh -> 跳过
        {"id": 2, "start": 2.0, "end": 3.0, "en": "World", "zh": "世界"},
    ])

    d4.run(ctx)

    out = read_json(ctx.job, d4.OUTPUT)
    ids = [s["id"] for s in out["segments"]]
    assert ids == [0, 2]                       # 段 1 被跳过，其余不错位
    assert not (ctx.job / "narration" / "n001.mp3").exists()
    assert any("1/3" in w for w in ctx.warnings)


# ---------------------------------------------------------------------------
# 0 usable segments -> empty manifest + warn, no crash
# ---------------------------------------------------------------------------

def test_zero_segments_writes_empty_manifest_and_warns(tmp_path):
    ctx = _ctx(tmp_path)
    _write_zh_script(ctx, [], duration=0.0)

    d4.run(ctx)          # must not raise

    out = read_json(ctx.job, d4.OUTPUT)
    assert out["segments"] == []
    assert ctx.warnings                       # warned, didn't silently no-op


def test_all_zh_empty_writes_empty_manifest_and_warns(tmp_path):
    ctx = _ctx(tmp_path)
    _write_zh_script(ctx, [
        {"id": 0, "start": 0.0, "end": 1.0, "en": "Hi", "zh": ""},
        {"id": 1, "start": 1.0, "end": 2.0, "en": "There", "zh": "  "},   # 纯空白也算空
    ])

    d4.run(ctx)          # must not raise

    out = read_json(ctx.job, d4.OUTPUT)
    assert out["segments"] == []
    assert ctx.warnings


# ---------------------------------------------------------------------------
# offline safety: module import does no network at top level; FakeTTS only
# ---------------------------------------------------------------------------

def test_module_import_does_no_network_at_top_level():
    banned = {"requests", "urllib3", "httpx", "paramiko", "fabric", "google", "edge_tts", "asyncio"}
    tree = ast.parse(inspect.getsource(d4))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] not in banned for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in banned


def test_fake_provider_never_touches_edge_tts(tmp_path, monkeypatch):
    """确保 tts.provider=fake 时压根不导入/调用 edge_tts（离线测试不触网的关键护栏）。"""
    import sys

    monkeypatch.setitem(sys.modules, "edge_tts", None)   # import edge_tts 会当场炸
    ctx = _ctx(tmp_path)
    _write_zh_script(ctx, [{"id": 0, "start": 0.0, "end": 1.0, "en": "Hi", "zh": "你好"}])

    d4.run(ctx)          # must not raise / must not import edge_tts

    out = read_json(ctx.job, d4.OUTPUT)
    assert len(out["segments"]) == 1
