from __future__ import annotations

import ast
import inspect
import json
import subprocess
from pathlib import Path

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


# ---------------------------------------------------------------------------
# P7 T17 — 声音克隆配音(tts.provider=="clone")：remote(fake) 批量克隆整部片，
# 这里核对逐段落地 + 缺失回退。remote=fake 全程离线，不 ssh/不触网。
# ---------------------------------------------------------------------------

def _ctx_clone(tmp_path, extra: dict | None = None) -> Ctx:
    data = {
        "remote": {"provider": "fake"},
        "tts": {"provider": "clone", "fallback_provider": "fake",
                "voice": "zh-CN-YunyangNeural", "voice_female": "zh-CN-XiaoxiaoNeural"},
    }
    if extra:
        data.update(extra)
    return Ctx(slug="doc1", config=Config(data), root=tmp_path, llm=FakeLLM())


def _write_dummy_vocals(ctx: Ctx) -> None:
    (ctx.job / "vocals.mp3").write_bytes(b"\x00")   # FakeRemoteGPU.clone_synthesize 不读内容


class _SkippingCloneRemote:
    """跟 FakeRemoteGPU.clone_synthesize 行为一致，但对 skip_ids 里的段故意不
    产出 wav——模拟真实 F5 单段推理失败（remote 侧已按计划跳过、只打印 warn）。"""

    def __init__(self, skip_ids: set[int]):
        self.skip_ids = skip_ids

    def clone_synthesize(self, zh_script: Path, vocals: Path, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        data = json.loads(Path(zh_script).read_text(encoding="utf-8"))
        for seg in data.get("segments", []):
            zh = (seg.get("zh") or "").strip()
            if not zh:
                continue
            seg_id = int(seg.get("id", 0))
            if seg_id in self.skip_ids:
                continue
            out_path = out_dir / f"n{seg_id:03d}.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                 "-t", "1", str(out_path)],
                check=True, capture_output=True,
            )
        return out_dir


def test_clone_mode_writes_clone_voice_manifest_entries(tmp_path):
    ctx = _ctx_clone(tmp_path)
    _write_zh_script(ctx, [
        {"id": 0, "start": 0.0, "end": 1.0, "en": "Hello", "zh": "你好", "gender": "male"},
        {"id": 1, "start": 1.0, "end": 2.0, "en": "World", "zh": "世界", "gender": "female"},
    ])
    _write_dummy_vocals(ctx)

    d4.run(ctx)

    out = read_json(ctx.job, d4.OUTPUT)
    assert len(out["segments"]) == 2
    for seg in out["segments"]:
        assert seg["voice"] == "clone"
        assert seg["file"].endswith(".wav")
        assert seg["tts_dur"] > 0
        p = ctx.job / seg["file"]
        assert p.exists()
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", str(p)],
            capture_output=True, text=True, check=True,
        )
        assert "audio" in probe.stdout
    assert ctx.warnings == []


def test_clone_mode_missing_wav_falls_back_to_fake_engine_and_warns(tmp_path, monkeypatch):
    ctx = _ctx_clone(tmp_path)
    _write_zh_script(ctx, [
        {"id": 0, "start": 0.0, "end": 1.0, "en": "Hello", "zh": "你好", "gender": "male"},
        {"id": 1, "start": 1.0, "end": 2.0, "en": "World", "zh": "世界", "gender": "female"},
    ])
    _write_dummy_vocals(ctx)

    monkeypatch.setattr(d4, "get_remote", lambda config: _SkippingCloneRemote({1}))

    d4.run(ctx)

    out = read_json(ctx.job, d4.OUTPUT)
    voices = {s["id"]: s for s in out["segments"]}

    # id=0 正常克隆
    assert voices[0]["voice"] == "clone"
    assert voices[0]["file"] == "narration/n000.wav"
    assert (ctx.job / voices[0]["file"]).exists()

    # id=1 克隆缺 wav -> 回退 fake 引擎，按性别选中 female 音色，文件落地且完整
    assert voices[1]["voice"] == "zh-CN-XiaoxiaoNeural"
    assert voices[1]["file"] == "narration/n001.mp3"
    assert (ctx.job / voices[1]["file"]).exists()
    assert voices[1]["tts_dur"] > 0
    assert voices[1]["zh"] == "世界"

    assert any("段 1" in w and "克隆配音缺失" in w for w in ctx.warnings)


def test_clone_mode_fallback_provider_default_is_edge(tmp_path):
    """未显式配置 tts.fallback_provider 时默认落 edge（跟非 clone 模式默认引擎一致）。"""
    ctx = _ctx_clone(tmp_path, extra={"tts": {"provider": "clone"}})
    assert ctx.config.get("tts.fallback_provider", "edge") == "edge"


def test_clone_mode_fallback_fake_never_touches_edge_tts(tmp_path, monkeypatch):
    """clone 模式 + fallback_provider=fake 时，缺失段回退也绝不该导入 edge_tts
    （离线测试不触网的关键护栏，跟非 clone 分支的同名护栏对齐）。"""
    import sys

    monkeypatch.setitem(sys.modules, "edge_tts", None)   # import edge_tts 会当场炸
    ctx = _ctx_clone(tmp_path)
    _write_zh_script(ctx, [{"id": 0, "start": 0.0, "end": 1.0, "en": "Hi", "zh": "你好"}])
    _write_dummy_vocals(ctx)

    monkeypatch.setattr(d4, "get_remote", lambda config: _SkippingCloneRemote({0}))

    d4.run(ctx)          # must not raise / must not import edge_tts

    out = read_json(ctx.job, d4.OUTPUT)
    assert out["segments"][0]["voice"] != "clone"
    assert (ctx.job / out["segments"][0]["file"]).exists()


def test_clone_mode_empty_zh_still_skipped_and_warned(tmp_path):
    ctx = _ctx_clone(tmp_path)
    _write_zh_script(ctx, [
        {"id": 0, "start": 0.0, "end": 1.0, "en": "Hello", "zh": "你好"},
        {"id": 1, "start": 1.0, "end": 2.0, "en": "", "zh": ""},
    ])
    _write_dummy_vocals(ctx)

    d4.run(ctx)

    out = read_json(ctx.job, d4.OUTPUT)
    ids = [s["id"] for s in out["segments"]]
    assert ids == [0]
    assert any("1/2" in w for w in ctx.warnings)
