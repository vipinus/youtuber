from __future__ import annotations

import ast
import inspect
import subprocess
from pathlib import Path

import pytest

from src.config import Config
from src.context import Ctx
from src.llm import FakeLLM
from src.util.jobs import write_json

import src.stages.d7_assemble as d7
from src.stages.d7_assemble import (
    BACKGROUND,
    NARRATION,
    NARRATION_TRACK,
    OUTPUT,
    SOURCE,
    SUBS,
    build_assemble_cmd,
    build_narration_cmd,
    build_narration_filter,
    run,
)


def _ctx(tmp_path, cfg_data=None) -> Ctx:
    return Ctx(slug="doc1", config=Config(cfg_data or {}), root=tmp_path, llm=FakeLLM())


def _minimal_job(ctx: Ctx) -> None:
    ctx.job.mkdir(parents=True, exist_ok=True)
    (ctx.job / SOURCE).write_bytes(b"MP4")
    (ctx.job / BACKGROUND).write_bytes(b"MP3")
    (ctx.job / SUBS).write_text("[Events]\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# build_narration_filter — cheap deterministic guard on the adelay/amix string
# ---------------------------------------------------------------------------

def test_build_narration_filter_adelay_per_segment_and_amix_count():
    segs = [
        {"id": 0, "start": 1.5, "file": "narration/n000.mp3"},
        {"id": 1, "start": 4.0, "file": "narration/n001.mp3"},
        {"id": 2, "start": 10.25, "file": "narration/n002.mp3"},
    ]
    fc = build_narration_filter(segs, 20.0)
    assert "[0:a]adelay=1500|1500[d0]" in fc
    assert "[1:a]adelay=4000|4000[d1]" in fc
    assert "[2:a]adelay=10250|10250[d2]" in fc
    assert "amix=inputs=3:normalize=0[nar]" in fc
    assert "[d0][d1][d2]amix" in fc                       # 三路都进了 amix
    assert "apad=whole_dur=20.0" in fc
    assert "atrim=0:20.0" in fc


def test_build_narration_filter_single_segment_amix_inputs_1():
    segs = [{"id": 0, "start": 0.0, "file": "narration/n000.mp3"}]
    fc = build_narration_filter(segs, 5.0)
    assert "[0:a]adelay=0|0[d0]" in fc
    assert "amix=inputs=1:normalize=0[nar]" in fc


def test_build_narration_filter_start_defaults_to_zero_when_missing():
    segs = [{"id": 0, "file": "narration/n000.mp3"}]     # 没有 start 字段
    fc = build_narration_filter(segs, 3.0)
    assert "[0:a]adelay=0|0[d0]" in fc


# ---------------------------------------------------------------------------
# build_narration_cmd — pass①：每段旁白 -i + adelay/amix filter
# ---------------------------------------------------------------------------

def test_build_narration_cmd_inputs_in_segment_order(tmp_path):
    segs = [
        {"id": 0, "start": 0.0, "file": "narration/n000.mp3"},
        {"id": 1, "start": 2.0, "file": "narration/n001.mp3"},
    ]
    out = tmp_path / NARRATION_TRACK
    cmd = build_narration_cmd(tmp_path, segs, 5.0, out)
    assert cmd[0] == "ffmpeg"
    assert cmd.count("-i") == 2
    assert str(tmp_path / "narration/n000.mp3") in cmd
    assert str(tmp_path / "narration/n001.mp3") in cmd
    # n000 的 -i 必须先于 n001 的 -i（决定 [0:a]/[1:a] 的映射对不对）
    assert cmd.index(str(tmp_path / "narration/n000.mp3")) < cmd.index(str(tmp_path / "narration/n001.mp3"))
    assert cmd[-1] == str(out)
    assert cmd[cmd.index("-map") + 1] == "[narp]"


# ---------------------------------------------------------------------------
# build_assemble_cmd — pass②：video + (bg duck + narration) + subs
# ---------------------------------------------------------------------------

def test_build_assemble_cmd_with_narration_wires_sidechaincompress(tmp_path):
    source, bg, nar, subs, out = (
        tmp_path / SOURCE, tmp_path / BACKGROUND, tmp_path / NARRATION_TRACK,
        tmp_path / SUBS, tmp_path / OUTPUT,
    )
    cmd = build_assemble_cmd(source, bg, nar, subs, out, bg_gain=0.5)
    assert cmd.count("-i") == 3                          # source + bg + narration
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "volume=0.5" in fc
    assert "sidechaincompress=threshold=0.03:ratio=8:attack=20:release=300" in fc
    assert "[bg][2:a]sidechaincompress" in fc             # bg 被旁白侧链压低
    assert "amix=inputs=2:normalize=0" in fc
    assert "alimiter=limit=0.95" in fc
    assert "subtitles=" in fc
    assert "-map" in cmd and "[vout]" in cmd and "[aout]" in cmd
    assert "-shortest" in cmd
    assert cmd[-1] == str(out)


def test_build_assemble_cmd_without_narration_skips_sidechain(tmp_path):
    source, bg, subs, out = tmp_path / SOURCE, tmp_path / BACKGROUND, tmp_path / SUBS, tmp_path / OUTPUT
    cmd = build_assemble_cmd(source, bg, None, subs, out, bg_gain=0.5)
    assert cmd.count("-i") == 2                          # 只有 source + bg，没有旁白输入
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "sidechaincompress" not in fc
    assert "volume=0.5" in fc
    assert "subtitles=" in fc


def test_build_assemble_cmd_bg_gain_knob_applied(tmp_path):
    source, bg, nar, subs, out = (
        tmp_path / SOURCE, tmp_path / BACKGROUND, tmp_path / NARRATION_TRACK,
        tmp_path / SUBS, tmp_path / OUTPUT,
    )
    cmd = build_assemble_cmd(source, bg, nar, subs, out, bg_gain=0.25)
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "volume=0.25" in fc


# ---------------------------------------------------------------------------
# run() — 两遍 ffmpeg 编排 + config 读取 + warn 通道
# ---------------------------------------------------------------------------

def test_run_two_pass_invokes_ffmpeg_twice(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, {"p7": {"bg_gain": 0.4}})
    _minimal_job(ctx)
    write_json(ctx.job, NARRATION, {"segments": [
        {"id": 0, "start": 0.0, "file": "narration/n000.mp3"},
    ]})

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"OUT")
        class R:
            returncode = 0
            stderr = ""
        return R()

    monkeypatch.setattr(d7, "duration_sec", lambda p: 3.0)
    monkeypatch.setattr(d7.subprocess, "run", fake_run)
    run(ctx)

    assert len(calls) == 2
    assert calls[0][-1] == str(ctx.job / NARRATION_TRACK)      # pass① 旁白轨
    assert calls[1][-1] == str(ctx.job / OUTPUT)                # pass② 成片
    fc2 = calls[1][calls[1].index("-filter_complex") + 1]
    assert "volume=0.4" in fc2                                  # bg_gain 从 config 读取
    assert (ctx.job / OUTPUT).exists()
    assert ctx.warnings == []


def test_run_missing_narration_json_skips_pass1_and_warns(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    _minimal_job(ctx)                                     # 没写 narration.json

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"OUT")
        class R:
            returncode = 0
            stderr = ""
        return R()

    monkeypatch.setattr(d7.subprocess, "run", fake_run)
    run(ctx)

    assert len(calls) == 1                                # 只跑了 pass②
    assert calls[0][-1] == str(ctx.job / OUTPUT)
    fc = calls[0][calls[0].index("-filter_complex") + 1]
    assert "sidechaincompress" not in fc
    assert any("无旁白段" in w for w in ctx.warnings)


def test_run_empty_narration_segments_skips_pass1_and_warns(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    _minimal_job(ctx)
    write_json(ctx.job, NARRATION, {"segments": []})

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"OUT")
        class R:
            returncode = 0
            stderr = ""
        return R()

    monkeypatch.setattr(d7.subprocess, "run", fake_run)
    run(ctx)

    assert len(calls) == 1
    assert ctx.warnings


def test_run_raises_with_ffmpeg_stderr_pass1(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    _minimal_job(ctx)
    write_json(ctx.job, NARRATION, {"segments": [
        {"id": 0, "start": 0.0, "file": "narration/n000.mp3"},
    ]})

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        class R:
            returncode = 1
            stderr = "boom: narration pass failed"
            stdout = ""
        return R()

    monkeypatch.setattr(d7, "duration_sec", lambda p: 3.0)
    monkeypatch.setattr(d7.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as ei:
        run(ctx)
    assert "boom: narration pass failed" in str(ei.value)
    assert len(calls) == 1                                # pass② 不该在 pass① 失败后被调用


def test_run_raises_with_ffmpeg_stderr_pass2(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    _minimal_job(ctx)                                     # 无 narration -> 直接进 pass②

    def fake_run(cmd, **kw):
        class R:
            returncode = 1
            stderr = "boom: assemble pass failed"
            stdout = ""
        return R()

    monkeypatch.setattr(d7.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as ei:
        run(ctx)
    assert "boom: assemble pass failed" in str(ei.value)


# ---------------------------------------------------------------------------
# offline safety: module import does no network at top level
# ---------------------------------------------------------------------------

def test_module_import_does_no_network_at_top_level():
    banned = {"requests", "urllib3", "httpx", "paramiko", "fabric", "google", "edge_tts", "asyncio", "socket"}
    tree = ast.parse(inspect.getsource(d7))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] not in banned for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in banned


# ---------------------------------------------------------------------------
# real-ffmpeg smoke tests — tiny lavfi-synthesized inputs, offline, fast
# ---------------------------------------------------------------------------

def _ffprobe_streams(path: Path) -> tuple[bool, bool, float]:
    """returns (has_video, has_audio, duration_sec)"""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    kinds = set(r.stdout.split())
    d = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return ("video" in kinds, "audio" in kinds, float(d.stdout.strip() or 0.0))


def _require_ffmpeg():
    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0:
        pytest.skip("ffmpeg not available")


def test_real_ffmpeg_smoke_with_narration_produces_valid_video(tmp_path):
    _require_ffmpeg()
    ctx = _ctx(tmp_path, {"p7": {"bg_gain": 0.5}})
    job = ctx.job
    job.mkdir(parents=True)
    (job / "narration").mkdir()

    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=5",
         "-f", "lavfi", "-i", "sine=frequency=220", "-t", "2",
         "-pix_fmt", "yuv420p", "-shortest", str(job / SOURCE)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=110",
         "-t", "2", str(job / BACKGROUND)],
        check=True,
    )
    for name, freq in (("n000.mp3", 440), ("n001.mp3", 550)):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency={freq}",
             "-t", "0.4", str(job / "narration" / name)],
            check=True,
        )
    write_json(job, NARRATION, {"segments": [
        {"id": 0, "start": 0.2, "file": "narration/n000.mp3", "tts_dur": 0.4, "zh": "你好"},
        {"id": 1, "start": 1.0, "file": "narration/n001.mp3", "tts_dur": 0.4, "zh": "世界"},
    ]})
    (job / SUBS).write_text(
        "[Script Info]\n[Events]\nFormat: Layer, Start, End, Text\n", encoding="utf-8",
    )

    run(ctx)

    out = job / OUTPUT
    assert out.exists() and out.stat().st_size > 0
    has_v, has_a, dur = _ffprobe_streams(out)
    assert has_v and has_a
    assert 1.5 <= dur <= 2.5                              # ~源片 2s 时长附近


def test_real_ffmpeg_smoke_zero_narration_produces_valid_video(tmp_path):
    _require_ffmpeg()
    ctx = _ctx(tmp_path)
    job = ctx.job
    job.mkdir(parents=True)

    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=5",
         "-f", "lavfi", "-i", "sine=frequency=220", "-t", "1.5",
         "-pix_fmt", "yuv420p", "-shortest", str(job / SOURCE)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=110",
         "-t", "1.5", str(job / BACKGROUND)],
        check=True,
    )
    (job / SUBS).write_text(
        "[Script Info]\n[Events]\nFormat: Layer, Start, End, Text\n", encoding="utf-8",
    )
    # 故意不写 narration.json -> 0 段旁白路径

    run(ctx)                                              # must not raise

    out = job / OUTPUT
    assert out.exists() and out.stat().st_size > 0
    has_v, has_a, dur = _ffprobe_streams(out)
    assert has_v and has_a
    assert 1.0 <= dur <= 2.0
    assert any("无旁白段" in w for w in ctx.warnings)


def test_tempo_factors_compress_overlapping_clips():
    """真跑发现:-18%慢速让85%段溢出槽位叠到下一条。tempo_factors 须逐段压回窗口。"""
    from src.stages.d7_assemble import tempo_factors, MAX_TEMPO
    segs = [
        {"id": 0, "start": 0.0, "tts_dur": 7.42},   # 窗口4.0s→需压 7.42/3.85≈1.93→封顶1.4
        {"id": 1, "start": 4.0, "tts_dur": 3.0},    # 窗口6.0s→装得下→1.0
        {"id": 2, "start": 10.0, "tts_dur": 5.5},   # 末段,窗口=video_dur-start=5.0→压~1.13
    ]
    f = tempo_factors(segs, video_dur=15.0)
    assert f[0] == MAX_TEMPO                        # 超限封顶
    assert f[1] == 1.0                              # 不溢出不动
    assert 1.05 < f[2] < 1.2                        # 轻度压缩


def test_narration_filter_includes_atempo_only_when_needed():
    from src.stages.d7_assemble import build_narration_filter
    segs = [{"id": 0, "start": 0.0, "tts_dur": 7.0},    # 窗口3s→要压
            {"id": 1, "start": 3.0, "tts_dur": 1.0}]    # 装得下
    fc = build_narration_filter(segs, video_dur=10.0)
    assert "[0:a]atempo=1.4000,atrim=0:2.850,afade=t=out:st=2.650:d=0.2,adelay=0|0[d0]" in fc  # 压满仍超窗→截尾+淡出
    assert "[1:a]adelay=3000|3000[d1]" in fc            # 正常段不带


def test_capped_overlong_clip_is_trimmed_with_fade():
    """压到 MAX_TEMPO 仍超窗的段必须截尾+淡出——绝不双声同放(用户听出叠音)。"""
    from src.stages.d7_assemble import build_narration_filter
    segs = [{"id": 0, "start": 0.0, "tts_dur": 10.0},   # 窗口2.85s,压1.4后仍7.1s→截
            {"id": 1, "start": 3.0, "tts_dur": 1.0}]
    fc = build_narration_filter(segs, video_dur=10.0)
    assert "atrim=0:2.850" in fc and "afade=t=out" in fc
    # 装得下的段不截
    assert "[1:a]adelay=3000|3000[d1]" in fc


def test_fitting_clip_not_trimmed():
    """压缩后能装进窗口的段不截尾(只 atempo)。"""
    from src.stages.d7_assemble import build_narration_filter
    segs = [{"id": 0, "start": 0.0, "tts_dur": 3.3},    # 窗口2.85s,压1.16即装下
            {"id": 1, "start": 3.0, "tts_dur": 1.0}]
    fc = build_narration_filter(segs, video_dur=10.0)
    assert "atempo=" in fc and "atrim=0:2.850" not in fc
