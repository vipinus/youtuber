from __future__ import annotations

import ast
import inspect
import subprocess
from pathlib import Path

import pytest

from src.config import Config
from src.context import Ctx
from src.llm import FakeLLM
from src.util.jobs import read_json, write_json

import src.stages.d7_assemble as d7
from src.stages.d7_assemble import (
    BACKGROUND,
    BGM_MANIFEST,
    JOB_BGM,
    NARRATION,
    NARRATION_TRACK,
    OUTPUT,
    SOURCE,
    SUBS,
    build_assemble_cmd,
    build_narration_cmd,
    build_narration_filter,
    pick_bgm_track,
    run,
)


def _ctx(tmp_path, cfg_data=None) -> Ctx:
    """默认 p7.bgm=original——这批既有测试盯的是旁白/字幕两遍 ffmpeg 编排本身，
    不该被"replace 是新默认值"这件事顺带牵连(也不该意外读到仓库根的真实
    assets/bgm)。要测 replace 模式的用例显式传 cfg_data 覆盖。"""
    data = {"p7": {"bgm": "original"}}
    if cfg_data:
        merged_p7 = {**data["p7"], **(cfg_data.get("p7") or {})}
        data = {**cfg_data, "p7": merged_p7}
    return Ctx(slug="doc1", config=Config(data), root=tmp_path, llm=FakeLLM())


def _minimal_job(ctx: Ctx) -> None:
    ctx.job.mkdir(parents=True, exist_ok=True)
    (ctx.job / SOURCE).write_bytes(b"MP4")
    (ctx.job / BACKGROUND).write_bytes(b"MP3")
    (ctx.job / SUBS).write_text("[Events]\n", encoding="utf-8")


def _write_bgm_fixture(bgm_dir: Path, n_tracks: int = 2) -> None:
    """在 bgm_dir 下建一份最小合法清单 + 对应的假 mp3 文件(内容不必是真音频，
    只测清单/路径解析逻辑的用例够用；真 ffmpeg 冒烟测试另建真音频)。"""
    bgm_dir.mkdir(parents=True, exist_ok=True)
    tracks = []
    for i in range(n_tracks):
        fname = f"track{i}.mp3"
        (bgm_dir / fname).write_bytes(b"FAKEMP3")
        tracks.append({
            "file": fname, "title": f"Track {i}", "artist": "Test Artist",
            "license": "CC-BY-4.0", "attribution": f"背景音乐：Track {i} - Test Artist, CC BY 4.0",
        })
    write_json(bgm_dir, BGM_MANIFEST, {"tracks": tracks})


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
# pick_bgm_track — 确定性选曲 + 各种缺失/损坏路径都 warn 后回 None，绝不崩
# ---------------------------------------------------------------------------

def test_pick_bgm_track_deterministic_same_slug_same_track(tmp_path):
    bgm_dir = tmp_path / "bgm"
    _write_bgm_fixture(bgm_dir, n_tracks=2)
    warnings: list[str] = []
    r1 = pick_bgm_track(bgm_dir, "same-slug", warnings.append)
    r2 = pick_bgm_track(bgm_dir, "same-slug", warnings.append)
    assert r1 is not None and r2 is not None
    assert r1["entry"]["file"] == r2["entry"]["file"]
    assert warnings == []


def test_pick_bgm_track_different_slugs_can_differ(tmp_path):
    bgm_dir = tmp_path / "bgm"
    _write_bgm_fixture(bgm_dir, n_tracks=2)
    # crc32("doc1") % 2 == 1, crc32("roswell-1947") % 2 == 0 -- 验证过的确定性哈希，
    # 保证这两个 slug 在 2 首曲子的清单里真的落到不同索引(不是巧合碰运气)。
    a = pick_bgm_track(bgm_dir, "doc1", lambda m: None)
    b = pick_bgm_track(bgm_dir, "roswell-1947", lambda m: None)
    assert a["entry"]["file"] != b["entry"]["file"]


def test_pick_bgm_track_manifest_missing_warns_and_none(tmp_path):
    bgm_dir = tmp_path / "nope"                          # 目录都不存在
    warnings: list[str] = []
    result = pick_bgm_track(bgm_dir, "doc1", warnings.append)
    assert result is None
    assert any("清单缺失" in w and "回退" in w for w in warnings)


def test_pick_bgm_track_manifest_empty_tracks_warns_and_none(tmp_path):
    bgm_dir = tmp_path / "bgm"
    bgm_dir.mkdir()
    write_json(bgm_dir, BGM_MANIFEST, {"tracks": []})
    warnings: list[str] = []
    result = pick_bgm_track(bgm_dir, "doc1", warnings.append)
    assert result is None
    assert any("清单为空" in w for w in warnings)


def test_pick_bgm_track_manifest_malformed_json_warns_and_none(tmp_path):
    bgm_dir = tmp_path / "bgm"
    bgm_dir.mkdir()
    (bgm_dir / BGM_MANIFEST).write_text("{not valid json", encoding="utf-8")
    warnings: list[str] = []
    result = pick_bgm_track(bgm_dir, "doc1", warnings.append)
    assert result is None
    assert any("解析失败" in w for w in warnings)


def test_pick_bgm_track_chosen_file_missing_warns_and_none(tmp_path):
    bgm_dir = tmp_path / "bgm"
    bgm_dir.mkdir()
    write_json(bgm_dir, BGM_MANIFEST, {"tracks": [
        {"file": "ghost.mp3", "title": "Ghost", "artist": "X",
         "license": "CC-BY-4.0", "attribution": "背景音乐：Ghost - X, CC BY 4.0"},
    ]})                                                    # ghost.mp3 没真的落盘
    warnings: list[str] = []
    result = pick_bgm_track(bgm_dir, "doc1", warnings.append)
    assert result is None
    assert any("文件缺失" in w for w in warnings)


# ---------------------------------------------------------------------------
# run() —— p7.bgm=replace：背景床改用 assets/bgm 曲目，loop 到源片时长
# ---------------------------------------------------------------------------

def test_run_replace_mode_uses_bgm_track_with_loop_and_duck(tmp_path, monkeypatch):
    bgm_dir = tmp_path / "bgmpool"
    _write_bgm_fixture(bgm_dir, n_tracks=2)
    ctx = _ctx(tmp_path, {"p7": {"bgm": "replace", "bgm_dir": str(bgm_dir), "bg_gain": 0.5}})
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

    monkeypatch.setattr(d7, "duration_sec", lambda p: 12.0)
    monkeypatch.setattr(d7.subprocess, "run", fake_run)
    run(ctx)

    assert ctx.warnings == []                             # 曲目齐全，不该有 fallback 告警
    cmd2 = calls[1]
    expected = pick_bgm_track(bgm_dir, ctx.slug, lambda m: None)["entry"]["file"]
    assert str(bgm_dir / expected) in cmd2                # 用的是曲库文件
    assert str(ctx.job / BACKGROUND) not in cmd2           # 不是 no_vocals.mp3
    i = cmd2.index(str(bgm_dir / expected))
    assert cmd2[i - 5:i] == ["-stream_loop", "-1", "-t", "12.0", "-i"]   # loop 到源片时长
    fc = cmd2[cmd2.index("-filter_complex") + 1]
    assert "sidechaincompress" in fc                       # ducking 链还在

    bgm_record = read_json(ctx.job, JOB_BGM)
    assert bgm_record["file"] == expected
    assert "attribution" in bgm_record


def test_run_replace_mode_no_narration_still_loops_bgm(tmp_path, monkeypatch):
    """0 段旁白也要走 replace 逻辑（bgm 循环跟旁白轨是否存在无关）。"""
    bgm_dir = tmp_path / "bgmpool"
    _write_bgm_fixture(bgm_dir, n_tracks=1)
    ctx = _ctx(tmp_path, {"p7": {"bgm": "replace", "bgm_dir": str(bgm_dir)}})
    _minimal_job(ctx)                                     # 没写 narration.json

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"OUT")
        class R:
            returncode = 0
            stderr = ""
        return R()

    monkeypatch.setattr(d7, "duration_sec", lambda p: 7.5)
    monkeypatch.setattr(d7.subprocess, "run", fake_run)
    run(ctx)

    assert len(calls) == 1                                # 仍只跑 pass②(无旁白轨)
    cmd = calls[0]
    assert str(bgm_dir / "track0.mp3") in cmd
    i = cmd.index(str(bgm_dir / "track0.mp3"))
    assert cmd[i - 5:i] == ["-stream_loop", "-1", "-t", "7.5", "-i"]
    assert (ctx.job / JOB_BGM).exists()


def test_run_replace_mode_deterministic_same_slug_across_runs(tmp_path, monkeypatch):
    bgm_dir = tmp_path / "bgmpool"
    _write_bgm_fixture(bgm_dir, n_tracks=2)

    def fake_run(cmd, **kw):
        Path(cmd[-1]).write_bytes(b"OUT")
        class R:
            returncode = 0
            stderr = ""
        return R()

    picks = []
    for _ in range(2):
        ctx = Ctx(slug="doc1", config=Config({"p7": {"bgm": "replace", "bgm_dir": str(bgm_dir)}}),
                   root=tmp_path / f"root-{_}", llm=FakeLLM())
        _minimal_job(ctx)
        monkeypatch.setattr(d7, "duration_sec", lambda p: 5.0)
        monkeypatch.setattr(d7.subprocess, "run", fake_run)
        run(ctx)
        picks.append(read_json(ctx.job, JOB_BGM)["file"])
    assert picks[0] == picks[1]                           # 同 slug 两次跑，选曲一致


def test_run_original_mode_uses_no_vocals_unchanged(tmp_path, monkeypatch):
    """显式 p7.bgm=original：背景床还是 no_vocals.mp3，不写 bgm.json，行为与老版本一致。"""
    ctx = _ctx(tmp_path, {"p7": {"bgm": "original"}})
    _minimal_job(ctx)

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

    cmd = calls[0]
    assert str(ctx.job / BACKGROUND) in cmd
    assert "-stream_loop" not in cmd
    assert not (ctx.job / JOB_BGM).exists()


def test_run_replace_mode_manifest_missing_falls_back_to_original(tmp_path, monkeypatch):
    """p7.bgm=replace 但清单/目录不存在 -> warn + 回退 no_vocals，不中止 run。"""
    ctx = _ctx(tmp_path, {"p7": {"bgm": "replace", "bgm_dir": str(tmp_path / "does-not-exist")}})
    _minimal_job(ctx)

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

    cmd = calls[0]
    assert str(ctx.job / BACKGROUND) in cmd                # 回退用了 no_vocals
    assert "-stream_loop" not in cmd
    assert not (ctx.job / JOB_BGM).exists()
    assert any("回退" in w for w in ctx.warnings)
    assert (ctx.job / OUTPUT).exists()                     # 没中止，照样出片


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


def test_real_ffmpeg_smoke_replace_mode_produces_valid_video_with_looped_bgm(tmp_path):
    """replace 模式端到端：真曲目(比源片短，靠 -stream_loop 补足)真烧字幕真编码，
    出片要有视频+音频轨，且 job 里落了 bgm.json 供 d8 署名。"""
    _require_ffmpeg()
    bgm_dir = tmp_path / "bgmpool"
    bgm_dir.mkdir()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=330",
         "-t", "0.6", str(bgm_dir / "short.mp3")],           # 比源片(2s)短，逼真走循环
        check=True,
    )
    write_json(bgm_dir, BGM_MANIFEST, {"tracks": [
        {"file": "short.mp3", "title": "Short Loop", "artist": "Test",
         "license": "CC-BY-4.0", "attribution": "背景音乐：Short Loop - Test, CC BY 4.0"},
    ]})

    ctx = _ctx(tmp_path, {"p7": {"bgm": "replace", "bgm_dir": str(bgm_dir), "bg_gain": 0.5}})
    job = ctx.job
    job.mkdir(parents=True)
    (job / "narration").mkdir()

    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=5",
         "-f", "lavfi", "-i", "sine=frequency=220", "-t", "2",
         "-pix_fmt", "yuv420p", "-shortest", str(job / SOURCE)],
        check=True,
    )
    # no_vocals.mp3 仍然要存在(d2 照常产出)，但 replace 模式不该用它——用不同频率
    # 方便万一断言想区分，这里不做频谱分析，只验证走的是 bgmpool 曲目路径。
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=110",
         "-t", "2", str(job / BACKGROUND)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440",
         "-t", "0.4", str(job / "narration" / "n000.mp3")],
        check=True,
    )
    write_json(job, NARRATION, {"segments": [
        {"id": 0, "start": 0.2, "file": "narration/n000.mp3", "tts_dur": 0.4, "zh": "你好"},
    ]})
    (job / SUBS).write_text(
        "[Script Info]\n[Events]\nFormat: Layer, Start, End, Text\n", encoding="utf-8",
    )

    run(ctx)

    out = job / OUTPUT
    assert out.exists() and out.stat().st_size > 0
    has_v, has_a, dur = _ffprobe_streams(out)
    assert has_v and has_a
    assert 1.5 <= dur <= 2.5
    bgm_record = read_json(job, JOB_BGM)
    assert bgm_record["file"] == "short.mp3"
    assert "attribution" in bgm_record


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
