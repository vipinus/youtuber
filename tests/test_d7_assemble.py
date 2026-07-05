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
    DEFAULT_ENCODER_LIB_PATH,
    DEFAULT_WATERMARK,
    JOB_BGM,
    NARRATION,
    NARRATION_TRACK,
    OUTPUT,
    SOURCE,
    SUBS,
    WM_BR,
    WM_TR,
    build_assemble_cmd,
    build_narration_cmd,
    build_narration_filter,
    build_watermark_filter,
    nvenc_env,
    pick_bgm_track,
    run,
    watermark_windows,
)


def _ctx(tmp_path, cfg_data=None) -> Ctx:
    """默认 p7.bgm=original + p7.video_watermark=false——这批既有测试盯的是旁白/
    字幕两遍 ffmpeg 编排本身，不该被"replace 是新默认值"或"水印默认开"这些事
    顺带牵连(也不该意外读到仓库根的真实 assets/bgm、assets/watermark_tr.png)。
    要测 replace/水印的用例显式传 cfg_data 覆盖。"""
    data = {"p7": {"bgm": "original", "video_watermark": False}}
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
    segs = [  # tts_dur=0 的占位段:重排后各段起点=段落锚点顺排(0时长段紧贴)
        {"id": 0, "start": 1.5, "file": "narration/n000.mp3"},
        {"id": 1, "start": 4.0, "file": "narration/n001.mp3"},
        {"id": 2, "start": 10.25, "file": "narration/n002.mp3"},
    ]
    fc = build_narration_filter(segs, 20.0)
    # 排布v2(2026-07-04):句子贴各自原片起点(有富余不移动,空白留在原位)
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


def test_build_narration_cmd_accepts_wav_files_from_clone_mode(tmp_path):
    """P7 T17：声音克隆配音(d4_tts tts.provider=="clone")产出的旁白文件是
    .wav，不是 .mp3。build_narration_cmd 只按 manifest 的 "file" 字段原样拼
    -i，不看/不假设扩展名——ffmpeg 按内容自动探测格式，wav 一样正常工作。"""
    segs = [
        {"id": 0, "start": 0.0, "file": "narration/n000.wav"},
        {"id": 1, "start": 2.0, "file": "narration/n001.wav"},
    ]
    out = tmp_path / NARRATION_TRACK
    cmd = build_narration_cmd(tmp_path, segs, 5.0, out)
    assert cmd[0] == "ffmpeg"
    assert cmd.count("-i") == 2
    assert str(tmp_path / "narration/n000.wav") in cmd
    assert str(tmp_path / "narration/n001.wav") in cmd
    assert cmd[-1] == str(out)


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
# build_assemble_cmd — P7 T14 encoder 开关 (libx264 默认 | nvenc)
# ---------------------------------------------------------------------------

def test_build_assemble_cmd_default_encoder_is_libx264(tmp_path):
    source, bg, subs, out = tmp_path / SOURCE, tmp_path / BACKGROUND, tmp_path / SUBS, tmp_path / OUTPUT
    cmd = build_assemble_cmd(source, bg, None, subs, out, bg_gain=0.5)
    assert "-c:v" in cmd
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert "h264_nvenc" not in cmd


def test_build_assemble_cmd_nvenc_encoder_wires_preset_and_cq(tmp_path):
    source, bg, subs, out = tmp_path / SOURCE, tmp_path / BACKGROUND, tmp_path / SUBS, tmp_path / OUTPUT
    cmd = build_assemble_cmd(source, bg, None, subs, out, bg_gain=0.5, encoder="nvenc")
    assert "libx264" not in cmd
    i = cmd.index("-c:v")
    assert cmd[i:i + 6] == ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23"]
    # 依旧保留通用编码参数
    assert "-pix_fmt" in cmd and cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    assert "-c:a" in cmd and cmd[cmd.index("-c:a") + 1] == "aac"


def test_build_assemble_cmd_nvenc_still_wires_narration_and_subs(tmp_path):
    """encoder 开关只改编码器参数，不该动 filter_complex/字幕/旁白编排逻辑。"""
    source, bg, nar, subs, out = (
        tmp_path / SOURCE, tmp_path / BACKGROUND, tmp_path / NARRATION_TRACK,
        tmp_path / SUBS, tmp_path / OUTPUT,
    )
    cmd = build_assemble_cmd(source, bg, nar, subs, out, bg_gain=0.5, encoder="nvenc")
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "sidechaincompress" in fc
    assert "subtitles=" in fc


# ---------------------------------------------------------------------------
# nvenc_env — LD_LIBRARY_PATH 合并(prepend，不替换)
# ---------------------------------------------------------------------------

def test_nvenc_env_prepends_lib_path_and_preserves_existing_env():
    cfg = Config({"p7": {"encoder_lib_path": "/custom/lib"}})
    base = {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/other/lib", "HOME": "/home/x"}
    env = nvenc_env(cfg, base_env=base)
    assert env["LD_LIBRARY_PATH"] == "/custom/lib:/other/lib"
    assert env["PATH"] == "/usr/bin"                      # 其它变量原样保留
    assert env["HOME"] == "/home/x"
    assert base["LD_LIBRARY_PATH"] == "/other/lib"        # 不修改传入的 base_env 原字典


def test_nvenc_env_default_lib_path_when_no_config_override():
    cfg = Config({})
    env = nvenc_env(cfg, base_env={"PATH": "/usr/bin"})
    assert env["LD_LIBRARY_PATH"] == DEFAULT_ENCODER_LIB_PATH


def test_nvenc_env_no_existing_ld_library_path_uses_lib_path_alone():
    cfg = Config({"p7": {"encoder_lib_path": "/run/opengl-driver/lib"}})
    env = nvenc_env(cfg, base_env={"PATH": "/usr/bin"})    # 没有既有 LD_LIBRARY_PATH
    assert env["LD_LIBRARY_PATH"] == "/run/opengl-driver/lib"


def test_nvenc_env_defaults_to_os_environ_when_base_env_omitted(monkeypatch):
    monkeypatch.setenv("SOME_MARKER_VAR", "present")
    cfg = Config({})
    env = nvenc_env(cfg)
    assert env["SOME_MARKER_VAR"] == "present"            # 真的合并了 os.environ


# ---------------------------------------------------------------------------
# run() —— p7.encoder=nvenc：env 透传 + 失败回退 libx264(never lose the render)
# ---------------------------------------------------------------------------

def test_run_nvenc_encoder_passes_env_with_ld_library_path(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, {"p7": {"encoder": "nvenc"}})
    _minimal_job(ctx)

    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        Path(cmd[-1]).write_bytes(b"OUT")
        class R:
            returncode = 0
            stderr = ""
        return R()

    monkeypatch.setattr(d7.subprocess, "run", fake_run)
    run(ctx)

    assert len(calls) == 1                                # 无旁白，只有 pass②
    cmd, kw = calls[0]
    assert "h264_nvenc" in cmd
    assert kw.get("env") is not None
    assert DEFAULT_ENCODER_LIB_PATH in kw["env"]["LD_LIBRARY_PATH"]
    assert not any("nvenc" in w for w in ctx.warnings)      # 成功一次过，没有 nvenc 相关告警


def test_run_default_encoder_passes_no_special_env(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)                                  # 默认 libx264
    _minimal_job(ctx)

    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        Path(cmd[-1]).write_bytes(b"OUT")
        class R:
            returncode = 0
            stderr = ""
        return R()

    monkeypatch.setattr(d7.subprocess, "run", fake_run)
    run(ctx)

    cmd, kw = calls[0]
    assert "libx264" in cmd
    assert kw.get("env") is None


def test_run_nvenc_falls_back_to_libx264_on_failure_and_warns(tmp_path, monkeypatch):
    """nvenc 挂了(驱动/显存问题)不该丢了这条片子——warn 后回退 libx264 重试一次并成功。"""
    ctx = _ctx(tmp_path, {"p7": {"encoder": "nvenc"}})
    _minimal_job(ctx)

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if "h264_nvenc" in cmd:
            class R:
                returncode = 1
                stderr = "Cannot load libnvidia-encode.so.1"
                stdout = ""
            return R()
        Path(cmd[-1]).write_bytes(b"OUT")
        class R2:
            returncode = 0
            stderr = ""
        return R2()

    monkeypatch.setattr(d7.subprocess, "run", fake_run)
    run(ctx)                                              # 不应抛异常

    assert len(calls) == 2                                # nvenc 失败 + libx264 重试
    assert "h264_nvenc" in calls[0]
    assert "libx264" in calls[1]
    assert (ctx.job / OUTPUT).exists()                    # 片子最终还是出了
    assert any("nvenc" in w and "回退" in w for w in ctx.warnings)


def test_run_nvenc_raises_when_both_nvenc_and_fallback_fail(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, {"p7": {"encoder": "nvenc"}})
    _minimal_job(ctx)

    def fake_run(cmd, **kw):
        class R:
            returncode = 1
            stderr = "boom"
            stdout = ""
        return R()

    monkeypatch.setattr(d7.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError):
        run(ctx)
    assert any("nvenc" in w and "回退" in w for w in ctx.warnings)


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
# watermark_windows — P7 T18：确定性伪随机窗口切分 + TR/BR 交替
# ---------------------------------------------------------------------------

def test_watermark_windows_deterministic_same_slug_same_result():
    w1 = watermark_windows("doc1", 1800.0)
    w2 = watermark_windows("doc1", 1800.0)
    assert w1 == w2                                       # 同 slug 两次调用完全一致
    assert w1                                              # 且真的切出了窗口


def test_watermark_windows_different_slugs_can_differ():
    a = watermark_windows("doc1", 1800.0)
    b = watermark_windows("roswell-1947", 1800.0)
    assert a != b                                          # 换 slug 大概率不同(种子不同)


def test_watermark_windows_intervals_within_bounds_except_possibly_last():
    windows = watermark_windows("doc1", 1800.0, wm_min_sec=300.0, wm_max_sec=600.0)
    assert len(windows) > 1
    for a, b, _ in windows[:-1]:
        assert 300.0 <= (b - a) <= 600.0                   # 除最后一个都在 [min,max] 内
    last_a, last_b, _ = windows[-1]
    assert last_b - last_a <= 600.0                        # 末尾窗口只可能被裁短，不会超限


def test_watermark_windows_tile_video_dur_exactly():
    video_dur = 1800.0
    windows = watermark_windows("doc1", video_dur, wm_min_sec=300.0, wm_max_sec=600.0)
    assert windows[0][0] == 0.0                            # 从 0 开始
    for (a1, b1, _), (a2, b2, _) in zip(windows, windows[1:]):
        assert b1 == a2                                    # 前一个窗口的 end == 下一个的 start
    assert windows[-1][1] == video_dur                     # 铺满到片尾，不多不少


def test_watermark_windows_alternate_positions_starting_tr():
    windows = watermark_windows("doc1", 1800.0, wm_min_sec=300.0, wm_max_sec=600.0)
    assert windows[0][2] == WM_TR                          # 起始位=右上
    for i, (_, _, pos) in enumerate(windows):
        assert pos == (WM_TR if i % 2 == 0 else WM_BR)     # 严格交替


def test_watermark_windows_short_video_single_tr_window():
    """片长比 wm_min_sec 还短(默认 300s)：切不出第二个窗口，整段只有一个 TR 窗口。"""
    windows = watermark_windows("doc1", 200.0)
    assert len(windows) == 1
    a, b, pos = windows[0]
    assert (a, b, pos) == (0.0, 200.0, WM_TR)


def test_watermark_windows_zero_or_negative_duration_returns_empty():
    assert watermark_windows("doc1", 0.0) == []
    assert watermark_windows("doc1", -5.0) == []


def test_watermark_windows_wm_min_max_sec_knobs_applied():
    windows = watermark_windows("doc1", 100.0, wm_min_sec=10.0, wm_max_sec=20.0)
    assert len(windows) > 1                                # 更短的区间下，200s 短片默认单窗口，10~20s 下能切出好几个
    for a, b, _ in windows[:-1]:
        assert 10.0 <= (b - a) <= 20.0


# ---------------------------------------------------------------------------
# build_watermark_filter — 拼 filter_complex 片段本身(scale/overlay/enable)
# ---------------------------------------------------------------------------

def test_build_watermark_filter_multi_window_wires_split_and_both_positions():
    windows = watermark_windows("doc1", 1800.0, wm_min_sec=300.0, wm_max_sec=600.0)
    fc = build_watermark_filter(3, windows)
    assert "[3:v]scale=140:-1,format=rgba,colorchannelmixer=aa=0.85,split[wmA][wmB]" in fc
    assert "[vsub][wmA]overlay=W-w-24:24:enable=" in fc
    assert "[vtr][wmB]overlay=W-w-24:H-h-130:enable=" in fc
    assert fc.endswith("[vout]")
    tr_expr = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b, p in windows if p == WM_TR)
    br_expr = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b, p in windows if p == WM_BR)
    assert tr_expr in fc
    assert br_expr in fc


def test_build_watermark_filter_single_window_skips_split_and_second_overlay():
    windows = watermark_windows("doc1", 200.0)                # 单一 TR 窗口
    fc = build_watermark_filter(2, windows)
    assert "split" not in fc
    assert "wmB" not in fc
    assert "H-h-130" not in fc
    assert "[2:v]scale=140:-1,format=rgba,colorchannelmixer=aa=0.85[wm]" in fc
    assert "overlay=W-w-24:24:enable='between(t,0.000,200.000)'[vout]" in fc


# ---------------------------------------------------------------------------
# build_assemble_cmd — P7 T18 水印叠加 wiring(narration/no-narration 两种变体)
# ---------------------------------------------------------------------------

def _wm_file(tmp_path) -> Path:
    p = tmp_path / "wm.png"
    p.write_bytes(b"FAKEPNG")                                 # 只测路径/命令编排，不需要真图
    return p


def test_build_assemble_cmd_watermark_wires_overlay_no_narration_variant(tmp_path):
    source, bg, subs, out = tmp_path / SOURCE, tmp_path / BACKGROUND, tmp_path / SUBS, tmp_path / OUTPUT
    wm = _wm_file(tmp_path)
    cfg = Config({"p7": {"video_watermark": True, "watermark": str(wm)}})
    cmd = build_assemble_cmd(source, bg, None, subs, out, bg_gain=0.5,
                              video_dur=1800.0, slug="doc1", config=cfg)
    assert cmd.count("-i") == 3                              # source + bg + watermark
    assert str(wm) in cmd
    assert cmd[cmd.index(str(wm)) - 1] == "-i"
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "[2:v]scale=140:-1" in fc                          # wm_idx=2(无旁白：source=0,bg=1)
    assert "colorchannelmixer=aa=0.85" in fc
    assert "split[wmA][wmB]" in fc
    assert "overlay=W-w-24:24:enable=" in fc
    assert "overlay=W-w-24:H-h-130:enable=" in fc
    assert "subtitles=" in fc and "[vsub]" in fc
    assert "sidechaincompress" not in fc                      # 无旁白变体：ducking 逻辑不受水印影响
    assert "volume=0.5" in fc                                 # 音频编排照旧
    assert "-map" in cmd and "[vout]" in cmd and "[aout]" in cmd
    assert cmd[-1] == str(out)


def test_build_assemble_cmd_watermark_wires_overlay_narration_variant(tmp_path):
    source, bg, nar, subs, out = (
        tmp_path / SOURCE, tmp_path / BACKGROUND, tmp_path / NARRATION_TRACK,
        tmp_path / SUBS, tmp_path / OUTPUT,
    )
    wm = _wm_file(tmp_path)
    cfg = Config({"p7": {"video_watermark": True, "watermark": str(wm)}})
    cmd = build_assemble_cmd(source, bg, nar, subs, out, bg_gain=0.5,
                              video_dur=1800.0, slug="doc1", config=cfg)
    assert cmd.count("-i") == 4                              # source + bg + narration + watermark
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "[3:v]scale=140:-1" in fc                          # wm_idx=3(有旁白：source=0,bg=1,narration=2)
    assert "sidechaincompress=threshold=0.03:ratio=8:attack=20:release=300" in fc
    assert "[bg][2:a]sidechaincompress" in fc                 # 旁白 duck 音频逻辑不受水印影响
    assert "amix=inputs=2:normalize=0" in fc
    assert "alimiter=limit=0.95" in fc
    assert "split[wmA][wmB]" in fc
    assert "-map" in cmd and "[vout]" in cmd and "[aout]" in cmd
    assert cmd[-1] == str(out)


def test_build_assemble_cmd_watermark_enable_exprs_match_generated_windows(tmp_path):
    source, bg, subs, out = tmp_path / SOURCE, tmp_path / BACKGROUND, tmp_path / SUBS, tmp_path / OUTPUT
    wm = _wm_file(tmp_path)
    cfg = Config({"p7": {"video_watermark": True, "watermark": str(wm), "wm_min_sec": 300, "wm_max_sec": 600}})
    cmd = build_assemble_cmd(source, bg, None, subs, out, bg_gain=0.5,
                              video_dur=1800.0, slug="doc1", config=cfg)
    fc = cmd[cmd.index("-filter_complex") + 1]
    windows = watermark_windows("doc1", 1800.0, 300, 600)
    tr_expr = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b, p in windows if p == WM_TR)
    br_expr = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b, p in windows if p == WM_BR)
    assert f"enable='{tr_expr}'" in fc
    assert f"enable='{br_expr}'" in fc


def test_build_assemble_cmd_watermark_short_video_single_window_no_second_overlay(tmp_path):
    source, bg, subs, out = tmp_path / SOURCE, tmp_path / BACKGROUND, tmp_path / SUBS, tmp_path / OUTPUT
    wm = _wm_file(tmp_path)
    cfg = Config({"p7": {"video_watermark": True, "watermark": str(wm)}})
    cmd = build_assemble_cmd(source, bg, None, subs, out, bg_gain=0.5,
                              video_dur=200.0, slug="doc1", config=cfg)     # < wm_min_sec(300s) 默认下限
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "split" not in fc
    assert "H-h-130" not in fc
    assert "overlay=W-w-24:24:enable='between(t,0.000,200.000)'[vout]" in fc


def test_build_assemble_cmd_watermark_wm_min_max_sec_config_knobs_applied(tmp_path):
    source, bg, subs, out = tmp_path / SOURCE, tmp_path / BACKGROUND, tmp_path / SUBS, tmp_path / OUTPUT
    wm = _wm_file(tmp_path)
    cfg = Config({"p7": {"video_watermark": True, "watermark": str(wm), "wm_min_sec": 10, "wm_max_sec": 20}})
    cmd = build_assemble_cmd(source, bg, None, subs, out, bg_gain=0.5,
                              video_dur=100.0, slug="doc1", config=cfg)
    fc = cmd[cmd.index("-filter_complex") + 1]
    windows = watermark_windows("doc1", 100.0, 10, 20)
    assert len(windows) > 1                                   # 用了更小的窗口才切得出好几个
    assert "split[wmA][wmB]" in fc


# ---------------------------------------------------------------------------
# build_assemble_cmd — P7 T18 水印关闭/缺失文件 → 回到老行为，绝不中止
# ---------------------------------------------------------------------------

def test_build_assemble_cmd_watermark_disabled_identical_to_no_config(tmp_path):
    source, bg, subs, out = tmp_path / SOURCE, tmp_path / BACKGROUND, tmp_path / SUBS, tmp_path / OUTPUT
    wm = _wm_file(tmp_path)                                   # 文件真实存在，但开关关了
    baseline = build_assemble_cmd(source, bg, None, subs, out, bg_gain=0.5)
    cfg = Config({"p7": {"video_watermark": False, "watermark": str(wm)}})
    warned = []
    cmd = build_assemble_cmd(source, bg, None, subs, out, bg_gain=0.5,
                              video_dur=1800.0, slug="doc1", config=cfg, warn=warned.append)
    assert cmd == baseline                                    # 关了就跟老行为一模一样
    assert warned == []                                       # 关闭不算异常，不该 warn


def test_build_assemble_cmd_watermark_missing_file_warns_and_falls_back(tmp_path):
    source, bg, subs, out = tmp_path / SOURCE, tmp_path / BACKGROUND, tmp_path / SUBS, tmp_path / OUTPUT
    baseline = build_assemble_cmd(source, bg, None, subs, out, bg_gain=0.5)
    missing = tmp_path / "no-such-watermark.png"
    cfg = Config({"p7": {"video_watermark": True, "watermark": str(missing)}})
    warned = []
    cmd = build_assemble_cmd(source, bg, None, subs, out, bg_gain=0.5,
                              video_dur=1800.0, slug="doc1", config=cfg, warn=warned.append)
    assert cmd == baseline                                    # 文件缺失 → 回到老命令形状
    assert any("缺失" in w and str(missing) in w for w in warned)


def test_build_assemble_cmd_watermark_no_config_identical_to_no_config(tmp_path):
    """config=None(既有直调用例的默认调法) → 跟压根没有水印功能时一模一样，
    保证老测试不用改就能继续过。"""
    source, bg, subs, out = tmp_path / SOURCE, tmp_path / BACKGROUND, tmp_path / SUBS, tmp_path / OUTPUT
    baseline = build_assemble_cmd(source, bg, None, subs, out, bg_gain=0.5)
    cmd = build_assemble_cmd(source, bg, None, subs, out, bg_gain=0.5,
                              video_dur=1800.0, slug="doc1")   # 没传 config
    assert cmd == baseline


def test_build_assemble_cmd_watermark_no_video_dur_skips_even_if_file_exists(tmp_path):
    """video_dur 缺失(没算出片长)时即便开关开着文件也在，也不该硬叠(没有时长就没法切窗口)。"""
    source, bg, subs, out = tmp_path / SOURCE, tmp_path / BACKGROUND, tmp_path / SUBS, tmp_path / OUTPUT
    wm = _wm_file(tmp_path)
    baseline = build_assemble_cmd(source, bg, None, subs, out, bg_gain=0.5)
    cfg = Config({"p7": {"video_watermark": True, "watermark": str(wm)}})
    cmd = build_assemble_cmd(source, bg, None, subs, out, bg_gain=0.5, slug="doc1", config=cfg)
    assert cmd == baseline


# ---------------------------------------------------------------------------
# run() —— P7 T18：video_dur/slug/config 接进 build_assemble_cmd 的调用点
# ---------------------------------------------------------------------------

def test_run_watermark_enabled_no_narration_computes_duration_and_wires_overlay(tmp_path, monkeypatch):
    """0 段旁白也要能出水印(run() 之前只在有旁白/replace bgm 时才 ffprobe 时长，
    水印功能要求任何路径都得先拿到 video_dur 才能切窗口)。"""
    wm = _wm_file(tmp_path)
    ctx = _ctx(tmp_path, {"p7": {"video_watermark": True, "watermark": str(wm)}})
    _minimal_job(ctx)                                         # 没写 narration.json

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"OUT")
        class R:
            returncode = 0
            stderr = ""
        return R()

    monkeypatch.setattr(d7, "duration_sec", lambda p: 400.0)
    monkeypatch.setattr(d7.subprocess, "run", fake_run)
    run(ctx)

    assert len(calls) == 1                                    # 仍只跑 pass②(无旁白)
    cmd = calls[0]
    assert str(wm) in cmd
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "scale=140:-1" in fc
    assert "[vsub]" in fc
    assert not any("水印" in w for w in ctx.warnings)         # 文件齐全，不该有水印相关告警


def test_run_watermark_enabled_missing_file_warns_and_skips_overlay(tmp_path, monkeypatch):
    missing = tmp_path / "no-such-wm.png"
    ctx = _ctx(tmp_path, {"p7": {"video_watermark": True, "watermark": str(missing)}})
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
    assert str(missing) not in cmd
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "scale=140" not in fc
    assert any("水印" in w and "缺失" in w for w in ctx.warnings)
    assert (ctx.job / OUTPUT).exists()                        # 没中止，照样出片


def test_run_watermark_disabled_by_default_in_test_helper_matches_old_behavior(tmp_path, monkeypatch):
    """default_watermark(assets/watermark_tr.png) 在仓库里是真实存在的文件——
    _ctx() 默认关闭 video_watermark，防止这批既有测试被"水印默认开"意外牵连。"""
    ctx = _ctx(tmp_path)
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

    fc = calls[0][calls[0].index("-filter_complex") + 1]
    assert "scale=140" not in fc
    assert DEFAULT_WATERMARK not in "".join(calls[0])


# ---------------------------------------------------------------------------
# real-ffmpeg smoke — 水印真叠加(tiny 2s lavfi 源 + tiny png 水印)
# ---------------------------------------------------------------------------

def test_real_ffmpeg_smoke_with_watermark_overlay_produces_valid_video(tmp_path):
    """真 ffmpeg 冒烟：2s 源片 + 极小间隔(0.5~0.6s)逼真触发好几次 TR/BR 切换，
    验证 filter_complex 语法真的能被 ffmpeg 接受、渲染出正常的视频+音频。"""
    _require_ffmpeg()
    wm_dir = tmp_path / "wmassets"
    wm_dir.mkdir()
    wm_path = wm_dir / "wm.png"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0x224488:s=32x32",
         "-frames:v", "1", str(wm_path)],
        check=True,
    )

    ctx = _ctx(tmp_path, {"p7": {"video_watermark": True, "watermark": str(wm_path),
                                  "wm_min_sec": 0.5, "wm_max_sec": 0.6}})
    job = ctx.job
    job.mkdir(parents=True)

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
    (job / SUBS).write_text(
        "[Script Info]\n[Events]\nFormat: Layer, Start, End, Text\n", encoding="utf-8",
    )
    # 故意不写 narration.json -> 0 段旁白路径，同时验证水印在这个变体也能正确叠加

    run(ctx)

    out = job / OUTPUT
    assert out.exists() and out.stat().st_size > 0
    has_v, has_a, dur = _ffprobe_streams(out)
    assert has_v and has_a
    assert 1.0 <= dur <= 2.5
    assert not any("水印" in w for w in ctx.warnings)          # 曲目齐全，不该有水印相关告警


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
    """用户定(2026-07-04):按角色分块变速——同一角色连续段共用一个速率(块内取
    最大需求,封顶1.25),块间可不同;字数预算把多数需求压到1.0附近。"""
    from src.stages.d7_assemble import tempo_factors, MAX_TEMPO
    segs = [
        {"id": 0, "start": 0.0, "tts_dur": 7.42, "gender": "male"},   # 男声段落:重排吃掉空白
        {"id": 1, "start": 4.0, "tts_dur": 3.0, "gender": "male"},    # 同段落同一速率
        {"id": 2, "start": 10.0, "tts_dur": 4.0, "gender": "female"}, # 女声段落:装得下→1.0
    ]
    f = tempo_factors(segs, video_dur=15.0)
    # 段落窗口=10-0.15=9.85, 语音10.42+呼吸0.35 → 需1.097(装不下0.8才按需加快)
    assert f[0] == f[1] and 1.05 < f[0] < 1.15      # 段落内一致且远低于封顶
    assert 0.8 <= f[2] < 0.85                       # 另一段落:回落目标语速0.8
    assert MAX_TEMPO == 1.25


def test_narration_filter_includes_atempo_only_when_needed():
    from src.stages.d7_assemble import build_narration_filter
    segs = [{"id": 0, "start": 0.0, "tts_dur": 7.0},    # 窗口3s→要压
            {"id": 1, "start": 3.0, "tts_dur": 1.0}]    # 装得下
    segs[0]["gender"], segs[1]["gender"] = "male", "female"   # 两个角色两块
    fc = build_narration_filter(segs, video_dur=10.0)
    assert "[0:a]atempo=1.2500,atrim=0:2.850,afade=t=out:st=2.650:d=0.2,adelay=0|0[d0]" in fc  # 男块封顶仍超→截尾
    assert "[1:a]atempo=0.8000,adelay=3000|3000[d1]" in fc  # 女块装得下:目标语速0.8


def test_capped_overlong_clip_is_trimmed_with_fade():
    """压到 MAX_TEMPO 仍超窗的段必须截尾+淡出——绝不双声同放(用户听出叠音)。"""
    from src.stages.d7_assemble import build_narration_filter
    segs = [{"id": 0, "start": 0.0, "tts_dur": 10.0, "gender": "male"},   # 窗口2.85s,封顶仍8s→截
            {"id": 1, "start": 3.0, "tts_dur": 1.0, "gender": "female"}]  # 另一角色块,不被牵连
    fc = build_narration_filter(segs, video_dur=10.0)
    assert "atrim=0:2.850" in fc and "afade=t=out" in fc
    # 另一角色块装得下:目标语速0.8,不截
    assert "[1:a]atempo=0.8000,adelay=3000|3000[d1]" in fc


def test_fitting_clip_not_trimmed():
    """块内轻度变速即可装下的段:统一速率、不截尾;另一角色不被牵连。"""
    from src.stages.d7_assemble import build_narration_filter
    segs = [{"id": 0, "start": 0.0, "tts_dur": 3.3, "gender": "male"},    # 需1.158<1.25→块速1.1579
            {"id": 1, "start": 3.0, "tts_dur": 1.0, "gender": "female"}]  # 女块装得下
    fc = build_narration_filter(segs, video_dur=10.0)
    assert "[0:a]atempo=1.1579,adelay=0|0[d0]" in fc     # 按需加快装下:无截尾
    assert "[1:a]atempo=0.8000,adelay=3000|3000[d1]" in fc  # 目标语速0.8


# ---------------------------------------------------------------------------
# 按影片内容选 BGM(用户定 2026-07-03:不再全线同一首)
# ---------------------------------------------------------------------------

def test_infer_bgm_mood_keyword_buckets():
    from src.stages.d7_assemble import infer_bgm_mood
    assert infer_bgm_mood("Searching for Type D: A New Species of Killer Whale?") == "ocean"
    assert infer_bgm_mood("Wolves of Yellowstone") == "nature"
    assert infer_bgm_mood("NASA's Kepler Mission Announces Discovery") == "space"
    assert infer_bgm_mood("Annual Budget Briefing") is None            # 命不中→None→全池


def test_pick_bgm_track_filters_by_mood(tmp_path):
    import json
    from src.stages.d7_assemble import pick_bgm_track
    bgm = tmp_path / "bgm"
    bgm.mkdir()
    tracks = [
        {"file": "space1.mp3", "moods": ["space"], "attribution": "a"},
        {"file": "ocean1.mp3", "moods": ["ocean"], "attribution": "b"},
        {"file": "any1.mp3", "attribution": "c"},                      # 没配 moods=百搭
    ]
    for t in tracks:
        (bgm / t["file"]).write_bytes(b"MP3")
    (bgm / "bgm.json").write_text(json.dumps({"tracks": tracks}), encoding="utf-8")

    warns = []
    got = pick_bgm_track(bgm, "some-slug", warns.append, mood="ocean")
    assert got["entry"]["file"] in ("ocean1.mp3", "any1.mp3")          # space 被滤掉
    # 同 slug 同 mood 重跑选曲不变(确定性)
    again = pick_bgm_track(bgm, "some-slug", warns.append, mood="ocean")
    assert again["entry"]["file"] == got["entry"]["file"]
    # mood=None → 全池;桶空(没有 somber 专属曲) → 回退全池不报错
    assert pick_bgm_track(bgm, "s", warns.append) is not None
    assert pick_bgm_track(bgm, "s", warns.append, mood="somber") is not None
    assert not warns


def test_ensure_fd_limit_raises_soft_limit():
    """1261段旁白撞默认fd软限1024(Cosmic Dawn真跑)——软限按需提升,不超硬限。"""
    import resource
    from src.stages.d7_assemble import _ensure_fd_limit
    soft0, hard0 = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        need = min(soft0 + 512, hard0 if hard0 != resource.RLIM_INFINITY else soft0 + 512)
        _ensure_fd_limit(need)
        soft1, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        assert soft1 >= need
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft0, hard0))


def test_narration_filter_loudness_normalizes_voice():
    """用户反馈(2026-07-04):配音音量太小——旁白轨 amix 后统一 loudnorm 到 -16 LUFS。"""
    from src.stages.d7_assemble import build_narration_filter
    fc = build_narration_filter(
        [{"start": 0.0, "tts_dur": 3.0, "file": "narration/n000.wav"}], 20.0)
    assert "loudnorm=I=-16" in fc
    assert fc.index("amix") < fc.index("loudnorm") < fc.index("apad")   # 混完归一再补齐


def test_place_segments_tracks_original_starts_and_respects_hard_breaks():
    """排布v2(用户定 2026-07-04):句子贴原片起点保画面同步;与前句冲突才后移
    (留0.35s呼吸);原间隙>2.5s 是自然停顿,新段落锚回原位;装得下语速恒1.0。"""
    from src.stages.d7_assemble import place_segments
    segs = [
        {"id": 0, "start": 0.0, "end": 2.8, "tts_dur": 3.2, "gender": "male"},   # 中文比原声长
        {"id": 1, "start": 3.0, "end": 5.0, "tts_dur": 1.5, "gender": "male"},   # 原点3.0与前句冲突
        {"id": 2, "start": 12.0, "end": 14.0, "tts_dur": 1.5, "gender": "male"}, # 原间隙>2.5→新段落
    ]
    pl = place_segments(segs, video_dur=20.0)
    assert pl[0]["start"] == 0.0
    assert pl[1]["start"] == 4.35            # 前句0+3.2/0.8+0.35呼吸:后让不重叠
    assert pl[2]["start"] == 12.0            # 大停顿保留,锚回原位(画面同步)
    assert pl[0]["factor"] == pl[1]["factor"] == 0.8   # 装得下→目标语速0.8(纪录片感)
    assert pl[2]["factor"] == 0.8
    assert all(p["cut"] is None for p in pl)


def test_place_segments_keeps_slack_at_original_positions():
    """有富余时句子不向左压实——空白留在原句间位置,消灭段尾死寂(用户:图像
    在说话没配音=v1把句子全提前了)。"""
    from src.stages.d7_assemble import place_segments
    segs = [
        {"id": 0, "start": 0.0, "end": 2.0, "tts_dur": 1.0, "gender": "male"},
        {"id": 1, "start": 2.2, "end": 4.4, "tts_dur": 1.0, "gender": "male"},
        {"id": 2, "start": 4.5, "end": 6.0, "tts_dur": 1.0, "gender": "male"},
    ]
    pl = place_segments(segs, video_dur=10.0)
    assert [p["start"] for p in pl] == [0.0, 2.2, 4.5]   # 全部贴原点,不提前
    assert all(p["factor"] == 0.8 for p in pl)           # 中文远短于原声:统一降速拉满


def test_assemble_cmd_vocals_sidechain_suppresses_residual_speech(tmp_path):
    """用户反馈(2026-07-04):配音间隙偶尔漏英文原声——原声 vocals 当侧链钥匙,
    原片说话处强压背景床(确定性灭残留),旁白 duck 链保持不变。"""
    from src.stages.d7_assemble import build_assemble_cmd
    vocals = tmp_path / "vocals.mp3"; vocals.write_bytes(b"V")
    cmd = build_assemble_cmd(tmp_path / "s.mp4", tmp_path / "nv.mp3", tmp_path / "narr.wav",
                             tmp_path / "subs.ass", tmp_path / "out.mp4", 0.5,
                             vocals=vocals)
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "[bg][3:a]sidechaincompress=threshold=0.02:ratio=12" in fc   # 人声钥匙强压
    assert "[bgv][2:a]sidechaincompress=threshold=0.03:ratio=8" in fc   # 旁白duck接力
    assert str(vocals) in cmd
    # 不传 vocals 走老链路
    cmd2 = build_assemble_cmd(tmp_path / "s.mp4", tmp_path / "nv.mp3", tmp_path / "narr.wav",
                              tmp_path / "subs.ass", tmp_path / "out.mp4", 0.5)
    fc2 = cmd2[cmd2.index("-filter_complex") + 1]
    assert "[bg][2:a]sidechaincompress" in fc2 and "bgv" not in fc2


def test_assemble_cmd_subs_only_keeps_original_audio(tmp_path):
    """字幕版(2026-07-04 用户定):原声完整直通,无背景床/旁白输入,只烧字幕。"""
    from src.stages.d7_assemble import build_assemble_cmd
    cmd = build_assemble_cmd(tmp_path / "s.mp4", tmp_path / "nv.mp3", None,
                             tmp_path / "subs.ass", tmp_path / "out.mp4", 0.5,
                             original_audio=True)
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "[0:a]anull[aout]" in fc            # 原声直通
    assert "sidechaincompress" not in fc       # 无duck链
    assert str(tmp_path / "nv.mp3") not in cmd # 背景床压根不进输入
    assert "subtitles=" in fc


def test_assemble_cmd_embeds_english_original_as_second_track(tmp_path):
    """双音轨(2026-07-05 用户定):配音版成片带 轨1中文(默认)+轨2英文原声;
    字幕版本就是原声单轨,不重复添加。"""
    from src.stages.d7_assemble import build_assemble_cmd
    cmd = build_assemble_cmd(tmp_path / "s.mp4", tmp_path / "nv.mp3", tmp_path / "narr.wav",
                             tmp_path / "subs.ass", tmp_path / "out.mp4", 0.5)
    joined = " ".join(cmd)
    assert "-map 0:a?" in joined
    assert "language=chi" in joined and "language=eng" in joined
    # 字幕版:单轨原声,无第二轨
    cmd2 = build_assemble_cmd(tmp_path / "s.mp4", tmp_path / "nv.mp3", None,
                              tmp_path / "subs.ass", tmp_path / "out.mp4", 0.5,
                              original_audio=True)
    assert "language=eng" not in " ".join(cmd2)
