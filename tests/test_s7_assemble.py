import subprocess
from pathlib import Path

import pytest

from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.util.jobs import write_json, read_json
from src.stages.s7_assemble import build_cmd, resolve_media, run, OUTPUT, INPUT, CREDITS
import src.stages.s7_assemble as s7


def test_build_cmd_structure(tmp_path):
    segs = [
        {"id": 1, "audio": "seg_001.mp3", "dur_sec": 6.0},
        {"id": 2, "audio": "seg_002.mp3", "dur_sec": 7.0},
    ]
    media = {1: "seg_001.jpg", 2: "seg_002.jpg"}
    cmd = build_cmd(tmp_path, segs, media, tmp_path / "subs.ass", fps=30, w=1920, h=1080)
    assert cmd[0] == "ffmpeg"
    assert cmd.count("-loop") == 2                        # 2 张图输入
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "concat=n=2:v=1:a=0" in fc                     # 拼 2 段视频
    assert "concat=n=2:v=0:a=1" in fc                     # 拼 2 段音频
    assert "subtitles=" in fc                             # 烧录字幕
    assert cmd[-1].endswith("video.mp4")
    assert "-t" in cmd and "6.0" in cmd and "7.0" in cmd  # 每段按 dur 展示


def test_build_cmd_filter_chain_has_scale_crop_setsar_fps_setpts(tmp_path):
    segs = [{"id": 1, "audio": "seg_001.mp3", "dur_sec": 3.0}]
    media = {1: "seg_001.jpg"}
    cmd = build_cmd(tmp_path, segs, media, tmp_path / "subs.ass", fps=25, w=640, h=480)
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "scale=640:480:force_original_aspect_ratio=increase" in fc
    assert "crop=640:480" in fc
    assert "setsar=1" in fc
    assert "fps=25" in fc
    assert "setpts=PTS-STARTPTS" in fc                    # 干净时间戳，供 concat 用
    assert "aresample=44100" in fc


def test_build_cmd_image_segment_uses_loop(tmp_path):
    segs = [{"id": 1, "audio": "seg_001.mp3", "dur_sec": 4.0}]
    media = {1: "seg_001.jpg"}
    cmd = build_cmd(tmp_path, segs, media, tmp_path / "subs.ass", fps=30, w=1920, h=1080)
    img_path = str(tmp_path / "media" / "seg_001.jpg")
    idx = cmd.index("-loop")
    assert cmd[idx:idx + 6] == ["-loop", "1", "-t", "4.0", "-i", img_path]
    assert "-stream_loop" not in cmd


def test_build_cmd_video_segment_uses_stream_loop(tmp_path):
    segs = [{"id": 2, "audio": "seg_002.mp3", "dur_sec": 5.0}]
    media = {2: "seg_002.mp4"}
    cmd = build_cmd(tmp_path, segs, media, tmp_path / "subs.ass", fps=30, w=1920, h=1080)
    vid_path = str(tmp_path / "media" / "seg_002.mp4")
    idx = cmd.index("-stream_loop")
    assert cmd[idx:idx + 6] == ["-stream_loop", "-1", "-t", "5.0", "-i", vid_path]
    assert "-loop" not in cmd                             # 视频段不应出现图片式 -loop 1


def test_build_cmd_mixed_image_and_video(tmp_path):
    segs = [
        {"id": 1, "audio": "seg_001.mp3", "dur_sec": 3.0},
        {"id": 2, "audio": "seg_002.mp3", "dur_sec": 4.0},
    ]
    media = {1: "seg_001.jpg", 2: "seg_002.mp4"}
    cmd = build_cmd(tmp_path, segs, media, tmp_path / "subs.ass", fps=30, w=1920, h=1080)
    assert cmd.count("-loop") == 1
    assert cmd.count("-stream_loop") == 1
    assert str(tmp_path / "media" / "seg_001.jpg") in cmd
    assert str(tmp_path / "media" / "seg_002.mp4") in cmd
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "concat=n=2:v=1:a=0" in fc
    assert "concat=n=2:v=0:a=1" in fc


def test_resolve_media_reads_credits_json(tmp_path):
    write_json(tmp_path, CREDITS, {"items": [
        {"id": 1, "file": "seg_001.jpg", "source": "nasa"},
        {"id": 2, "file": "seg_002.mp4", "source": "archive"},
    ]})
    segs = [{"id": 1, "dur_sec": 1.0}, {"id": 2, "dur_sec": 1.0}]
    media = resolve_media(tmp_path, segs)
    assert media == {1: "seg_001.jpg", 2: "seg_002.mp4"}


def test_resolve_media_missing_credits_falls_back_to_jpg(tmp_path):
    segs = [{"id": 1, "dur_sec": 1.0}, {"id": 2, "dur_sec": 1.0}]
    media = resolve_media(tmp_path, segs)                 # 无 credits.json
    assert media == {1: "seg_001.jpg", 2: "seg_002.jpg"}


def test_resolve_media_partial_credits_falls_back_per_segment(tmp_path):
    write_json(tmp_path, CREDITS, {"items": [
        {"id": 1, "file": "seg_001.mp4", "source": "archive"},
        # id 2 缺失
    ]})
    segs = [{"id": 1, "dur_sec": 1.0}, {"id": 2, "dur_sec": 1.0}]
    media = resolve_media(tmp_path, segs)
    assert media == {1: "seg_001.mp4", 2: "seg_002.jpg"}   # 缺的段兜底 jpg


def test_run_invokes_ffmpeg(tmp_path, monkeypatch):
    ctx = Ctx(slug="c", config=Config({"video": {"fps": 30}}), root=tmp_path, llm=FakeLLM())
    write_json(ctx.job, INPUT, {"segments": [
        {"id": 1, "audio": "seg_001.mp3", "dur_sec": 1.0},
    ], "total_sec": 1.0})
    (ctx.job / "subs.ass").write_text("[Events]\n", encoding="utf-8")

    calls = {}

    def fake_run(cmd, **kw):
        calls["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"MP4")                 # 假装 ffmpeg 出了文件
        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(s7.subprocess, "run", fake_run)
    run(ctx)
    assert "ffmpeg" in calls["cmd"][0]
    assert calls["cmd"][-1] == str(ctx.job / OUTPUT)
    assert (ctx.job / OUTPUT).exists()
    assert "-loop" in calls["cmd"]                        # 无 credits.json→兜底图片


def test_run_uses_credits_json_for_mixed_media(tmp_path, monkeypatch):
    """run() 从 credits.json 解析每段真实文件（图或视频），而不是硬编码 .jpg。"""
    ctx = Ctx(slug="c", config=Config({"video": {"fps": 30}}), root=tmp_path, llm=FakeLLM())
    write_json(ctx.job, INPUT, {"segments": [
        {"id": 1, "audio": "seg_001.mp3", "dur_sec": 2.0},
        {"id": 2, "audio": "seg_002.mp3", "dur_sec": 3.0},
    ], "total_sec": 5.0})
    write_json(ctx.job, CREDITS, {"items": [
        {"id": 1, "file": "seg_001.jpg", "source": "nasa"},
        {"id": 2, "file": "seg_002.mp4", "source": "archive"},
    ]})
    (ctx.job / "subs.ass").write_text("[Events]\n", encoding="utf-8")

    calls = {}

    def fake_run(cmd, **kw):
        calls["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"MP4")
        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(s7.subprocess, "run", fake_run)
    run(ctx)
    cmd = calls["cmd"]
    assert str(ctx.job / "media" / "seg_001.jpg") in cmd
    assert str(ctx.job / "media" / "seg_002.mp4") in cmd
    assert "-stream_loop" in cmd and "-loop" in cmd


def test_run_raises_with_ffmpeg_stderr(tmp_path, monkeypatch):
    ctx = Ctx(slug="c", config=Config({"video": {"fps": 30}}), root=tmp_path, llm=FakeLLM())
    write_json(ctx.job, INPUT, {"segments": [{"id": 1, "audio": "seg_001.mp3", "dur_sec": 1.0}], "total_sec": 1.0})
    (ctx.job / "subs.ass").write_text("[Events]\n", encoding="utf-8")

    def fake_run(cmd, **kw):
        class R:
            returncode = 1
            stderr = "boom: invalid filtergraph"
            stdout = ""
        return R()

    monkeypatch.setattr(s7.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as ei:
        run(ctx)
    assert "boom: invalid filtergraph" in str(ei.value)     # ffmpeg 错误被抛出来


def test_real_ffmpeg_smoke_assembles_mixed_image_video(tmp_path):
    """真实 ffmpeg 冒烟：1 图 + 1 视频段拼接，验证真能产出可读的视频文件。
    刻意用极小分辨率/极短时长/极低帧率，保持离线、亚秒级，不进网络。"""
    ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True)
    if ffmpeg.returncode != 0:
        pytest.skip("ffmpeg not available")

    job = tmp_path
    media_dir = job / "media"
    audio_dir = job / "audio"
    media_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)

    from PIL import Image
    Image.new("RGB", (32, 32), (200, 50, 50)).save(media_dir / "seg_001.jpg")

    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=32x32:rate=5",
         "-t", "1", "-pix_fmt", "yuv420p", str(media_dir / "seg_002.mp4")],
        check=True,
    )
    for name in ("seg_001.mp3", "seg_002.mp3"):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
             "-t", "1", str(audio_dir / name)],
            check=True,
        )
    (job / "subs.ass").write_text(
        "[Script Info]\n[Events]\nFormat: Layer, Start, End, Text\n", encoding="utf-8",
    )

    segs = [
        {"id": 1, "audio": "seg_001.mp3", "dur_sec": 1.0},
        {"id": 2, "audio": "seg_002.mp3", "dur_sec": 1.0},
    ]
    media = {1: "seg_001.jpg", 2: "seg_002.mp4"}
    cmd = build_cmd(job, segs, media, job / "subs.ass", fps=5, w=64, h=48)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = job / OUTPUT
    assert out.exists() and out.stat().st_size > 0

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    )
    assert probe.stdout.strip() == "64,48"
