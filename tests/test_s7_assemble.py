from pathlib import Path
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.util.jobs import write_json
from src.stages.s7_assemble import build_cmd, run, OUTPUT, INPUT
import src.stages.s7_assemble as s7


def test_build_cmd_structure(tmp_path):
    segs = [
        {"id": 1, "audio": "seg_001.mp3", "dur_sec": 6.0},
        {"id": 2, "audio": "seg_002.mp3", "dur_sec": 7.0},
    ]
    cmd = build_cmd(tmp_path, segs, tmp_path / "subs.ass", fps=30, w=1920, h=1080)
    assert cmd[0] == "ffmpeg"
    assert cmd.count("-loop") == 2                        # 2 张图输入
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "concat=n=2:v=1:a=0" in fc                     # 拼 2 段视频
    assert "concat=n=2:v=0:a=1" in fc                     # 拼 2 段音频
    assert "subtitles=" in fc                             # 烧录字幕
    assert cmd[-1].endswith("video.mp4")
    assert "-t" in cmd and "6.0" in cmd and "7.0" in cmd  # 每段按 dur 展示


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
