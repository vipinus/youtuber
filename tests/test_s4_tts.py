import json
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.util.jobs import write_json, read_json
from src.stages.s4_tts import run, OUTPUT, INPUT


def _ctx(tmp_path, script):
    cfg = Config({"tts": {"provider": "fake"}})
    ctx = Ctx(slug="c", config=cfg, root=tmp_path, llm=FakeLLM())
    write_json(ctx.job, INPUT, script)
    return ctx


def test_run_writes_timing_and_audio(tmp_path):
    script = {"segments": [
        {"id": 1, "text": "飞碟"},     # 2 字 → 0.6s
        {"id": 2, "text": "外星人"},   # 3 字 → 0.9s
    ]}
    ctx = _ctx(tmp_path, script)
    run(ctx)
    timing = read_json(ctx.job, OUTPUT)
    assert len(timing["segments"]) == 2
    s1, s2 = timing["segments"]
    assert s1["start_sec"] == 0.0
    assert round(s2["start_sec"], 1) == 0.6           # 累積：段2 起点 = 段1 时长
    assert timing["total_sec"] == round(0.6 + 0.9, 3)
    # 全局字时间：段2 首字从 0.6 开始
    assert s2["words"][0]["start_sec"] == 0.6
    assert (ctx.job / "audio" / "seg_001.mp3").exists()
    assert (ctx.job / "audio" / "seg_002.mp3").exists()
