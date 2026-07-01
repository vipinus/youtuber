from pathlib import Path
import json
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.util.jobs import write_json, read_json
from src.errors import StageBlocked
import src.run as run_mod


def test_run_pipeline_skips_when_output_exists(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_stage(ctx: Ctx) -> None:
        calls["n"] += 1
        write_json(ctx.job, "fake.json", {"ok": True})

    monkeypatch.setattr(run_mod, "STAGES", [("fake", fake_stage, "fake.json")])

    args = ("slugX", Config({}), tmp_path, FakeLLM())
    run_mod.run_pipeline(*args)
    assert calls["n"] == 1                       # 第一次跑
    run_mod.run_pipeline(*args)
    assert calls["n"] == 1                       # 产物已存在 → 跳过
    run_mod.run_pipeline(*args, force=True)
    assert calls["n"] == 2                       # --force → 重跑


def test_run_pipeline_from_to_subrange(tmp_path, monkeypatch):
    log = []

    def mk(name):
        def fn(ctx: Ctx) -> None:
            log.append(name)
            write_json(ctx.job, f"{name}.json", {})
        return fn

    monkeypatch.setattr(run_mod, "STAGES", [
        ("a", mk("a"), "a.json"),
        ("b", mk("b"), "b.json"),
        ("c", mk("c"), "c.json"),
    ])
    run_mod.run_pipeline("s", Config({}), tmp_path, FakeLLM(), frm="b", to="b")
    assert log == ["b"]


def test_run_pipeline_stops_on_blocked(tmp_path, monkeypatch):
    ran = []

    def blocker(ctx):
        ran.append("a")
        raise StageBlocked("nope")

    def after(ctx):
        ran.append("b")
        write_json(ctx.job, "b.json", {})

    monkeypatch.setattr(run_mod, "STAGES", [
        ("a", blocker, "a.json"),
        ("b", after, "b.json"),
    ])
    run_mod.run_pipeline("s", Config({}), tmp_path, FakeLLM())
    assert ran == ["a"]                       # b 未运行（被 blocked 停下）


class SeqLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.i = 0

    def generate(self, system, prompt):
        r = self.responses[self.i]
        self.i += 1
        return r


def test_end_to_end_topic_research_script(tmp_path):
    topic = json.dumps({"title_working": "罗斯威尔", "year": 1947})
    research = json.dumps({"facts": [
        {"claim": "军方新闻稿", "evidence": "documented", "source": "存档"},
        {"claim": "Mogul", "evidence": "documented", "source": "报告"},
        {"claim": "Mogul2", "evidence": "documented", "source": "报告"},
    ]})
    script = json.dumps({"title_options": ["a", "b", "c"], "segments": [
        {"id": 1, "text": "一二三四五六七八九十", "visual_query": "q", "est_sec": 8},
    ]})
    run_mod.run_pipeline("roswell-1947", Config({}), tmp_path,
                         SeqLLM([topic, research, script]))
    job = tmp_path / "roswell-1947"
    assert read_json(job, "topic.json")["year"] == 1947
    assert read_json(job, "research.json")["blocked"] is False
    assert read_json(job, "script.json")["total_words"] == 10


def test_e2e_through_subtitles(tmp_path):
    import json
    from src.util.jobs import read_json
    topic = json.dumps({"title_working": "罗斯威尔", "year": 1947})
    research = json.dumps({"facts": [
        {"claim": "a", "evidence": "documented", "source": "s"},
        {"claim": "b", "evidence": "documented", "source": "s"},
        {"claim": "c", "evidence": "documented", "source": "s"},
    ]})
    script = json.dumps({"title_options": ["t"], "segments": [
        {"id": 1, "text": "飞碟来了。", "visual_query": "q", "est_sec": 3},
    ]})
    cfg = Config({"tts": {"provider": "fake"}})
    run_mod.run_pipeline("roswell-1947", cfg, tmp_path, SeqLLM([topic, research, script]))
    job = tmp_path / "roswell-1947"
    assert read_json(job, "timing.json")["segments"][0]["words"][0]["text"] == "飞"
    assert "Dialogue:" in (job / "subs.ass").read_text(encoding="utf-8")
    assert (job / "audio" / "seg_001.mp3").exists()
