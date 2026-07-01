from pathlib import Path
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.util.jobs import write_json
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
