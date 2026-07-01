import json
import pytest
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.errors import StageBlocked
from src.util.jobs import write_json, read_json
from src.stages.s2_research import run, summarize, OUTPUT, INPUT


def _ctx(tmp_path, canned, cfg=None):
    ctx = Ctx(slug="c", config=cfg or Config({}), root=tmp_path,
              llm=FakeLLM(canned=canned))
    write_json(ctx.job, INPUT, {"slug": "c", "title_working": "案例"})
    return ctx


def test_summarize_counts():
    facts = [{"evidence": "documented"}, {"evidence": "documented"},
             {"evidence": "claim"}, {"evidence": "speculation"}, {"evidence": "x"}]
    assert summarize(facts) == {"documented": 2, "claim": 1, "speculation": 1}


def test_run_normal_writes_research(tmp_path):
    canned = json.dumps({"facts": [
        {"claim": "军方发过飞碟新闻稿", "evidence": "documented", "source": "存档"},
        {"claim": "Mogul 气球计划", "evidence": "documented", "source": "报告"},
        {"claim": "有非人类残骸", "evidence": "claim", "source": "证词"},
    ]})
    ctx = _ctx(tmp_path, canned)
    run(ctx)
    out = read_json(ctx.job, OUTPUT)
    assert out["blocked"] is False
    assert out["evidence_summary"] == {"documented": 2, "claim": 1, "speculation": 0}


def test_run_blocked_when_no_documented(tmp_path):
    canned = json.dumps({"facts": [
        {"claim": "据说有外星人", "evidence": "claim", "source": "论坛"},
    ]})
    ctx = _ctx(tmp_path, canned)
    with pytest.raises(StageBlocked):
        run(ctx)
    out = read_json(ctx.job, OUTPUT)   # 产物仍写下（记录为何 blocked）
    assert out["blocked"] is True


def test_run_warns_below_threshold(tmp_path):
    canned = json.dumps({"facts": [
        {"claim": "一条有记录的事实", "evidence": "documented", "source": "存档"},
    ]})
    ctx = _ctx(tmp_path, canned, cfg=Config({"guardrails": {"min_documented_facts": 3}}))
    run(ctx)
    out = read_json(ctx.job, OUTPUT)
    assert out["blocked"] is False
    assert ctx.warnings                 # 有 warning
    assert out.get("warnings")          # 产物里也记了


def test_run_malformed_facts_fail_closed(tmp_path):
    ctx = _ctx(tmp_path, json.dumps({"facts": "oops-not-a-list"}))
    with pytest.raises(StageBlocked):
        run(ctx)
    out = read_json(ctx.job, OUTPUT)
    assert out["blocked"] is True
    assert out["evidence_summary"] == {"documented": 0, "claim": 0, "speculation": 0}
