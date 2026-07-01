import json
import pytest
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.errors import StageBlocked
from src.util.jobs import write_json, read_json
from src.stages.s3_script import run, OUTPUT, INPUT


def _ctx(tmp_path, canned, research):
    ctx = Ctx(slug="c", config=Config({}), root=tmp_path, llm=FakeLLM(canned=canned))
    write_json(ctx.job, INPUT, research)
    return ctx


def test_run_writes_script_and_computes_stats(tmp_path):
    canned = json.dumps({
        "title_options": ["标题A", "标题B", "标题C"],
        "segments": [
            {"id": 1, "text": "一二三四五", "visual_query": "desert night", "est_sec": 5},
            {"id": 2, "text": "六七八九十", "visual_query": "old photo", "est_sec": 5},
        ],
    })
    ctx = _ctx(tmp_path, canned, {"blocked": False, "facts": []})
    run(ctx)
    out = read_json(ctx.job, OUTPUT)
    assert len(out["segments"]) == 2
    assert out["total_words"] == 10           # 5 + 5 个字
    assert out["est_minutes"] == round(10 / 300, 1)


def test_run_refuses_blocked_research(tmp_path):
    ctx = _ctx(tmp_path, "{}", {"blocked": True, "facts": []})
    with pytest.raises(StageBlocked):
        run(ctx)
    assert not (ctx.job / OUTPUT).exists()     # 不生成脚本
