import json
import pytest
from pathlib import Path
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.stages.s1_topic import load_seed, extract_json, run, OUTPUT


SEEDS = """- slug: roswell-1947
  title_working: 罗斯威尔事件
  year: 1947
"""


def _seeds_file(tmp_path) -> Path:
    p = tmp_path / "cases.yaml"
    p.write_text(SEEDS, encoding="utf-8")
    return p


def test_load_seed_found(tmp_path):
    seed = load_seed("roswell-1947", _seeds_file(tmp_path))
    assert seed["year"] == 1947


def test_load_seed_missing_raises(tmp_path):
    with pytest.raises(KeyError):
        load_seed("does-not-exist", _seeds_file(tmp_path))


def test_extract_json_fenced_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('好的，结果是：{"a": 2} 完毕') == {"a": 2}


def test_run_writes_topic_json(tmp_path):
    seeds = _seeds_file(tmp_path)
    canned = json.dumps({"title_working": "罗斯威尔事件", "year": 1947})
    cfg = Config({"paths": {"seeds": str(seeds)}})
    ctx = Ctx(slug="roswell-1947", config=cfg, root=tmp_path / "jobs",
              llm=FakeLLM(canned=canned))
    run(ctx)
    out = json.loads((ctx.job / OUTPUT).read_text(encoding="utf-8"))
    assert out["slug"] == "roswell-1947"      # run 补上的
    assert out["year"] == 1947                 # LLM 返回的
