import json
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.util.jobs import write_json, read_json
from src.stages.s8_meta import run, OUTPUT, INPUT


def _ctx(tmp_path, canned):
    ctx = Ctx(slug="c", config=Config({}), root=tmp_path, llm=FakeLLM(canned=canned))
    write_json(ctx.job, INPUT, {"title_options": ["罗斯威尔之谜"],
                                "segments": [{"id": 1, "text": "飞碟坠落。"}]})
    write_json(ctx.job, "credits.json", {"items": [
        {"id": 1, "source": "nasa", "license": "public-domain", "attribution": "NASA / Roswell"},
        {"id": 2, "source": "fallback", "license": "public-domain", "attribution": "fallback"},
    ]})
    (ctx.job / "media").mkdir(parents=True, exist_ok=True)
    return ctx


def test_run_writes_meta_and_thumb(tmp_path):
    canned = json.dumps({"title": "罗斯威尔：军方隐瞒了什么？",
                         "description": "1947年的真相。", "tags": ["UFO", "罗斯威尔"]})
    ctx = _ctx(tmp_path, canned)
    run(ctx)
    meta = read_json(ctx.job, OUTPUT)
    assert meta["title"] == "罗斯威尔：军方隐瞒了什么？"
    assert "NASA / Roswell" in meta["description"]     # 署名追加
    assert "fallback" not in meta["description"]        # 兜底不署名
    thumb = ctx.job / "thumb.jpg"
    assert thumb.exists() and thumb.stat().st_size > 0
    from PIL import Image
    assert Image.open(thumb).size == (1280, 720)
