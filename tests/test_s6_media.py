import json
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.util.jobs import write_json, read_json
from src.stages.s6_media import run, OUTPUT, INPUT


def _ctx(tmp_path):
    cfg = Config({"media": {"provider": "fake"}})
    ctx = Ctx(slug="c", config=cfg, root=tmp_path, llm=FakeLLM())
    write_json(ctx.job, INPUT, {"segments": [
        {"id": 1, "visual_query": "roswell desert"},
        {"id": 2, "visual_query": "phoenix lights"},
    ]})
    return ctx


def test_run_fetches_whitelisted_media(tmp_path):
    ctx = _ctx(tmp_path)
    run(ctx)
    credits = read_json(ctx.job, OUTPUT)
    assert len(credits["items"]) == 2
    assert all(c["license"] == "public-domain" for c in credits["items"])
    from PIL import Image
    with Image.open(ctx.job / "media" / "seg_001.jpg") as im:   # 合法 JPEG，下游 ffmpeg/PIL 可用
        assert im.format == "JPEG"
    assert (ctx.job / "media" / "seg_002.jpg").exists()


def test_run_fallback_on_fetch_error(tmp_path, monkeypatch):
    import src.stages.s6_media as s6

    class BoomSource:
        def search(self, q, limit=5):
            return [{"license": "public-domain", "source": "boom",
                     "attribution": "x", "manifest": "m", "ext": "jpg"}]
        def fetch(self, c, dest):
            raise RuntimeError("boom")

    monkeypatch.setattr(s6, "get_media_source", lambda cfg: BoomSource())
    ctx = _ctx(tmp_path)                       # reuse the helper from this test file
    s6.run(ctx)
    credits = read_json(ctx.job, s6.OUTPUT)
    assert credits["items"][0]["source"] == "fallback"     # fetch 崩了也兜底，不炸
    assert (ctx.job / "media" / "seg_001.jpg").exists()


def test_run_rejects_disallowed_license(tmp_path, monkeypatch):
    import src.stages.s6_media as s6

    class DenySource:
        def search(self, q, limit=5):
            return [{"license": "all-rights-reserved", "source": "deny",
                     "attribution": "x", "manifest": "m", "ext": "jpg"}]
        def fetch(self, c, dest):
            dest.write_bytes(b"SHOULD_NOT_HAPPEN")

    monkeypatch.setattr(s6, "get_media_source", lambda cfg: DenySource())
    ctx = _ctx(tmp_path)
    s6.run(ctx)
    credits = read_json(ctx.job, s6.OUTPUT)
    assert all(item["source"] == "fallback" for item in credits["items"])   # 非法license被拒→兜底
    assert (ctx.job / "media" / "seg_001.jpg").read_bytes() != b"SHOULD_NOT_HAPPEN"  # 没fetch非法源


def test_run_picks_first_allowed_over_disallowed(tmp_path, monkeypatch):
    import src.stages.s6_media as s6

    class MixSource:
        def search(self, q, limit=5):
            return [
                {"license": "all-rights-reserved", "source": "deny", "attribution": "x",
                 "manifest": "m", "ext": "jpg"},
                {"license": "cc-by", "source": "ok", "attribution": "y",
                 "manifest": "m2", "ext": "jpg"},
            ]
        def fetch(self, c, dest):
            dest.write_bytes(b"OKIMG")

    monkeypatch.setattr(s6, "get_media_source", lambda cfg: MixSource())
    ctx = _ctx(tmp_path)
    s6.run(ctx)
    credits = read_json(ctx.job, s6.OUTPUT)
    assert credits["items"][0]["source"] == "ok"       # 跳过非法、选首个合法(cc-by)
    assert credits["items"][0]["license"] == "cc-by"


def test_run_splits_compound_query_into_terms(tmp_path, monkeypatch):
    """visual_query 是逗号复合串时，整串搜不到、拆成子词能命中(首集实跑的坑)。"""
    import src.stages.s6_media as s6

    class OnlyExactTerm:
        def search(self, q, limit=5):
            if q == "weather balloon":                 # 只有这个精确子词有货
                return [{"license": "public-domain", "source": "nasa",
                         "attribution": "NASA / balloon", "manifest": "m", "ext": "jpg"}]
            return []
        def fetch(self, c, dest):
            dest.write_bytes(b"\xff\xd8\xff\xe0REAL")   # 假装真图

    cfg = Config({"media": {"provider": "fake"}})
    ctx = Ctx(slug="c", config=cfg, root=tmp_path, llm=FakeLLM())
    write_json(ctx.job, s6.INPUT, {"segments": [
        {"id": 1, "visual_query": "Old headlines 1947, weather balloon, desert New Mexico"},
    ]})
    monkeypatch.setattr(s6, "get_media_source", lambda c: OnlyExactTerm())
    s6.run(ctx)
    items = read_json(ctx.job, s6.OUTPUT)["items"]
    assert items[0]["source"] == "nasa"                # 拆词后命中真素材，不再兜底
    assert (ctx.job / "media" / "seg_001.jpg").read_bytes().startswith(b"\xff\xd8")


def test_terms_splits_and_strips():
    from src.stages.s6_media import _terms
    assert _terms("a, b ,  c") == ["a", "b", "c"]
    assert _terms("single") == ["single"]
    assert _terms("") == []
