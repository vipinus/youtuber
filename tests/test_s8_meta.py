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


def test_make_thumbnail_survives_corrupt_bg(tmp_path):
    """素材是非法/损坏字节时，make_thumbnail 回退纯色底而不崩（生产：NASA 返回非图）。"""
    from src.stages.s8_meta import make_thumbnail
    from PIL import Image
    bad = tmp_path / "media" / "seg_001.jpg"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"NOT-A-JPEG-\x00\xff garbage")
    dest = tmp_path / "thumb.jpg"
    make_thumbnail(dest, "罗斯威尔之谜", bad)          # 不应抛 UnidentifiedImageError
    assert dest.exists() and Image.open(dest).size == (1280, 720)


def test_make_thumbnail_survives_missing_font(tmp_path, monkeypatch):
    """字体文件缺失(换机器)时用兜底字体渲染，不崩 OSError。"""
    import src.stages.s8_meta as s8
    from PIL import Image
    monkeypatch.setattr(s8, "TITLE_FONT", "/nonexistent/font.ttc")
    dest = tmp_path / "thumb.jpg"
    s8.make_thumbnail(dest, "罗斯威尔之谜", tmp_path / "nope.jpg")   # 字体+底图都缺
    assert dest.exists() and Image.open(dest).size == (1280, 720)


# ---------------------------------------------------------------------------
# P7 T14：可配置字体路径(config `paths.title_font`)
# ---------------------------------------------------------------------------

def test_make_thumbnail_font_path_param_missing_falls_back_to_default(tmp_path):
    """font_path 参数传了一个不存在的路径时也走 _font 的 OSError 兜底，不崩。"""
    from src.stages.s8_meta import make_thumbnail
    from PIL import Image
    dest = tmp_path / "thumb.jpg"
    make_thumbnail(dest, "罗斯威尔之谜", tmp_path / "nope.jpg", font_path="/nonexistent/configured.ttc")
    assert dest.exists() and Image.open(dest).size == (1280, 720)


def test_run_threads_configured_title_font_into_make_thumbnail(tmp_path, monkeypatch):
    """ctx.config 的 paths.title_font 要真的传给 make_thumbnail，而不是被忽略。"""
    import src.stages.s8_meta as s8
    from PIL import Image

    captured = {}

    def fake_make_thumbnail(dest, title, bg, font_path=None):
        captured["font_path"] = font_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1280, 720)).save(dest, "JPEG")

    monkeypatch.setattr(s8, "make_thumbnail", fake_make_thumbnail)

    canned = json.dumps({"title": "标题", "description": "desc", "tags": []})
    ctx = Ctx(slug="c", config=Config({"paths": {"title_font": "/custom/configured-font.ttc"}}),
              root=tmp_path, llm=FakeLLM(canned=canned))
    write_json(ctx.job, INPUT, {"title_options": ["x"], "segments": []})
    write_json(ctx.job, "credits.json", {"items": []})
    (ctx.job / "media").mkdir(parents=True, exist_ok=True)

    s8.run(ctx)
    assert captured["font_path"] == "/custom/configured-font.ttc"


def test_run_defaults_font_to_title_font_constant_when_unconfigured(tmp_path, monkeypatch):
    """没配 paths.title_font 时回退模块常量 TITLE_FONT，不是 None。"""
    import src.stages.s8_meta as s8
    from PIL import Image

    captured = {}

    def fake_make_thumbnail(dest, title, bg, font_path=None):
        captured["font_path"] = font_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1280, 720)).save(dest, "JPEG")

    monkeypatch.setattr(s8, "make_thumbnail", fake_make_thumbnail)

    canned = json.dumps({"title": "标题", "description": "desc", "tags": []})
    ctx = Ctx(slug="c", config=Config({}), root=tmp_path, llm=FakeLLM(canned=canned))
    write_json(ctx.job, INPUT, {"title_options": ["x"], "segments": []})
    write_json(ctx.job, "credits.json", {"items": []})
    (ctx.job / "media").mkdir(parents=True, exist_ok=True)

    s8.run(ctx)
    assert captured["font_path"] == s8.TITLE_FONT
