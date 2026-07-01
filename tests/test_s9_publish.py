from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.util.jobs import write_json, read_json
from src.stages.s9_publish import run, OUTPUT, INPUT


def _ctx(tmp_path, cfg):
    ctx = Ctx(slug="c", config=cfg, root=tmp_path, llm=FakeLLM())
    write_json(ctx.job, INPUT, {"title": "t", "description": "d", "tags": ["x"]})
    (ctx.job / "video.mp4").write_bytes(b"MP4")
    (ctx.job / "thumb.jpg").write_bytes(b"JPG")
    return ctx


def test_run_uploads_when_enabled(tmp_path):
    cfg = Config({"youtube": {"provider": "fake"},
                  "publish": {"auto_upload": True, "visibility": "private"}})
    ctx = _ctx(tmp_path, cfg)
    run(ctx)
    pub = read_json(ctx.job, OUTPUT)
    assert pub["videoId"] == "FAKE12345"
    assert pub["status"] == "private"          # 护栏#3 默认私享


def test_run_skips_when_auto_upload_off(tmp_path):
    cfg = Config({"youtube": {"provider": "fake"}, "publish": {"auto_upload": False}})
    ctx = _ctx(tmp_path, cfg)
    run(ctx)
    pub = read_json(ctx.job, OUTPUT)
    assert pub["status"] == "skipped"          # 关了就不传
    assert ctx.warnings                        # 有 warn


def test_run_skipped_records_local_paths(tmp_path):
    """护栏#3: 关闭上传时仍需落盘本地产物路径，方便人工复核/后续手动发布。"""
    cfg = Config({"youtube": {"provider": "fake"}, "publish": {"auto_upload": False}})
    ctx = _ctx(tmp_path, cfg)
    run(ctx)
    pub = read_json(ctx.job, OUTPUT)
    assert pub["video"] == str(ctx.job / "video.mp4")
    assert pub["meta"] == str(ctx.job / INPUT)


def test_run_defaults_to_private_when_visibility_unset(tmp_path):
    """护栏#3: auto_upload 打开但没写 visibility 时，绝不能默认公开。"""
    cfg = Config({"youtube": {"provider": "fake"}, "publish": {"auto_upload": True}})
    ctx = _ctx(tmp_path, cfg)
    run(ctx)
    pub = read_json(ctx.job, OUTPUT)
    assert pub["status"] == "private"


def test_run_never_defaults_auto_upload_to_true(tmp_path):
    """护栏#3: 冷启动安全档 —— publish 配置缺失时也不能偷偷上传。"""
    cfg = Config({"youtube": {"provider": "fake"}})
    ctx = _ctx(tmp_path, cfg)
    run(ctx)
    pub = read_json(ctx.job, OUTPUT)
    assert pub["status"] == "skipped"
