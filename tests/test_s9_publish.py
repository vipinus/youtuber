from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.util.jobs import write_json, read_json
from src.youtube import FakeUploader
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


def test_run_adds_to_playlist_when_configured(tmp_path, monkeypatch):
    """publish.playlist 配了 → 上传成功后自动入列，publish.json 记 playlist/playlistId。"""
    cfg = Config({"youtube": {"provider": "fake"},
                  "publish": {"auto_upload": True, "visibility": "private",
                              "playlist": "太空探索/NASA"}})
    ctx = _ctx(tmp_path, cfg)

    fake = FakeUploader()
    monkeypatch.setattr("src.stages.s9_publish.get_uploader", lambda cfg: fake)

    run(ctx)
    pub = read_json(ctx.job, OUTPUT)
    assert pub["videoId"] == "FAKE12345"
    assert pub["playlist"] == "太空探索/NASA"
    assert pub["playlistId"] == "FAKEPL123"
    assert fake.playlist_calls == [("FAKE12345", "太空探索/NASA")]   # 假上传器真收到了调用


def test_run_skips_playlist_when_unset(tmp_path, monkeypatch):
    """publish.playlist 没配（默认空）→ 完全不碰播放列表接口。"""
    cfg = Config({"youtube": {"provider": "fake"},
                  "publish": {"auto_upload": True, "visibility": "private"}})
    ctx = _ctx(tmp_path, cfg)

    fake = FakeUploader()
    monkeypatch.setattr("src.stages.s9_publish.get_uploader", lambda cfg: fake)

    run(ctx)
    pub = read_json(ctx.job, OUTPUT)
    assert "playlist" not in pub and "playlistId" not in pub
    assert fake.playlist_calls == []


class _RaisingPlaylistUploader:
    """模拟老 token 缺 scope：上传成功，但 add_to_playlist 403。"""

    def upload(self, video, meta, thumb, visibility):
        return {"videoId": "VID_OK", "url": "https://youtu.be/VID_OK", "status": visibility}

    def add_to_playlist(self, video_id, playlist_title):
        raise RuntimeError(
            "403 insufficientPermissions: request had insufficient authentication scopes")


def test_run_playlist_failure_keeps_video_id_and_warns(tmp_path, monkeypatch):
    """播放列表失败绝不能丢已上传的视频；权限类报错要带重新授权提示。"""
    cfg = Config({"youtube": {"provider": "fake"},
                  "publish": {"auto_upload": True, "visibility": "private",
                              "playlist": "太空探索/NASA"}})
    ctx = _ctx(tmp_path, cfg)

    monkeypatch.setattr("src.stages.s9_publish.get_uploader",
                         lambda cfg: _RaisingPlaylistUploader())

    run(ctx)
    pub = read_json(ctx.job, OUTPUT)
    assert pub["videoId"] == "VID_OK"                  # 视频本体没丢
    assert "playlist" not in pub and "playlistId" not in pub
    assert "playlist_error" in pub
    assert "insufficient" in pub["playlist_error"].lower()
    assert "auth_youtube" in pub["playlist_error"]      # 重新授权提示
    assert ctx.warnings
    assert any("auth_youtube" in w for w in ctx.warnings)
