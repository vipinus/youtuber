import pytest
from pathlib import Path
from src.config import Config
from src.youtube import FakeUploader, YouTubeUploader, get_uploader


def test_fake_uploader(tmp_path):
    up = FakeUploader()
    r = up.upload(tmp_path / "v.mp4", {"title": "t"}, tmp_path / "th.jpg", "private")
    assert r["videoId"] and r["status"] == "private" and "youtu" in r["url"]


def test_get_uploader_fake():
    assert isinstance(get_uploader(Config({"youtube": {"provider": "fake"}})), FakeUploader)


def test_get_uploader_unknown_raises():
    with pytest.raises(ValueError):
        get_uploader(Config({"youtube": {"provider": "nope"}}))


def test_load_creds_persists_refreshed_token(tmp_path, monkeypatch):
    """过期 token 刷新后必须回写磁盘，否则无人值守次日又得重新授权。"""
    import src.youtube as yt
    tokfile = tmp_path / "token.json"
    tokfile.write_text('{"stale": true}')

    written = {}

    class _Creds:
        expired = True
        refresh_token = "rt"
        def refresh(self, req): self.expired = False
        def to_json(self): return '{"refreshed": true}'

    class _CredsFactory:
        @staticmethod
        def from_authorized_user_file(path, scopes): return _Creds()

    # 拦截 google 惰性 import：注入假的 Credentials / Request
    import sys, types
    fake_oauth2 = types.ModuleType("google.oauth2.credentials")
    fake_oauth2.Credentials = _CredsFactory
    fake_req = types.ModuleType("google.auth.transport.requests")
    fake_req.Request = object
    monkeypatch.setitem(sys.modules, "google.oauth2.credentials", fake_oauth2)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", fake_req)

    yt.load_creds(str(tokfile), "cs.json")
    assert tokfile.read_text() == '{"refreshed": true}'   # 刷新后的 token 已落盘


def test_thumbnail_failure_keeps_video_id(tmp_path):
    """视频 insert 成功后封面 set 抛异常时，仍须返回 videoId（不能丢已上传的视频）。"""
    video = tmp_path / "v.mp4"; video.write_bytes(b"\x00" * 32)
    thumb = tmp_path / "th.jpg"; thumb.write_bytes(b"\xff" * 32)

    class _Exec:
        def __init__(self, ret=None, boom=False):
            self._ret, self._boom = ret, boom
        def execute(self):
            if self._boom:
                raise RuntimeError("403 thumbnail forbidden (channel not verified)")
            return self._ret

    class _YT:
        def videos(self):
            outer = self
            class _V:
                def insert(self, **kw):
                    return _Exec({"id": "VID99"})
            return _V()
        def thumbnails(self):
            class _T:
                def set(self, **kw):
                    return _Exec(boom=True)
            return _T()

    up = YouTubeUploader.__new__(YouTubeUploader)  # 跳过会 build() 的 __init__
    up._yt = _YT()
    r = up.upload(video, {"title": "t"}, thumb, "private")
    assert r["videoId"] == "VID99"
    assert r["url"] == "https://youtu.be/VID99"
    assert "thumbnail_error" in r and "403" in r["thumbnail_error"]
