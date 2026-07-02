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


def test_scopes_is_full_youtube_scope():
    """播放列表管理（playlists/playlistItems）需要完整 scope，仅 .upload 做不到。"""
    from src.youtube import SCOPES
    assert SCOPES == ["https://www.googleapis.com/auth/youtube"]


def test_fake_uploader_add_to_playlist_records_call():
    up = FakeUploader()
    pl_id = up.add_to_playlist("VID1", "太空探索/NASA")
    assert pl_id == "FAKEPL123"
    assert up.playlist_calls == [("VID1", "太空探索/NASA")]


def test_add_to_playlist_finds_existing():
    """标题已存在的播放列表：不新建，直接加视频进去。"""
    calls = {"insert_playlist": 0, "insert_item": None}

    class _Exec:
        def __init__(self, ret):
            self._ret = ret
        def execute(self):
            return self._ret

    class _Playlists:
        def list(self, **kw):
            return _Exec({"items": [{"id": "PL_EXIST", "snippet": {"title": "太空探索/NASA"}}]})
        def insert(self, **kw):
            calls["insert_playlist"] += 1
            return _Exec({"id": "PL_NEW"})

    class _PlaylistItems:
        def insert(self, **kw):
            calls["insert_item"] = kw
            return _Exec({"id": "PLI1"})

    class _YT:
        def playlists(self):
            return _Playlists()
        def playlistItems(self):
            return _PlaylistItems()

    up = YouTubeUploader.__new__(YouTubeUploader)  # 跳过会 build() 的 __init__
    up._yt = _YT()
    pl_id = up.add_to_playlist("VID1", "太空探索/NASA")

    assert pl_id == "PL_EXIST"
    assert calls["insert_playlist"] == 0            # 已存在，不该新建
    assert calls["insert_item"]["body"]["snippet"]["playlistId"] == "PL_EXIST"
    assert calls["insert_item"]["body"]["snippet"]["resourceId"]["videoId"] == "VID1"


def test_add_to_playlist_creates_when_missing():
    """标题不存在：先建播放列表（公开），再把视频插进去。"""
    calls = {"insert_playlist_body": None, "insert_item": None}

    class _Exec:
        def __init__(self, ret):
            self._ret = ret
        def execute(self):
            return self._ret

    class _Playlists:
        def list(self, **kw):
            return _Exec({"items": [{"id": "OTHER", "snippet": {"title": "别的列表"}}]})
        def insert(self, **kw):
            calls["insert_playlist_body"] = kw["body"]
            return _Exec({"id": "PL_NEW"})

    class _PlaylistItems:
        def insert(self, **kw):
            calls["insert_item"] = kw
            return _Exec({"id": "PLI2"})

    class _YT:
        def playlists(self):
            return _Playlists()
        def playlistItems(self):
            return _PlaylistItems()

    up = YouTubeUploader.__new__(YouTubeUploader)
    up._yt = _YT()
    pl_id = up.add_to_playlist("VID2", "太空探索/NASA")

    assert pl_id == "PL_NEW"
    assert calls["insert_playlist_body"]["snippet"]["title"] == "太空探索/NASA"
    assert calls["insert_playlist_body"]["status"]["privacyStatus"] == "public"
    assert calls["insert_item"]["body"]["snippet"]["playlistId"] == "PL_NEW"
    assert calls["insert_item"]["body"]["snippet"]["resourceId"]["videoId"] == "VID2"


def test_add_to_playlist_paginates_search():
    """播放列表超过一页时要翻页找全，别漏掉第 51 个之后的同名列表。"""
    pages = [
        {"items": [{"id": "PL1", "snippet": {"title": "别的"}}], "nextPageToken": "p2"},
        {"items": [{"id": "PL2", "snippet": {"title": "太空探索/NASA"}}]},
    ]
    seen_tokens = []

    class _Exec:
        def __init__(self, ret):
            self._ret = ret
        def execute(self):
            return self._ret

    class _Playlists:
        def list(self, **kw):
            seen_tokens.append(kw.get("pageToken"))
            return _Exec(pages[len(seen_tokens) - 1])
        def insert(self, **kw):
            raise AssertionError("不该新建，第二页已经找到同名列表了")

    class _PlaylistItems:
        def insert(self, **kw):
            return _Exec({"id": "PLI3"})

    class _YT:
        def playlists(self):
            return _Playlists()
        def playlistItems(self):
            return _PlaylistItems()

    up = YouTubeUploader.__new__(YouTubeUploader)
    up._yt = _YT()
    pl_id = up.add_to_playlist("VID3", "太空探索/NASA")

    assert pl_id == "PL2"
    assert seen_tokens == [None, "p2"]


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
