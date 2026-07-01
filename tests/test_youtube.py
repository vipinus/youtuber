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
