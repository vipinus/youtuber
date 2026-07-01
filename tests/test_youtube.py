import pytest
from pathlib import Path
from src.config import Config
from src.youtube import FakeUploader, get_uploader


def test_fake_uploader(tmp_path):
    up = FakeUploader()
    r = up.upload(tmp_path / "v.mp4", {"title": "t"}, tmp_path / "th.jpg", "private")
    assert r["videoId"] and r["status"] == "private" and "youtu" in r["url"]


def test_get_uploader_fake():
    assert isinstance(get_uploader(Config({"youtube": {"provider": "fake"}})), FakeUploader)


def test_get_uploader_unknown_raises():
    with pytest.raises(ValueError):
        get_uploader(Config({"youtube": {"provider": "nope"}}))
