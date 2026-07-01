import pytest
from pathlib import Path
from src.config import Config
from src.media import FakeMedia, get_media_source


def test_fake_media_search_and_fetch(tmp_path):
    m = FakeMedia()
    cands = m.search("roswell desert")
    assert cands and cands[0]["license"] == "public-domain"
    dest = tmp_path / "x.jpg"
    m.fetch(cands[0], dest)
    assert dest.read_bytes() == b"FAKEIMG"


def test_get_media_source_fake():
    assert isinstance(get_media_source(Config({"media": {"provider": "fake"}})), FakeMedia)


def test_get_media_source_unknown_raises():
    with pytest.raises(ValueError):
        get_media_source(Config({"media": {"provider": "nope"}}))
