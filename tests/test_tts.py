import pytest
from src.config import Config
from src.tts import FakeTTS, get_tts


def test_fake_tts_deterministic():
    tts = FakeTTS()
    data, words = tts.synth("飞碟")
    assert data  # 有音频字节
    assert [w["text"] for w in words] == ["飞", "碟"]
    assert words[0]["offset_sec"] == 0.0
    assert words[1]["offset_sec"] == 0.3
    assert all(w["dur_sec"] == 0.3 for w in words)


def test_get_tts_fake():
    assert isinstance(get_tts(Config({"tts": {"provider": "fake"}})), FakeTTS)


def test_get_tts_unknown_raises():
    with pytest.raises(ValueError):
        get_tts(Config({"tts": {"provider": "nope"}}))
