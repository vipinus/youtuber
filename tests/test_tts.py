import pytest
from src.config import Config
from src.tts import EdgeTTS, FakeTTS, get_tts


def test_fake_tts_deterministic():
    tts = FakeTTS()
    data, words = tts.synth("飞碟")
    assert data[:4] == b"RIFF"        # 合法 WAV，ffprobe 可读
    assert [w["text"] for w in words] == ["飞", "碟"]
    assert words[0]["offset_sec"] == 0.0
    assert words[1]["offset_sec"] == 0.3
    assert all(w["dur_sec"] == 0.3 for w in words)


def test_get_tts_fake():
    assert isinstance(get_tts(Config({"tts": {"provider": "fake"}})), FakeTTS)


def test_get_tts_unknown_raises():
    with pytest.raises(ValueError):
        get_tts(Config({"tts": {"provider": "nope"}}))


# ---------------------------------------------------------------------------
# P7 T9 — get_tts 的 voice= 覆盖参数（性别匹配配音：按段挑男/女声引擎）
# ---------------------------------------------------------------------------

def test_get_tts_fake_ignores_voice_override():
    """provider=fake 时无论传不传 voice 都还是 FakeTTS——离线测试永远不触网。"""
    tts = get_tts(Config({"tts": {"provider": "fake"}}), voice="zh-CN-XiaoxiaoNeural")
    assert isinstance(tts, FakeTTS)


def test_get_tts_edge_voice_override_wins_over_config():
    tts = get_tts(
        Config({"tts": {"provider": "edge", "voice": "zh-CN-YunyangNeural"}}),
        voice="zh-CN-XiaoxiaoNeural",
    )
    assert isinstance(tts, EdgeTTS)
    assert tts.voice == "zh-CN-XiaoxiaoNeural"


def test_get_tts_edge_no_override_falls_back_to_config_voice():
    tts = get_tts(Config({"tts": {"provider": "edge", "voice": "zh-CN-YunyangNeural"}}))
    assert tts.voice == "zh-CN-YunyangNeural"
