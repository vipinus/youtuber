import pytest
from src.config import Config
from src.llm import FakeLLM, get_llm


def test_fake_llm_records_and_returns():
    llm = FakeLLM(canned="hello")
    out = llm.generate("sys", "prompt")
    assert out == "hello"
    assert llm.calls == [("sys", "prompt")]


def test_get_llm_fake():
    llm = get_llm(Config({"llm": {"provider": "fake"}}))
    assert isinstance(llm, FakeLLM)


def test_get_llm_unknown_raises():
    with pytest.raises(ValueError):
        get_llm(Config({"llm": {"provider": "nope"}}))


def test_get_llm_claude_not_implemented():
    with pytest.raises(NotImplementedError):
        get_llm(Config({"llm": {"provider": "claude"}}))


def test_gemini_retries_transient_503(monkeypatch):
    """瞬时 503 应退避重试后成功；非瞬时错误立即抛出。"""
    import src.llm as llm
    monkeypatch.setattr("time.sleep", lambda *a: None)      # 不真睡
    calls = {"n": 0}

    class _Boom(Exception):
        code = 503

    class _Models:
        @staticmethod
        def generate_content(**kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _Boom("503 UNAVAILABLE high demand")
            class R: text = "剧本OK"
            return R()

    class _Client:
        models = _Models()

    g = llm.GeminiLLM.__new__(llm.GeminiLLM)               # 跳过会联网的 __init__
    g._client, g._model = _Client(), "m"
    assert g.generate("sys", "p") == "剧本OK"
    assert calls["n"] == 3                                  # 前两次 503，第三次成功


def test_gemini_non_transient_raises_immediately(monkeypatch):
    import src.llm as llm
    monkeypatch.setattr("time.sleep", lambda *a: None)
    calls = {"n": 0}

    class _Auth(Exception):
        code = 401

    class _Models:
        @staticmethod
        def generate_content(**kw):
            calls["n"] += 1
            raise _Auth("401 unauthorized")

    class _Client:
        models = _Models()

    g = llm.GeminiLLM.__new__(llm.GeminiLLM)
    g._client, g._model = _Client(), "m"
    import pytest
    with pytest.raises(_Auth):
        g.generate("sys", "p")
    assert calls["n"] == 1                                  # 非瞬时错误不重试
