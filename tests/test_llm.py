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


def test_get_llm_ollama_factory():
    from src.config import Config
    from src.llm import get_llm, OllamaLLM
    llm = get_llm(Config({"llm": {"provider": "ollama", "ollama_host": "http://eros:11434/",
                                    "ollama_model": "gemma4:31b-cloud"}}))
    assert isinstance(llm, OllamaLLM)
    assert llm.host == "http://eros:11434" and llm.model == "gemma4:31b-cloud"


def test_ollama_generate_parses_and_4xx_no_retry(monkeypatch):
    import src.llm as llm_mod
    calls = {"n": 0}

    class _R:
        status_code = 200
        text = ""
        def json(self): return {"message": {"content": "你好"}}
        def raise_for_status(self): pass

    class _R403:
        status_code = 403
        text = "requires a subscription"
        def json(self): return {}
        def raise_for_status(self):
            raise RuntimeError("403 Client Error: Forbidden")

    def _post_ok(*a, **k):
        calls["n"] += 1
        return _R()
    monkeypatch.setattr("time.sleep", lambda *a: None)
    import types, sys as _sys
    fake_req = types.ModuleType("requests")
    fake_req.post = _post_ok
    monkeypatch.setitem(_sys.modules, "requests", fake_req)
    o = llm_mod.OllamaLLM("http://x:1", "m")
    assert o.generate("s", "p") == "你好"

    calls["n"] = 0
    def _post_403(*a, **k):
        calls["n"] += 1
        return _R403()
    fake_req.post = _post_403
    import pytest
    with pytest.raises(RuntimeError):
        o.generate("s", "p")
    assert calls["n"] == 1          # 4xx 不重试(订阅墙/权限,重试无意义)


def test_transient_includes_connection_errors():
    from src.llm import _is_transient
    class ConnectionError(Exception): pass
    class ReadTimeout(Exception): pass
    assert _is_transient(ConnectionError("boom"))
    assert _is_transient(ReadTimeout("slow"))
