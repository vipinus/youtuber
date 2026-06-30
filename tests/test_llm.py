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
