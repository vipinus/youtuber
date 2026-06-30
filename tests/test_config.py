import os
from pathlib import Path
from src.config import Config, load_env, load_config


def test_get_dotted_and_default():
    cfg = Config({"llm": {"provider": "gemini"}, "video": {"fps": 30}})
    assert cfg.get("llm.provider") == "gemini"
    assert cfg.get("video.fps") == 30
    assert cfg.get("llm.missing", "x") == "x"
    assert cfg.get("nope.deep.path") is None


def test_load_env_no_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('A=1\n# c\nB="two"\nC=keep\n', encoding="utf-8")
    monkeypatch.setenv("C", "already")
    load_env(env)
    assert os.environ["A"] == "1"
    assert os.environ["B"] == "two"
    assert os.environ["C"] == "already"  # 不覆盖已有


def test_load_config(tmp_path):
    (tmp_path / "config.yaml").write_text("llm:\n  provider: fake\n", encoding="utf-8")
    cfg = load_config(tmp_path / "config.yaml")
    assert cfg.get("llm.provider") == "fake"
