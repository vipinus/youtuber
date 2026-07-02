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
    monkeypatch.delenv("A", raising=False)
    monkeypatch.delenv("B", raising=False)
    load_env(env)
    assert os.environ["A"] == "1"
    assert os.environ["B"] == "two"
    assert os.environ["C"] == "already"  # 不覆盖已有


def test_load_config(tmp_path):
    (tmp_path / "config.yaml").write_text("llm:\n  provider: fake\n", encoding="utf-8")
    cfg = load_config(tmp_path / "config.yaml")
    assert cfg.get("llm.provider") == "fake"


# ---------------------------------------------------------------------------
# P7 T14：config-zion.yaml —— zion 原生跑管线的专用 config，须能 yaml.safe_load
# 且 5 个 zion 专属键正确（跟 config.yaml 的差异点）。
# ---------------------------------------------------------------------------

def test_config_zion_yaml_parses_and_has_zion_keys():
    import yaml
    from pathlib import Path

    data = yaml.safe_load(Path("config-zion.yaml").read_text(encoding="utf-8"))

    assert data["remote"]["provider"] == "local"
    assert data["remote"]["wrapper"] == "~/gpu-py"
    assert data["p7"]["encoder"] == "nvenc"
    assert data["paths"]["title_font"] == "/home/fabric/.local/share/fonts/NotoSansCJK-Black.ttc"
    assert data["archive"]["dir"] == "~/nas/Share/youtuber"


def test_config_zion_yaml_loadable_via_load_config():
    cfg = load_config("config-zion.yaml")
    assert cfg.get("remote.provider") == "local"
    assert cfg.get("p7.encoder") == "nvenc"
