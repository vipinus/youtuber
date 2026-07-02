"""P7 T19：d9b_bilibili——YouTube 之外追加发布到 B 站。全离线，`subprocess.run`
一律 monkeypatch 掉，绝不真跑 biliup（也绝不真的 ssh/网络）。"""
from __future__ import annotations

import subprocess

import pytest

from src.config import Config
from src.context import Ctx
from src.llm import FakeLLM
from src.util.jobs import read_json, write_json

import src.stages.d9b_bilibili as d9b_bilibili
from src.stages.d9b_bilibili import OUTPUT, INPUT, run


def _ctx(tmp_path, cfg: Config, slug: str = "c") -> Ctx:
    ctx = Ctx(slug=slug, config=cfg, root=tmp_path / "jobs", llm=FakeLLM())
    write_json(ctx.job, INPUT, {
        "title": "t", "description": "d", "tags": ["x", "y"],
    })
    (ctx.job / "video.mp4").write_bytes(b"MP4")
    (ctx.job / "thumb.jpg").write_bytes(b"JPG")
    return ctx


def _fail_if_called(*a, **kw):
    raise AssertionError("subprocess 不该被调用——离线契约破了")


# ---------------------------------------------------------------------------
# 1. bilibili.enabled=false（默认）→ disabled，不碰 subprocess
# ---------------------------------------------------------------------------

def test_disabled_by_default_skips_without_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fail_if_called)
    ctx = _ctx(tmp_path, Config({}))

    run(ctx)

    out = read_json(ctx.job, OUTPUT)
    assert out == {"status": "disabled"}


def test_disabled_explicit_false_skips_without_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fail_if_called)
    ctx = _ctx(tmp_path, Config({"bilibili": {"enabled": False}}))

    run(ctx)

    out = read_json(ctx.job, OUTPUT)
    assert out["status"] == "disabled"


# ---------------------------------------------------------------------------
# 2. enabled 但 cookie 文件缺失（还没走一次性 QR 登录）→ no_login，不碰 subprocess
# ---------------------------------------------------------------------------

def test_enabled_but_no_cookie_warns_and_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fail_if_called)
    cookie_dir = tmp_path / "secrets_not_logged_in"     # 从不创建 cookies.json
    ctx = _ctx(tmp_path, Config({
        "bilibili": {"enabled": True, "cookie_dir": str(cookie_dir)},
    }))

    run(ctx)

    out = read_json(ctx.job, OUTPUT)
    assert out == {"status": "no_login"}
    assert ctx.warnings
    assert any("登录" in w for w in ctx.warnings)


# ---------------------------------------------------------------------------
# 3. happy path：cmd 形状正确 + cwd=cookie_dir + 解析 BV 号
# ---------------------------------------------------------------------------

def _make_cookie_dir(tmp_path):
    cookie_dir = tmp_path / "secrets"
    cookie_dir.mkdir()
    (cookie_dir / "cookies.json").write_text("{}", encoding="utf-8")
    return cookie_dir


def test_happy_path_cmd_shape_and_output(tmp_path, monkeypatch):
    cookie_dir = _make_cookie_dir(tmp_path)
    ctx = _ctx(tmp_path, Config({
        "bilibili": {"enabled": True, "cookie_dir": str(cookie_dir),
                     "biliup_bin": "/opt/bili/biliup", "tid": 201, "only_self": True},
    }))
    write_json(ctx.job, "source.json", {
        "identifier": "nasadoc1", "title": "NASA Doc",
        "source_url": "https://archive.org/download/nasadoc1/nasadoc1.mp4",
    })

    captured = {}

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["timeout"] = timeout
        return subprocess.CompletedProcess(
            cmd, 0, stdout="upload ok, bvid=BV1xx411c7mD\n", stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    run(ctx)

    cmd = captured["cmd"]
    assert cmd[0] == "/opt/bili/biliup"
    assert cmd[1] == "upload"
    assert cmd[2] == str(ctx.job / "video.mp4")
    assert "--title" in cmd and cmd[cmd.index("--title") + 1] == "t"
    assert "--desc" in cmd and cmd[cmd.index("--desc") + 1] == "d"
    assert "--tag" in cmd and cmd[cmd.index("--tag") + 1] == "x,y"
    assert "--cover" in cmd and cmd[cmd.index("--cover") + 1] == str(ctx.job / "thumb.jpg")
    assert "--tid" in cmd and cmd[cmd.index("--tid") + 1] == "201"
    assert "--copyright" in cmd and cmd[cmd.index("--copyright") + 1] == "2"
    assert "--source" in cmd
    assert cmd[cmd.index("--source") + 1] == "https://archive.org/download/nasadoc1/nasadoc1.mp4"
    assert "--is-only-self" in cmd and cmd[cmd.index("--is-only-self") + 1] == "1"

    assert captured["cwd"] == str(cookie_dir)
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["timeout"] == 3600     # 默认超时

    out = read_json(ctx.job, OUTPUT)
    assert out["status"] == "published"
    assert out["bvid"] == "BV1xx411c7mD"
    assert out["url"] == "https://www.bilibili.com/video/BV1xx411c7mD"
    assert out["only_self"] is True
    assert not ctx.warnings


def test_happy_path_no_source_json_falls_back_to_default_source(tmp_path, monkeypatch):
    """doc 管线理论上总有 source.json，但防御一下缺失情形：回退默认 --source。"""
    cookie_dir = _make_cookie_dir(tmp_path)
    ctx = _ctx(tmp_path, Config({"bilibili": {"enabled": True, "cookie_dir": str(cookie_dir)}}))
    assert not (ctx.job / "source.json").exists()

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout="no bv id here", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    run(ctx)

    out = read_json(ctx.job, OUTPUT)
    assert out["status"] == "published"
    assert out["bvid"] is None
    assert out["url"] is None


def test_line_flag_passed_when_configured(tmp_path, monkeypatch):
    cookie_dir = _make_cookie_dir(tmp_path)
    ctx = _ctx(tmp_path, Config({
        "bilibili": {"enabled": True, "cookie_dir": str(cookie_dir), "line": "bda2"},
    }))

    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run(ctx)

    cmd = captured["cmd"]
    assert "--line" in cmd and cmd[cmd.index("--line") + 1] == "bda2"


def test_only_self_false_passes_zero(tmp_path, monkeypatch):
    cookie_dir = _make_cookie_dir(tmp_path)
    ctx = _ctx(tmp_path, Config({
        "bilibili": {"enabled": True, "cookie_dir": str(cookie_dir), "only_self": False},
    }))

    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run(ctx)

    cmd = captured["cmd"]
    assert cmd[cmd.index("--is-only-self") + 1] == "0"
    out = read_json(ctx.job, OUTPUT)
    assert out["only_self"] is False


# ---------------------------------------------------------------------------
# 4. rc != 0 → failed，只 warn，绝不 raise
# ---------------------------------------------------------------------------

def test_nonzero_rc_marks_failed_and_warns_without_raising(tmp_path, monkeypatch):
    cookie_dir = _make_cookie_dir(tmp_path)
    ctx = _ctx(tmp_path, Config({"bilibili": {"enabled": True, "cookie_dir": str(cookie_dir)}}))

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="error: 审核未通过，标题违规",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    run(ctx)   # 不该抛异常

    out = read_json(ctx.job, OUTPUT)
    assert out["status"] == "failed"
    assert "审核未通过" in out["error"]
    assert ctx.warnings


def test_subprocess_exception_marks_failed_and_warns_without_raising(tmp_path, monkeypatch):
    """biliup 二进制不存在/超时等 subprocess.run 自身抛异常的情形——同样只
    留标记，不能让 run 崩掉（YouTube 已经发布成功了）。"""
    cookie_dir = _make_cookie_dir(tmp_path)
    ctx = _ctx(tmp_path, Config({"bilibili": {"enabled": True, "cookie_dir": str(cookie_dir)}}))

    def fake_run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 3600)

    monkeypatch.setattr(subprocess, "run", fake_run)

    run(ctx)   # 不该抛异常

    out = read_json(ctx.job, OUTPUT)
    assert out["status"] == "failed"
    assert ctx.warnings


# ---------------------------------------------------------------------------
# 5. clamp：标题 >80 字截断，tags 每个 >20 字截断 + 总数 >10 截断
# ---------------------------------------------------------------------------

def test_title_over_80_chars_is_clamped(tmp_path, monkeypatch):
    cookie_dir = _make_cookie_dir(tmp_path)
    ctx = _ctx(tmp_path, Config({"bilibili": {"enabled": True, "cookie_dir": str(cookie_dir)}}))
    long_title = "标" * 120
    write_json(ctx.job, INPUT, {"title": long_title, "description": "d", "tags": []})

    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run(ctx)

    cmd = captured["cmd"]
    sent_title = cmd[cmd.index("--title") + 1]
    assert len(sent_title) == 80
    assert sent_title == long_title[:80]


def test_tags_over_10_and_over_20_chars_each_are_clamped(tmp_path, monkeypatch):
    cookie_dir = _make_cookie_dir(tmp_path)
    ctx = _ctx(tmp_path, Config({"bilibili": {"enabled": True, "cookie_dir": str(cookie_dir)}}))
    long_tag = "签" * 30
    many_tags = [f"tag{i}" for i in range(15)] + [long_tag]
    write_json(ctx.job, INPUT, {"title": "t", "description": "d", "tags": many_tags})

    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run(ctx)

    cmd = captured["cmd"]
    sent_tags = cmd[cmd.index("--tag") + 1].split(",")
    assert len(sent_tags) <= 10
    assert all(len(t) <= 20 for t in sent_tags)


def test_source_prefers_details_page(tmp_path, monkeypatch):
    """--source 优先 archive.org /details/ 页(利于B站审核),非直接下载链。"""
    import src.stages.d9b_bilibili as d9b
    from src.config import Config
    from src.context import Ctx
    from src.llm import FakeLLM
    from src.util.jobs import write_json
    import subprocess as sp

    seen = {}
    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        class R: returncode = 0; stdout = "BV1abcdefgh12"; stderr = ""
        return R()
    monkeypatch.setattr(sp, "run", fake_run)

    cookie_dir = tmp_path / "sec"; cookie_dir.mkdir(); (cookie_dir / "cookies.json").write_text("{}")
    cfg = Config({"bilibili": {"enabled": True, "cookie_dir": str(cookie_dir),
                                "biliup_bin": "/bin/biliup"}})
    ctx = Ctx(slug="c", config=cfg, root=tmp_path, llm=FakeLLM())
    write_json(ctx.job, "meta.json", {"title": "t", "description": "d", "tags": ["x"]})
    write_json(ctx.job, "source.json", {"identifier": "SomeDoc123",
                                         "source_url": "https://archive.org/download/SomeDoc123/f.mp4"})
    d9b.run(ctx)
    i = seen["cmd"].index("--source")
    assert seen["cmd"][i+1] == "https://archive.org/details/SomeDoc123"
