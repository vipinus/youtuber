from __future__ import annotations

import json

from PIL import Image

from src.config import Config
from src.context import Ctx
from src.llm import FakeLLM
from src.util.jobs import read_json, write_json
from src.stages.d8_meta import (
    OUTPUT,
    INPUT,
    SOURCE_INPUT,
    make_thumbnail,
    run,
)


def _ctx(tmp_path, canned, cfg_extra: dict | None = None):
    cfg_data = {"remote": {"provider": "fake"}}
    if cfg_extra:
        cfg_data.update(cfg_extra)
    ctx = Ctx(slug="c", config=Config(cfg_data), root=tmp_path, llm=FakeLLM(canned=canned))
    write_json(ctx.job, INPUT, {"duration": 5.0, "segments": [
        {"id": 0, "start": 0.0, "end": 2.0, "en": "Hello there.", "zh": "你好。"},
        {"id": 1, "start": 2.0, "end": 5.0, "en": "The end.", "zh": "结束。"},
    ]})
    write_json(ctx.job, SOURCE_INPUT, {
        "identifier": "nasadoc1",
        "title": "Apollo 11",
        "license": "public-domain",
        "attribution": "NASA / Apollo 11",
        "duration": 600,
    })
    return ctx


_CANNED = json.dumps({
    "title": "阿波罗11号：登月真相",
    "description": "一段震撼的太空旅程，重返历史现场。",
    "tags": ["NASA", "阿波罗", "登月"],
    "image_prompt": "a dramatic astronaut standing on the moon, no text",
})


# ---------------------------------------------------------------------------
# run() —— 端到端：不 ssh/不触网(remote=fake)，产出 meta.json + thumb.jpg
# ---------------------------------------------------------------------------

def test_run_writes_meta_and_thumb(tmp_path):
    ctx = _ctx(tmp_path, _CANNED)
    run(ctx)

    meta = read_json(ctx.job, OUTPUT)
    assert meta["title"] == "阿波罗11号：登月真相"
    assert meta["tags"] == ["NASA", "阿波罗", "登月"]
    assert "NASA / Apollo 11" in meta["description"]     # 署名追加
    assert "公有领域" in meta["description"]              # public-domain -> 中文标签

    thumb = ctx.job / "thumb.jpg"
    assert thumb.exists() and thumb.stat().st_size > 0
    assert Image.open(thumb).size == (1280, 720)


def test_run_generates_background_via_remote(tmp_path):
    """image_prompt 真的喂给了 remote.generate_image（FakeRemoteGPU 落盘一张
    真实可解码 PNG），不是空跑。"""
    ctx = _ctx(tmp_path, _CANNED)
    run(ctx)
    bg = ctx.job / "thumb_bg.png"
    assert bg.exists() and bg.stat().st_size > 0
    assert Image.open(bg).size == (1280, 720)


def test_run_falls_back_to_source_title_when_llm_omits_title(tmp_path):
    canned = json.dumps({"description": "无标题", "tags": []})
    ctx = _ctx(tmp_path, canned)
    run(ctx)
    meta = read_json(ctx.job, OUTPUT)
    assert meta["title"] == "Apollo 11"                  # 回退到 source.json 的标题


def test_run_no_attribution_when_source_missing(tmp_path):
    """source.json 缺失（理论上不该发生，但别崩）——没有署名可加，只是不追加。"""
    ctx = Ctx(slug="c", config=Config({"remote": {"provider": "fake"}}), root=tmp_path,
               llm=FakeLLM(canned=_CANNED))
    write_json(ctx.job, INPUT, {"duration": 1.0, "segments": []})
    run(ctx)
    meta = read_json(ctx.job, OUTPUT)
    assert "素材来源" not in meta["description"]


def test_run_no_network_no_ssh(tmp_path, monkeypatch):
    """头号回归风险：d8_meta 必须能在 remote=fake 下完全离线跑——任何 ssh/scp/
    真实 HTTP 请求都判失败。"""
    import subprocess as sp

    def _boom_subprocess(cmd, *a, **k):
        if cmd and cmd[0] in ("ssh", "scp"):
            raise AssertionError(f"d8_meta 不该发起 {cmd[0]}: {cmd}")
        raise AssertionError(f"d8_meta 不该调用 subprocess: {cmd}")

    monkeypatch.setattr(sp, "run", _boom_subprocess)

    def _boom_requests(*a, **k):
        raise AssertionError("d8_meta 不该发起真实 HTTP 请求")

    monkeypatch.setattr("requests.get", _boom_requests, raising=False)

    ctx = _ctx(tmp_path, _CANNED)
    run(ctx)                              # 不应抛出
    assert (ctx.job / OUTPUT).exists()


# ---------------------------------------------------------------------------
# make_thumbnail —— 合成健壮性：背景/水印任一缺失或损坏都不该崩最后一阶段
# ---------------------------------------------------------------------------

def test_make_thumbnail_survives_corrupt_bg(tmp_path):
    bad_bg = tmp_path / "bg.png"
    bad_bg.write_bytes(b"NOT-A-PNG-\x00\xff garbage")
    dest = tmp_path / "thumb.jpg"
    make_thumbnail(dest, "太空探索", bad_bg, tmp_path / "watermark.png")
    assert dest.exists() and Image.open(dest).size == (1280, 720)


def test_make_thumbnail_survives_missing_watermark(tmp_path):
    bg = tmp_path / "bg.png"
    Image.new("RGB", (1280, 720), (20, 20, 40)).save(bg)
    dest = tmp_path / "thumb.jpg"
    make_thumbnail(dest, "太空探索", bg, tmp_path / "nonexistent_watermark.png")
    assert dest.exists() and Image.open(dest).size == (1280, 720)


def test_make_thumbnail_applies_real_watermark_and_bg(tmp_path):
    """跟真实素材一起跑一遍：assets/watermark_tr.png 真实存在，产出应比纯底稍大
    (贴了水印+渐变)。主要验证不抛异常、尺寸正确。"""
    from pathlib import Path

    bg = tmp_path / "bg.png"
    Image.new("RGB", (1280, 720), (30, 10, 60)).save(bg)
    dest = tmp_path / "thumb.jpg"
    watermark = Path("assets/watermark_tr.png")
    make_thumbnail(dest, "深空探索纪录片", bg, watermark)
    assert dest.exists() and Image.open(dest).size == (1280, 720)


def test_make_thumbnail_long_title_does_not_crash(tmp_path):
    dest = tmp_path / "thumb.jpg"
    make_thumbnail(dest, "一个超级超级超级长的中文标题用来测试自动缩字号不崩溃", None, None)
    assert dest.exists() and Image.open(dest).size == (1280, 720)


# ---------------------------------------------------------------------------
# import 时不触网
# ---------------------------------------------------------------------------

def test_import_does_no_network():
    import ast
    import inspect

    import src.stages.d8_meta as mod

    banned = {"requests", "paramiko", "fabric"}
    tree = ast.parse(inspect.getsource(mod))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] not in banned for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in banned
