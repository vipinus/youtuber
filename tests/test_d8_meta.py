from __future__ import annotations

import json

from PIL import Image

from src.config import Config
from src.context import Ctx
from src.llm import FakeLLM
from src.util.jobs import read_json, write_json
from src.stages.d8_meta import (
    BGM_INPUT,
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


# ---------------------------------------------------------------------------
# BGM 署名追加 —— job 有 bgm.json(d7_assemble replace 模式写的)就追加 CC-BY
# 署名行；没有就静默跳过(STAGES_P7 里 d7_assemble 在 d8_meta 之前跑，见
# src/run.py：正常链路 d7 会先落 bgm.json，这里同时兜底缺失场景)。
# ---------------------------------------------------------------------------

def test_run_appends_bgm_attribution_when_job_bgm_json_exists(tmp_path):
    ctx = _ctx(tmp_path, _CANNED)
    write_json(ctx.job, BGM_INPUT, {
        "file": "Frozen_Star.mp3", "title": "Frozen Star", "artist": "Kevin MacLeod",
        "license": "CC-BY-4.0",
        "attribution": "背景音乐：Frozen Star - Kevin MacLeod (incompetech.com), CC BY 4.0",
    })
    run(ctx)
    meta = read_json(ctx.job, OUTPUT)
    assert "NASA / Apollo 11" in meta["description"]      # 源片署名还在
    assert "背景音乐：Frozen Star - Kevin MacLeod (incompetech.com), CC BY 4.0" in meta["description"]


def test_run_skips_bgm_attribution_when_job_bgm_json_absent(tmp_path):
    ctx = _ctx(tmp_path, _CANNED)                          # 没写 bgm.json(original 模式/回退场景)
    run(ctx)
    meta = read_json(ctx.job, OUTPUT)
    assert "背景音乐" not in meta["description"]


def test_run_bgm_attribution_survives_malformed_bgm_json(tmp_path):
    """bgm.json 存在但损坏/无 attribution 字段——不该崩最后一阶段，静默跳过。"""
    ctx = _ctx(tmp_path, _CANNED)
    (ctx.job / BGM_INPUT).write_text("{not valid json", encoding="utf-8")
    run(ctx)                                              # 不应抛出
    meta = read_json(ctx.job, OUTPUT)
    assert "背景音乐" not in meta["description"]


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


def test_wrap_title_splits_long_title_no_truncation():
    """长标题折两行不丢字(真跑发现 title[:14] 硬截断切掉了后半句)。"""
    from src.stages.d8_meta import _wrap_title
    t = "地球巨眼！阿雷西博望远镜：窥探宇宙奥秘的奇迹"
    lines = _wrap_title(t)
    assert len(lines) == 2
    assert "".join(lines) == t                      # 不丢一个字
    assert lines[0].endswith("：")                   # 在标点处断
    assert _wrap_title("短标题") == ["短标题"]        # 短标题单行


def test_d8_survives_sdxl_failure_falls_back(tmp_path, monkeypatch):
    """zion/SDXL 挂了不该中止整条 run:回退纯色底封面 + 仍产出 meta.json,视频照发。"""
    import src.stages.d8_meta as d8
    from src.config import Config
    from src.context import Ctx
    from src.llm import FakeLLM
    from src.util.jobs import write_json
    from PIL import Image
    import json

    class _BoomRemote:
        def generate_image(self, prompt, out): raise RuntimeError("zion down")
        def transcribe_separate(self, a, o): raise AssertionError

    canned = json.dumps({"title": "测试标题", "description": "描述", "tags": ["a"],
                         "image_prompt": "space"})
    ctx = Ctx(slug="c", config=Config({"remote": {"provider": "fake"}}),
              root=tmp_path, llm=FakeLLM(canned=canned))
    write_json(ctx.job, d8.INPUT, {"duration": 5, "segments": [{"id": 0, "start": 0,
               "end": 2, "en": "hi", "zh": "你好"}]})
    write_json(ctx.job, "source.json", {"title": "X", "attribution": "NASA / X"})
    monkeypatch.setattr(d8, "get_remote", lambda cfg: _BoomRemote())
    d8.run(ctx)                                   # 不应抛异常
    assert (ctx.job / "thumb.jpg").exists()
    assert Image.open(ctx.job / "thumb.jpg").size == (1280, 720)
    assert any("SDXL" in w for w in ctx.warnings)
