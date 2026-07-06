"""P7 d9d:中英简介双向互链(2026-07-06)。全离线,provider=fake。"""
from __future__ import annotations

from src.config import Config
from src.context import Ctx
from src.llm import FakeLLM
from src.stages import d9d_crosslink
from src.util.jobs import read_json, write_json
import src.youtube as ym


def _ctx(tmp_path, cfg=None):
    ctx = Ctx(slug="c", config=Config(cfg or {"youtube": {"provider": "fake"},
                                              "youtube_en": {"provider": "fake"}}),
              root=tmp_path / "jobs", llm=FakeLLM())
    return ctx


def test_crosslink_updates_both_descriptions(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    write_json(ctx.job, "publish.json", {"videoId": "ZH1", "url": "https://youtu.be/ZH1"})
    write_json(ctx.job, "publish_en.json", {"videoId": "EN1", "url": "https://youtu.be/EN1"})
    write_json(ctx.job, "meta.json", {"description": "中文简介"})
    write_json(ctx.job, "source.json", {"identifier": "yt:x", "title": "T", "source_url": "u"})

    updates: list = []
    monkeypatch.setattr(ym.FakeUploader, "update_description",
                        lambda self, vid, desc: updates.append((vid, desc)))

    d9d_crosslink.run(ctx)

    by = {v: d for v, d in updates}
    assert "https://youtu.be/EN1" in by["ZH1"]      # 中文视频 += 英文链接
    assert "中文简介" in by["ZH1"]                   # 保留原简介
    assert "https://youtu.be/ZH1" in by["EN1"]      # 英文视频 += 中文链接
    out = read_json(ctx.job, d9d_crosslink.OUTPUT)
    assert out["status"] == "done" and set(out["linked"]) == {"zh", "en"}


def test_crosslink_skips_when_one_side_missing(tmp_path):
    ctx = _ctx(tmp_path)
    write_json(ctx.job, "publish.json", {"videoId": "ZH1", "url": "https://youtu.be/ZH1"})
    # 无 publish_en.json
    d9d_crosslink.run(ctx)
    out = read_json(ctx.job, d9d_crosslink.OUTPUT)
    assert out["status"] == "skipped"


def test_crosslink_idempotent_no_double_append(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    write_json(ctx.job, "publish.json", {"videoId": "ZH1", "url": "https://youtu.be/ZH1"})
    write_json(ctx.job, "publish_en.json", {"videoId": "EN1", "url": "https://youtu.be/EN1"})
    # 中文简介里已含英文链接 → 不重复追加
    write_json(ctx.job, "meta.json", {"description": "中文简介 https://youtu.be/EN1"})
    write_json(ctx.job, "source.json", {"identifier": "yt:x", "title": "T", "source_url": "u"})
    updates: list = []
    monkeypatch.setattr(ym.FakeUploader, "update_description",
                        lambda self, vid, desc: updates.append((vid, desc)))
    d9d_crosslink.run(ctx)
    assert "ZH1" not in [v for v, _ in updates]      # 已含,跳过
