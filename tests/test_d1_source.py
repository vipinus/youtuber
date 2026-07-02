from __future__ import annotations

import ast
import inspect
import json

import pytest

from src.config import Config
from src.context import Ctx
from src.llm import FakeLLM
from src.util.jobs import read_json


# ---------------------------------------------------------------------------
# 假 requests.Response
# ---------------------------------------------------------------------------

class FakeResp:
    def __init__(self, json_data=None, content: bytes = b""):
        self._json = json_data
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return self._json

    def iter_content(self, chunk_size=1 << 16):
        yield self.content


def _search_docs(*docs) -> dict:
    return {"response": {"docs": list(docs)}}


def _ctx(tmp_path, cfg_data=None):
    cfg = Config(cfg_data or {})
    return Ctx(slug="doc1", config=cfg, root=tmp_path, llm=FakeLLM())


def _install_fake_get(monkeypatch, search_docs: list[dict], metadata_by_id: dict[str, dict],
                       download_content: bytes = b"\x00\x00\x00\x18ftypmp42" + b"X" * 300_000):
    """按 URL 分派：advancedsearch -> search_docs；metadata/<id> -> metadata_by_id[id]；
    download/<id>/<file> -> download_content。"""

    def fake_get(url, params=None, headers=None, timeout=None, stream=False):
        assert headers and "User-Agent" in headers          # 护栏：必须带 UA
        if "advancedsearch.php" in url:
            return FakeResp(json_data=_search_docs(*search_docs))
        if "/metadata/" in url:
            identifier = url.rsplit("/", 1)[-1]
            return FakeResp(json_data=metadata_by_id.get(identifier, {}))
        if "/download/" in url:
            return FakeResp(content=download_content)
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr("requests.get", fake_get)


# ---------------------------------------------------------------------------
# 护栏#2 —— license / 可信集合
# ---------------------------------------------------------------------------

def test_rejects_untrusted_or_no_license_candidate_and_picks_next(tmp_path, monkeypatch):
    import src.stages.d1_source as d1

    search_docs = [
        {"identifier": "bad1", "title": "Bad Copyrighted Doc"},
        {"identifier": "good1", "title": "Good NASA Doc"},
    ]
    metadata = {
        "bad1": {
            "metadata": {"licenseurl": "https://rightsstatements.org/vocab/InC/1.0/",
                         "collection": ["nasa"], "title": "Bad Copyrighted Doc"},
            "files": [{"name": "bad1.mp4", "size": "300000", "length": "600",
                       "width": "640", "height": "480"}],
        },
        "good1": {
            "metadata": {"licenseurl": None, "collection": ["nasa"], "title": "Good NASA Doc"},
            "files": [{"name": "good1.mp4", "size": "300000", "length": "600",
                       "width": "640", "height": "480"}],
        },
    }
    _install_fake_get(monkeypatch, search_docs, metadata)

    ctx = _ctx(tmp_path)
    d1.run(ctx)

    meta = read_json(ctx.job, d1.META_OUTPUT)
    assert meta["identifier"] == "good1"                  # bad1(明确版权)被拒，选下一个
    assert meta["license"] == "public-domain"


# ---------------------------------------------------------------------------
# 内容安全黑名单
# ---------------------------------------------------------------------------

def test_rejects_content_blocked_doc_even_if_pd(tmp_path, monkeypatch):
    import src.stages.d1_source as d1

    search_docs = [
        {"identifier": "death1", "title": "Death Mills"},
        {"identifier": "safe1", "title": "Apollo 11 Mission Report"},
    ]
    metadata = {
        "death1": {
            "metadata": {"licenseurl": "http://creativecommons.org/licenses/publicdomain/",
                         "collection": ["usgovfilms"], "title": "Death Mills",
                         "subject": "Holocaust concentration camp"},
            "files": [{"name": "death1.mp4", "size": "300000", "length": "600",
                       "width": "640", "height": "480"}],
        },
        "safe1": {
            "metadata": {"licenseurl": "http://creativecommons.org/licenses/publicdomain/",
                         "collection": ["nasa"], "title": "Apollo 11 Mission Report"},
            "files": [{"name": "safe1.mp4", "size": "300000", "length": "600",
                       "width": "640", "height": "480"}],
        },
    }
    _install_fake_get(monkeypatch, search_docs, metadata)

    ctx = _ctx(tmp_path)
    d1.run(ctx)

    meta = read_json(ctx.job, d1.META_OUTPUT)
    assert meta["identifier"] == "safe1"                   # Death Mills 命中黑名单，即便PD也拒


# ---------------------------------------------------------------------------
# 时长范围 + darkened(空files) 跳过
# ---------------------------------------------------------------------------

def test_rejects_out_of_duration_and_skips_darkened_item(tmp_path, monkeypatch):
    import src.stages.d1_source as d1

    search_docs = [
        {"identifier": "toolong", "title": "Too Long Doc"},
        {"identifier": "darkened", "title": "Darkened Doc"},
        {"identifier": "justright", "title": "Just Right Doc"},
    ]
    metadata = {
        "toolong": {
            "metadata": {"licenseurl": "http://creativecommons.org/licenses/publicdomain/",
                         "collection": ["nasa"], "title": "Too Long Doc"},
            "files": [{"name": "toolong.mp4", "size": "300000", "length": "5000",
                       "width": "640", "height": "480"}],   # 超过 max_sec
        },
        "darkened": {
            "metadata": {"licenseurl": "http://creativecommons.org/licenses/publicdomain/",
                         "collection": ["nasa"], "title": "Darkened Doc"},
            "files": [],                                     # 已下架，空files
        },
        "justright": {
            "metadata": {"licenseurl": "http://creativecommons.org/licenses/publicdomain/",
                         "collection": ["nasa"], "title": "Just Right Doc"},
            "files": [{"name": "justright.mp4", "size": "300000", "length": "900",
                       "width": "640", "height": "480"}],
        },
    }
    _install_fake_get(monkeypatch, search_docs, metadata)

    ctx = _ctx(tmp_path, {"p7": {"min_sec": 300, "max_sec": 1800}})
    d1.run(ctx)

    meta = read_json(ctx.job, d1.META_OUTPUT)
    assert meta["identifier"] == "justright"
    assert meta["duration"] == 900.0


# ---------------------------------------------------------------------------
# 选文件：高分辨率优先 + 写 source.mp4/source.json
# ---------------------------------------------------------------------------

def test_picks_highest_resolution_mp4_and_writes_source(tmp_path, monkeypatch):
    import src.stages.d1_source as d1

    search_docs = [{"identifier": "hdshow", "title": "Space Wonders"}]
    metadata = {
        "hdshow": {
            "metadata": {"licenseurl": "http://creativecommons.org/licenses/publicdomain/",
                         "collection": ["nasa"], "title": "Space Wonders"},
            "files": [
                {"name": "hdshow_512kb.mp4", "size": "300000", "length": "900",
                 "width": "320", "height": "240"},
                {"name": "hdshow.mp4", "size": "400000", "length": "900",
                 "width": "1280", "height": "720"},           # 最高分辨率
                {"name": "hdshow_tiny.mp4", "size": "50000", "length": "0.04"},  # 占位坏档
            ],
        },
    }
    content = b"\x00\x00\x00\x18ftypmp42" + b"Y" * 400_000
    _install_fake_get(monkeypatch, search_docs, metadata, download_content=content)

    ctx = _ctx(tmp_path)
    d1.run(ctx)

    meta = read_json(ctx.job, d1.META_OUTPUT)
    assert meta["identifier"] == "hdshow"
    assert meta["resolution"] == "1280x720"
    assert meta["source_url"].endswith("/hdshow/hdshow.mp4")
    assert meta["attribution"] == "NASA / Space Wonders"
    assert meta["license"] == "public-domain"
    assert meta["duration"] == 900.0

    dest = ctx.job / d1.OUTPUT
    assert dest.exists()
    assert dest.read_bytes() == content


def test_attribution_falls_back_to_internet_archive_when_not_nasa(tmp_path, monkeypatch):
    import src.stages.d1_source as d1

    search_docs = [{"identifier": "prelinger1", "title": "Old Space Reel"}]
    metadata = {
        "prelinger1": {
            "metadata": {"licenseurl": "http://creativecommons.org/licenses/publicdomain/",
                         "collection": ["prelinger"], "title": "Old Space Reel"},
            "files": [{"name": "prelinger1.mp4", "size": "300000", "length": "600",
                       "width": "640", "height": "480"}],
        },
    }
    _install_fake_get(monkeypatch, search_docs, metadata)

    ctx = _ctx(tmp_path)
    d1.run(ctx)

    meta = read_json(ctx.job, d1.META_OUTPUT)
    assert meta["attribution"] == "Internet Archive / Old Space Reel"


# ---------------------------------------------------------------------------
# 全不达标 -> 明确报错
# ---------------------------------------------------------------------------

def test_raises_when_nothing_qualifies(tmp_path, monkeypatch):
    import src.stages.d1_source as d1

    search_docs = [{"identifier": "onlybad", "title": "Only Bad"}]
    metadata = {
        "onlybad": {
            "metadata": {"licenseurl": "https://rightsstatements.org/vocab/InC/1.0/",
                         "collection": ["nasa"], "title": "Only Bad"},
            "files": [{"name": "onlybad.mp4", "size": "300000", "length": "600",
                       "width": "640", "height": "480"}],
        },
    }
    _install_fake_get(monkeypatch, search_docs, metadata)

    ctx = _ctx(tmp_path)
    with pytest.raises(RuntimeError, match="未找到符合条件"):
        d1.run(ctx)

    assert not (ctx.job / d1.OUTPUT).exists()


def test_raises_when_search_returns_no_docs(tmp_path, monkeypatch):
    import src.stages.d1_source as d1

    _install_fake_get(monkeypatch, [], {})
    ctx = _ctx(tmp_path)
    with pytest.raises(RuntimeError):
        d1.run(ctx)


# ---------------------------------------------------------------------------
# 辅助函数单测
# ---------------------------------------------------------------------------

def test_parse_duration_handles_seconds_and_hms():
    from src.stages.d1_source import _parse_duration
    assert _parse_duration("187.32") == pytest.approx(187.32)
    assert _parse_duration("01:02:03") == pytest.approx(3723.0)
    assert _parse_duration(None) == 0.0
    assert _parse_duration("") == 0.0
    assert _parse_duration("garbage") == 0.0


def test_pick_video_file_skips_tiny_placeholder_and_out_of_range():
    from src.stages.d1_source import _pick_video_file
    files = [
        {"name": "placeholder.mp4", "size": "50000", "length": "600"},   # 太小
        {"name": "toolong.mp4", "size": "300000", "length": "5000"},     # 时长超范围
        {"name": "ok.mp4", "size": "300000", "length": "600",
         "width": "640", "height": "480"},
    ]
    picked = _pick_video_file(files, min_sec=300, max_sec=1800, max_bytes=500_000_000)
    assert picked is not None
    f, dur = picked
    assert f["name"] == "ok.mp4"
    assert dur == 600.0


def test_pick_video_file_returns_none_when_all_over_max_bytes():
    from src.stages.d1_source import _pick_video_file
    files = [{"name": "huge.mp4", "size": "999999999999", "length": "600",
              "width": "1920", "height": "1080"}]
    assert _pick_video_file(files, min_sec=300, max_sec=1800, max_bytes=500_000_000) is None


# ---------------------------------------------------------------------------
# 离线安全：模块顶层禁止触网 import
# ---------------------------------------------------------------------------

def test_module_import_does_no_network_at_top_level():
    import src.stages.d1_source as d1_mod

    banned = {"requests", "urllib3", "httpx"}
    tree = ast.parse(inspect.getsource(d1_mod))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] not in banned for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in banned
