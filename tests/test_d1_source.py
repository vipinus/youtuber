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


import pytest as _pytest

@_pytest.fixture(autouse=True)
def _assume_audio_ok(monkeypatch):
    """旧测试的假下载写的是垃圾字节,真 ffprobe 必判无音轨——默认放行;
    专测无音轨行为的用例自行覆写 _has_audio。"""
    import src.stages.d1_source as _d1
    monkeypatch.setattr(_d1, "_has_audio", lambda p: True)


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


def test_looks_narrated_rejects_broll_footage():
    """真跑教训:NASA UHD 飞越/fisheye 素材无解说→转录0段→废片。d1 须过滤掉。"""
    from src.stages.d1_source import _looks_narrated
    assert not _looks_narrated("NASA Ultra High Definition")
    assert not _looks_narrated("Space Station Fisheye Fly-Through 4K Ultra-HD")
    assert not _looks_narrated("Earth Timelapse", "b-roll footage")
    assert not _looks_narrated("Aurora 8K UHD flyover")
    # 正常解说纪录片放行
    assert _looks_narrated("The Story of Apollo 11", "NASA history documentary narration")
    assert _looks_narrated("Our World: Arecibo - The Largest Radio Telescope")


def test_looks_narrated_rejects_resource_reels():
    """HD 实探发现:NASA 'Resource Reel'/stock footage 是给创作者的无旁白素材卷,须过滤。"""
    from src.stages.d1_source import _looks_narrated
    assert not _looks_narrated("NASA Orion Resource Reel")
    assert not _looks_narrated("Artemis stock footage collection")
    assert _looks_narrated("ISS Benefits for Humanity")      # 真解说系列片放行


# ---------------------------------------------------------------------------
# P7 T16：登记表去重 —— 已发布/在途 identifier 跳过，挑下一个候选
# ---------------------------------------------------------------------------

def test_skips_already_published_candidate_and_picks_next(tmp_path, monkeypatch):
    """published.json 里已经登记过的 identifier 不该被重新选中——这正是用户
    投诉的"同一条 query 反复重挑同一部片"问题。也校验 metadata 都没去取
    （在拿 metadata 之前就该跳过，不浪费一次 HTTP 请求）。"""
    import src.stages.d1_source as d1
    from src.util.registry import append_published

    search_docs = [
        {"identifier": "already_done", "title": "Old Pick"},
        {"identifier": "fresh1", "title": "Fresh Doc"},
    ]
    metadata = {
        "already_done": {
            "metadata": {"licenseurl": "http://creativecommons.org/licenses/publicdomain/",
                         "collection": ["nasa"], "title": "Old Pick"},
            "files": [{"name": "already_done.mp4", "size": "300000", "length": "600",
                       "width": "640", "height": "480"}],
        },
        "fresh1": {
            "metadata": {"licenseurl": "http://creativecommons.org/licenses/publicdomain/",
                         "collection": ["nasa"], "title": "Fresh Doc"},
            "files": [{"name": "fresh1.mp4", "size": "300000", "length": "600",
                       "width": "640", "height": "480"}],
        },
    }
    _install_fake_get(monkeypatch, search_docs, metadata)

    fetched_ids: list[str] = []
    real_fetch_metadata = d1._fetch_metadata

    def tracking_fetch_metadata(identifier):
        fetched_ids.append(identifier)
        return real_fetch_metadata(identifier)

    monkeypatch.setattr(d1, "_fetch_metadata", tracking_fetch_metadata)

    ctx = _ctx(tmp_path)
    append_published(ctx.root, {
        "identifier": "already_done", "title": "Old Pick", "videoId": "V1",
        "slug": "some-other-job", "published_at": "2026-01-01T00:00:00+00:00",
    })

    d1.run(ctx)

    meta = read_json(ctx.job, d1.META_OUTPUT)
    assert meta["identifier"] == "fresh1"
    assert any("已发布/在途" in w for w in ctx.warnings)
    assert fetched_ids == ["fresh1"]           # already_done 在拿 metadata 前就被跳过了


def test_skips_in_flight_sibling_job_candidate(tmp_path, monkeypatch):
    """同批另一个 job 目录已经下载了同一部片（source.json 存在但还没发布完），
    也不该被再挑一次——防止并行/连续跑 case 撞到同一部片。"""
    import src.stages.d1_source as d1
    from src.util.jobs import write_json

    search_docs = [
        {"identifier": "in_flight1", "title": "Being Processed"},
        {"identifier": "fresh2", "title": "Fresh Doc Two"},
    ]
    metadata = {
        "in_flight1": {
            "metadata": {"licenseurl": "http://creativecommons.org/licenses/publicdomain/",
                         "collection": ["nasa"], "title": "Being Processed"},
            "files": [{"name": "in_flight1.mp4", "size": "300000", "length": "600",
                       "width": "640", "height": "480"}],
        },
        "fresh2": {
            "metadata": {"licenseurl": "http://creativecommons.org/licenses/publicdomain/",
                         "collection": ["nasa"], "title": "Fresh Doc Two"},
            "files": [{"name": "fresh2.mp4", "size": "300000", "length": "600",
                       "width": "640", "height": "480"}],
        },
    }
    _install_fake_get(monkeypatch, search_docs, metadata)

    ctx = _ctx(tmp_path)             # slug="doc1"
    sibling = tmp_path / "doc2"
    write_json(sibling, d1.META_OUTPUT, {"identifier": "in_flight1", "title": "Being Processed"})

    d1.run(ctx)

    meta = read_json(ctx.job, d1.META_OUTPUT)
    assert meta["identifier"] == "fresh2"
    assert any("已发布/在途" in w for w in ctx.warnings)


def test_does_not_skip_own_slug_source_json(tmp_path, monkeypatch):
    """in_flight 排重要排除自己这个 job（重跑同一个 slug 时不能被自己的
    上一次 source.json 拦住）。"""
    import src.stages.d1_source as d1
    from src.util.jobs import write_json

    search_docs = [{"identifier": "same1", "title": "Same Doc"}]
    metadata = {
        "same1": {
            "metadata": {"licenseurl": "http://creativecommons.org/licenses/publicdomain/",
                         "collection": ["nasa"], "title": "Same Doc"},
            "files": [{"name": "same1.mp4", "size": "300000", "length": "600",
                       "width": "640", "height": "480"}],
        },
    }
    _install_fake_get(monkeypatch, search_docs, metadata)

    ctx = _ctx(tmp_path)             # slug="doc1"
    write_json(ctx.job, d1.META_OUTPUT, {"identifier": "same1", "title": "Same Doc"})

    d1.run(ctx)                      # 不该因为自己的 source.json 而把 same1 当"在途"跳过

    meta = read_json(ctx.job, d1.META_OUTPUT)
    assert meta["identifier"] == "same1"


def test_looks_narrated_rejects_onboard_film():
    """真跑踩坑:APOLLO 11 16MM ONBOARD FILM 是无音轨机载胶片。"""
    from src.stages.d1_source import _looks_narrated
    assert not _looks_narrated("APOLLO 11 16MM ONBOARD FILM")
    assert not _looks_narrated("Gemini onboard camera footage")


def test_run_skips_downloaded_file_without_audio(tmp_path, monkeypatch):
    """下载后实测无音轨→删文件换下一部候选,不能带着哑片进 d2。"""
    import src.stages.d1_source as d1
    from src.config import Config
    from src.context import Ctx
    from src.llm import FakeLLM

    docs = [
        {"identifier": "silent-one", "title": "Great Space Movie A",
         "licenseurl": "http://creativecommons.org/publicdomain/zero/1.0/"},
        {"identifier": "good-one", "title": "Great Space Movie B",
         "licenseurl": "http://creativecommons.org/publicdomain/zero/1.0/"},
    ]
    meta = {"files": [{"name": "m.mp4", "size": "5000000", "length": "600",
                       "width": "1280", "height": "720"}],
            "metadata": {"licenseurl": "http://creativecommons.org/publicdomain/zero/1.0/",
                          "collection": ["nasa"]}}
    monkeypatch.setattr(d1, "_search", lambda *a, **k: docs)
    monkeypatch.setattr(d1, "_fetch_metadata", lambda ident: dict(meta))
    monkeypatch.setattr(d1, "_download", lambda url, dest: dest.write_bytes(b"x" * 10))
    monkeypatch.setattr(d1, "_has_audio", lambda p: "good-one" in captured.get("last", ""))
    captured = {}
    orig_download = d1._download
    def spy_download(url, dest):
        captured["last"] = url
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x" * 10)
    monkeypatch.setattr(d1, "_download", spy_download)

    ctx = Ctx(slug="c", config=Config({"p7": {"collections": ["nasa"]}}),
              root=tmp_path, llm=FakeLLM())
    d1.run(ctx)
    import json
    src = json.loads((ctx.job / "source.json").read_text())
    assert src["identifier"] == "good-one"        # 哑片被弃,选中第二部
    assert any("无音轨" in w for w in ctx.warnings)


def test_pick_video_file_min_height_gate():
    """p7.min_height 画质门:低于下限的档一律跳过(zion 生产档只收 >=720p)。"""
    from src.stages.d1_source import _pick_video_file
    files = [
        {"name": "sd.mp4", "size": "5000000", "length": "600", "width": "640", "height": "360"},
        {"name": "hd.mp4", "size": "9000000", "length": "600", "width": "1280", "height": "720"},
    ]
    f, dur = _pick_video_file(files, 300, 1800, 10**9, min_height=700)
    assert f["name"] == "hd.mp4"
    assert _pick_video_file([files[0]], 300, 1800, 10**9, min_height=700) is None  # 只有SD→整item拒
    f2, _ = _pick_video_file([files[0]], 300, 1800, 10**9)                          # 不设门→SD可用
    assert f2["name"] == "sd.mp4"
