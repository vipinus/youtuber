import ast
import inspect
import pytest
from pathlib import Path
from src.config import Config
from src.media import ArchiveVideo, FakeMedia, _norm_license, get_media_source


def test_fake_media_search_and_fetch(tmp_path):
    m = FakeMedia()
    cands = m.search("roswell desert")
    assert cands and cands[0]["license"] == "public-domain"
    dest = tmp_path / "x.jpg"
    m.fetch(cands[0], dest)
    from PIL import Image
    with Image.open(dest) as im:                     # 必须是可解码的合法 JPEG，不是占位字节
        assert im.format == "JPEG" and im.size[0] > 0


def test_get_media_source_fake():
    assert isinstance(get_media_source(Config({"media": {"provider": "fake"}})), FakeMedia)


def test_get_media_source_unknown_raises():
    with pytest.raises(ValueError):
        get_media_source(Config({"media": {"provider": "nope"}}))


def test_get_media_source_archive():
    assert isinstance(get_media_source(Config({"media": {"provider": "archive"}})), ArchiveVideo)


def test_media_module_has_no_top_level_requests_import():
    """离线单测不能触网：requests 只能在方法体内惰性 import，模块顶层禁止。"""
    import src.media as media_mod

    tree = ast.parse(inspect.getsource(media_mod))
    for node in tree.body:                      # 只看模块顶层语句，不递归进函数体
        if isinstance(node, ast.Import):
            assert all(a.name != "requests" for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module != "requests"


# ---------------------------------------------------------------------------
# _norm_license 护栏#2 单测
# ---------------------------------------------------------------------------

def test_norm_license_public_domain_zero():
    assert _norm_license("https://creativecommons.org/publicdomain/zero/1.0/", None) == "public-domain"


def test_norm_license_publicdomain_mark():
    assert _norm_license("https://creativecommons.org/licenses/publicdomain/", None) == "public-domain"


def test_norm_license_cc_by():
    assert _norm_license("https://creativecommons.org/licenses/by/3.0/", None) == "cc-by"


def test_norm_license_cc_by_sa():
    assert _norm_license("https://creativecommons.org/licenses/by-sa/4.0/", None) == "cc-by-sa"


def test_norm_license_by_nc_rejected_even_with_pd_collection():
    # 显式 NC 许可证优先于集合白名单——硬拒，不允许集合覆盖
    assert _norm_license("https://creativecommons.org/licenses/by-nc/4.0/", "prelinger") is None


def test_norm_license_by_nd_rejected():
    assert _norm_license("https://creativecommons.org/licenses/by-nd/4.0/", None) is None


def test_norm_license_no_license_pd_collection_str():
    assert _norm_license(None, "prelinger") == "public-domain"


def test_norm_license_no_license_pd_collection_list():
    assert _norm_license(None, ["some_other", "FedFlix"]) == "public-domain"


def test_norm_license_no_license_non_pd_collection_rejected():
    assert _norm_license(None, "mirrortube") is None


def test_norm_license_no_license_no_collection_rejected():
    assert _norm_license(None, None) is None


def test_norm_license_unrecognized_licenseurl_falls_back_to_collection():
    assert _norm_license("https://rightsstatements.org/vocab/InC/1.0/", "mirrortube") is None


# ---------------------------------------------------------------------------
# ArchiveVideo — 全离线，monkeypatch requests.get
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, json_data=None, content=b""):
        self._json = json_data
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self._json

    def iter_content(self, chunk_size=8192):
        yield self.content


def _search_docs(*docs):
    return _FakeResp(json_data={"response": {"docs": list(docs)}})


def test_archive_search_accepts_pd_licenseurl_item(monkeypatch):
    import requests

    docs = _search_docs({
        "identifier": "gov.archives.arc.1663615", "title": "FedFlix reel",
        "licenseurl": "https://creativecommons.org/publicdomain/zero/1.0/",
        "collection": ["FedFlix"],
    })
    monkeypatch.setattr(requests, "get", lambda *a, **k: docs)
    out = ArchiveVideo().search("weather balloon")
    assert len(out) == 1
    assert out[0]["manifest"] == "gov.archives.arc.1663615"
    assert out[0]["license"] == "public-domain"
    assert out[0]["ext"] == "mp4"
    assert out[0]["source"] == "archive"


def test_archive_search_accepts_prelinger_no_licenseurl(monkeypatch):
    import requests

    docs = _search_docs({
        "identifier": "some_prelinger_reel", "title": "Old newsreel",
        "collection": "prelinger",
    })
    monkeypatch.setattr(requests, "get", lambda *a, **k: docs)
    out = ArchiveVideo().search("newsreel")
    assert len(out) == 1
    assert out[0]["license"] == "public-domain"


def test_archive_search_rejects_no_license_non_pd_collection(monkeypatch):
    import requests

    docs = _search_docs({
        "identifier": "mirrored_yt_video", "title": "Mirrored YouTube clip",
        "collection": "mirrortube",
    })
    monkeypatch.setattr(requests, "get", lambda *a, **k: docs)
    out = ArchiveVideo().search("ufo")
    assert out == []


def test_archive_search_rejects_by_nc_by_nd(monkeypatch):
    import requests

    docs = _search_docs(
        {"identifier": "nc_item", "title": "NC clip",
         "licenseurl": "https://creativecommons.org/licenses/by-nc/4.0/"},
        {"identifier": "nd_item", "title": "ND clip",
         "licenseurl": "https://creativecommons.org/licenses/by-nd/4.0/"},
    )
    monkeypatch.setattr(requests, "get", lambda *a, **k: docs)
    out = ArchiveVideo().search("ufo")
    assert out == []


def test_archive_search_mixed_results_keeps_only_legal(monkeypatch):
    import requests

    docs = _search_docs(
        {"identifier": "bad1", "title": "bad", "collection": "altcensored"},
        {"identifier": "good1", "title": "good",
         "licenseurl": "https://creativecommons.org/licenses/by/3.0/"},
        {"identifier": "bad2", "title": "bad2", "collection": "PeriscopeFilm"},
    )
    monkeypatch.setattr(requests, "get", lambda *a, **k: docs)
    out = ArchiveVideo().search("weather balloon", limit=5)
    assert [c["manifest"] for c in out] == ["good1"]
    assert out[0]["license"] == "cc-by"


def test_archive_search_all_results_come_from_allowed_license_set(monkeypatch):
    import requests

    allowed = {"public-domain", "cc0", "cc-by", "cc-by-sa"}
    docs = _search_docs(
        {"identifier": "a", "title": "a", "licenseurl": "https://creativecommons.org/publicdomain/zero/1.0/"},
        {"identifier": "b", "title": "b", "licenseurl": "https://creativecommons.org/licenses/by-sa/4.0/"},
        {"identifier": "c", "title": "c", "collection": "usgovfilms"},
    )
    monkeypatch.setattr(requests, "get", lambda *a, **k: docs)
    out = ArchiveVideo().search("weather balloon", limit=5)
    assert out and all(c["license"] in allowed for c in out)
    assert all(c["source"] == "archive" for c in out)


def test_archive_fetch_picks_smallest_mp4(tmp_path, monkeypatch):
    import requests

    identifier = "gov.archives.arc.1663615"
    meta = _FakeResp(json_data={
        "files": [
            {"name": f"{identifier}.mp4", "format": "MPEG4", "size": "59000000"},
            {"name": f"{identifier}_512kb.mp4", "format": "512Kb MPEG4", "size": "12000000"},
            {"name": f"{identifier}.ogv", "format": "Ogg Video", "size": "20000000"},
        ],
        "metadata": {"licenseurl": "https://creativecommons.org/publicdomain/zero/1.0/"},
    })
    download = _FakeResp(content=b"FAKE_MP4_BYTES")

    calls = []

    def fake_get(url, *a, **k):
        calls.append(url)
        if "metadata" in url:
            return meta
        return download

    monkeypatch.setattr(requests, "get", fake_get)
    dest = tmp_path / "seg_001.mp4"
    ArchiveVideo().fetch({"manifest": identifier, "ext": "mp4",
                          "license": "public-domain", "attribution": "x", "source": "archive"}, dest)
    assert dest.read_bytes() == b"FAKE_MP4_BYTES"
    assert any(f"{identifier}_512kb.mp4" in u for u in calls)   # 选了最小的 mp4，不是 59MB 那个


def test_archive_fetch_skips_tiny_junk_mp4_for_real_derivative(tmp_path, monkeypatch):
    """Live-probe 发现的真实坑：部分条目带 'Download.mp4'/'Download (1).mp4' 这类几KB的
    占位/损坏文件（实测 0.04s 时长），blindly '选最小 mp4' 会挑中它们而不是真片——必须有
    最小可信体积下限，垃圾文件不算数。"""
    import requests

    identifier = "this-is-radio-free-europe-1964_"
    meta = _FakeResp(json_data={
        "files": [
            {"name": "Download.mp4", "format": "h.264", "size": "12487"},
            {"name": "Download (1).mp4", "format": "h.264", "size": "15848"},
            {"name": "This Is Radio Free Europe (1964).mp4", "format": "MPEG4", "size": "81090346"},
        ],
        "metadata": {"licenseurl": "http://creativecommons.org/publicdomain/mark/1.0/"},
    })
    download = _FakeResp(content=b"REAL_VIDEO_BYTES")
    calls = []

    def fake_get(url, *a, **k):
        calls.append(url)
        return meta if "metadata" in url else download

    monkeypatch.setattr(requests, "get", fake_get)
    dest = tmp_path / "seg_001.mp4"
    ArchiveVideo().fetch({"manifest": identifier, "ext": "mp4",
                          "license": "public-domain", "attribution": "x", "source": "archive"}, dest)
    assert dest.read_bytes() == b"REAL_VIDEO_BYTES"
    assert any("This Is Radio Free Europe" in u for u in calls)   # 真片，不是 12KB 占位文件
    assert not any(u.endswith("/Download.mp4") for u in calls)


def test_archive_fetch_falls_back_to_ogv_when_no_mp4(tmp_path, monkeypatch):
    import requests

    identifier = "some_id"
    meta = _FakeResp(json_data={
        "files": [{"name": f"{identifier}.ogv", "format": "Ogg Video", "size": "1000"}],
        "metadata": {"collection": "prelinger"},
    })
    download = _FakeResp(content=b"OGV_BYTES")
    monkeypatch.setattr(requests, "get", lambda url, *a, **k: meta if "metadata" in url else download)
    dest = tmp_path / "seg_001.ogv"
    ArchiveVideo().fetch({"manifest": identifier, "ext": "mp4",
                          "license": "public-domain", "attribution": "x", "source": "archive"}, dest)
    assert dest.read_bytes() == b"OGV_BYTES"


def test_archive_fetch_rejects_when_metadata_license_fails_second_check(tmp_path, monkeypatch):
    """纵深防御：即便 search 阶段放行，fetch 时 metadata 揭示的 licenseurl 不合法也要拒。"""
    import requests

    identifier = "sneaky_id"
    meta = _FakeResp(json_data={
        "files": [{"name": f"{identifier}.mp4", "format": "MPEG4", "size": "1000"}],
        "metadata": {"licenseurl": "https://rightsstatements.org/vocab/InC/1.0/"},
    })
    monkeypatch.setattr(requests, "get", lambda url, *a, **k: meta)
    dest = tmp_path / "seg_001.mp4"
    with pytest.raises(Exception):
        ArchiveVideo().fetch({"manifest": identifier, "ext": "mp4",
                              "license": "public-domain", "attribution": "x", "source": "archive"}, dest)
    assert not dest.exists()


def test_archive_fetch_raises_when_no_video_files(tmp_path, monkeypatch):
    import requests

    identifier = "no_video_id"
    meta = _FakeResp(json_data={
        "files": [{"name": f"{identifier}.txt", "format": "Text", "size": "10"}],
        "metadata": {"licenseurl": "https://creativecommons.org/publicdomain/zero/1.0/"},
    })
    monkeypatch.setattr(requests, "get", lambda url, *a, **k: meta)
    dest = tmp_path / "seg_001.mp4"
    with pytest.raises(Exception):
        ArchiveVideo().fetch({"manifest": identifier, "ext": "mp4",
                              "license": "public-domain", "attribution": "x", "source": "archive"}, dest)
    assert not dest.exists()


def test_norm_license_rejects_explicit_incopyright_even_in_pd_collection():
    """封号级红线:显式非许可的 licenseurl(如 rightsstatements InC)即便在 PD 集合也必须拒。"""
    from src.media import _norm_license
    assert _norm_license("https://rightsstatements.org/vocab/InC/1.0/", "prelinger") is None
    assert _norm_license("https://rightsstatements.org/vocab/InC/1.0/", "usgovfilms") is None
    assert _norm_license("http://example.com/some-rights", ["FedFlix"]) is None
    # by-nd 即便在 PD 集合也拒(禁止改作)
    assert _norm_license("http://creativecommons.org/licenses/by-nd/4.0/", "prelinger") is None
    # 无 licenseurl + PD 集合 → 放行(collection 信任路径仅在无显式许可时生效)
    assert _norm_license(None, "prelinger") == "public-domain"
    assert _norm_license("", ["usgovfilms"]) == "public-domain"
    # 显式 PD/CC 正常放行
    assert _norm_license("http://creativecommons.org/publicdomain/zero/1.0/", None) == "public-domain"
    assert _norm_license("http://creativecommons.org/licenses/by/4.0/", "mirrortube") == "cc-by"


def test_is_blocked_catches_atrocity_content():
    """内容安全:大屠杀/暴行类(即便PD)必须挡——含首集实跑的 Death Mills case。"""
    from src.media import _is_blocked
    assert _is_blocked("Death Mills [Todesmuehlen]")                    # 标题命中
    assert _is_blocked("Some Film", "Holocaust; Concentration camps")   # subject命中
    assert _is_blocked("AUSCHWITZ liberation")                          # 大小写不敏感
    assert not _is_blocked("Roswell Reports Vol. 6", "UFO; military")   # 正常题材放行
    assert not _is_blocked("AAF Combat Camera Weekly Digest")


def test_search_drops_blocked_even_if_pd(monkeypatch):
    """search:内容黑名单命中的条目即便 PD 也不作为候选。"""
    import src.media as media

    class _Resp:
        def raise_for_status(self): pass
        def json(self):
            return {"response": {"docs": [
                {"identifier": "death_mills", "title": "Death Mills [Todesmuehlen]",
                 "licenseurl": "http://creativecommons.org/publicdomain/zero/1.0/",
                 "subject": ["Holocaust", "concentration camps"]},
                {"identifier": "roswell6", "title": "Roswell Reports Vol 6",
                 "licenseurl": "http://creativecommons.org/publicdomain/zero/1.0/",
                 "subject": ["UFO"]},
            ]}}
    monkeypatch.setattr("requests.get", lambda *a, **k: _Resp())
    out = media.ArchiveVideo().search("roswell", limit=5)
    ids = [c["manifest"] for c in out]
    assert "death_mills" not in ids and "roswell6" in ids
