"""P7 T16：已发布纪录片登记表——纯离线单测，不碰任何 job 阶段。"""
from __future__ import annotations

import json

from src.util.registry import (
    append_published,
    in_flight_identifiers,
    load_published,
    published_identifiers,
    registry_path,
)


def test_registry_path_is_jobs_root_slash_published_json(tmp_path):
    assert registry_path(tmp_path) == tmp_path / "published.json"


def test_load_published_missing_file_returns_empty_list(tmp_path):
    assert load_published(tmp_path) == []


def test_load_published_corrupt_json_returns_empty_list_not_crash(tmp_path):
    (tmp_path / "published.json").write_text("{not valid json", encoding="utf-8")
    assert load_published(tmp_path) == []


def test_load_published_non_list_json_returns_empty_list(tmp_path):
    (tmp_path / "published.json").write_text(
        json.dumps({"oops": "should be a list"}), encoding="utf-8")
    assert load_published(tmp_path) == []


def test_load_published_skips_non_dict_entries(tmp_path):
    (tmp_path / "published.json").write_text(
        json.dumps([{"identifier": "ok"}, "garbage", 42]), encoding="utf-8")
    assert load_published(tmp_path) == [{"identifier": "ok"}]


# ---------------------------------------------------------------------------
# append → load 往返 + published_identifiers
# ---------------------------------------------------------------------------

def test_append_published_then_load_roundtrip(tmp_path):
    append_published(tmp_path, {
        "identifier": "id1", "title": "T1", "videoId": "V1",
        "slug": "s1", "published_at": "2026-01-01T00:00:00+00:00",
    })
    append_published(tmp_path, {
        "identifier": "id2", "title": "T2", "videoId": "V2",
        "slug": "s2", "published_at": "2026-01-02T00:00:00+00:00",
    })

    entries = load_published(tmp_path)
    assert [e["identifier"] for e in entries] == ["id1", "id2"]
    assert entries[0]["videoId"] == "V1"
    assert published_identifiers(tmp_path) == {"id1", "id2"}


def test_append_published_write_is_atomic_no_leftover_tmp_files(tmp_path):
    append_published(tmp_path, {"identifier": "id1"})
    leftovers = list(tmp_path.glob(".published.json.*.tmp"))
    assert leftovers == []
    assert registry_path(tmp_path).is_file()
    # 写完的是合法 JSON 数组，不是半写的碎片
    data = json.loads(registry_path(tmp_path).read_text(encoding="utf-8"))
    assert data == [{"identifier": "id1"}]


def test_published_identifiers_empty_registry(tmp_path):
    assert published_identifiers(tmp_path) == set()


# ---------------------------------------------------------------------------
# in_flight_identifiers —— 扫同级 job 目录的 source.json
# ---------------------------------------------------------------------------

def test_in_flight_identifiers_finds_sibling_source_json(tmp_path):
    (tmp_path / "job-a").mkdir()
    (tmp_path / "job-a" / "source.json").write_text(
        json.dumps({"identifier": "sib1"}), encoding="utf-8")
    (tmp_path / "job-b").mkdir()
    (tmp_path / "job-b" / "source.json").write_text(
        json.dumps({"identifier": "sib2"}), encoding="utf-8")

    assert in_flight_identifiers(tmp_path, exclude_slug="job-c") == {"sib1", "sib2"}


def test_in_flight_identifiers_excludes_own_slug(tmp_path):
    (tmp_path / "job-a").mkdir()
    (tmp_path / "job-a" / "source.json").write_text(
        json.dumps({"identifier": "sib1"}), encoding="utf-8")

    assert in_flight_identifiers(tmp_path, exclude_slug="job-a") == set()


def test_in_flight_identifiers_ignores_missing_corrupt_and_non_dirs(tmp_path):
    (tmp_path / "job-a").mkdir()                       # 没有 source.json
    (tmp_path / "job-b").mkdir()
    (tmp_path / "job-b" / "source.json").write_text("{broken", encoding="utf-8")
    (tmp_path / "not-a-dir.txt").write_text("x", encoding="utf-8")  # 非目录，得跳过

    assert in_flight_identifiers(tmp_path, exclude_slug="none") == set()


def test_in_flight_identifiers_missing_jobs_root_returns_empty(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert in_flight_identifiers(missing, exclude_slug="x") == set()
