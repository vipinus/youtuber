"""P7 T20：d9c_publish_en——把英文原声原片(source.mp4,未配音)追加发布到第二个
YouTube 账号/频道。全离线：provider=fake 时走 src.youtube.FakeUploader，绝不
真的导入 googleapiclient/触网；enabled=false / token 缺失分支更是连 uploader
都不构造。"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Config
from src.context import Ctx
from src.llm import FakeLLM
from src.util.jobs import read_json, write_json

import src.youtube as youtube_mod
import src.stages.d9c_publish_en as d9c_publish_en
from src.stages.d9c_publish_en import OUTPUT, INPUT, _build_meta, run


def _ctx(tmp_path, cfg: Config, slug: str = "c") -> Ctx:
    ctx = Ctx(slug=slug, config=cfg, root=tmp_path / "jobs", llm=FakeLLM())
    write_json(ctx.job, INPUT, {
        "identifier": "nasadoc1",
        "title": "NASA Space Documentary",
        "attribution": "Source: NASA, Public Domain.",
    })
    (ctx.job / "source.mp4").write_bytes(b"MP4")
    return ctx


def _capture_fake_uploader(monkeypatch):
    """FakeUploader 是 src.youtube 里的同一个类对象；给 __init__/upload 都套一层
    记录壳，既走真实的 provider=fake 代码路径，又能在测试里拿到调用参数/实例。"""
    instances: list[youtube_mod.FakeUploader] = []
    calls: dict = {}

    orig_init = youtube_mod.FakeUploader.__init__
    orig_upload = youtube_mod.FakeUploader.upload

    def capturing_init(self):
        orig_init(self)
        instances.append(self)

    def capturing_upload(self, video, meta, thumb, visibility):
        calls["video"] = video
        calls["meta"] = meta
        calls["thumb"] = thumb
        calls["visibility"] = visibility
        return orig_upload(self, video, meta, thumb, visibility)

    monkeypatch.setattr(youtube_mod.FakeUploader, "__init__", capturing_init)
    monkeypatch.setattr(youtube_mod.FakeUploader, "upload", capturing_upload)
    return instances, calls


# ---------------------------------------------------------------------------
# 1. youtube_en.enabled=false（默认）→ disabled，不构造 uploader
# ---------------------------------------------------------------------------

def test_disabled_by_default_marks_status(tmp_path):
    ctx = _ctx(tmp_path, Config({}))

    run(ctx)

    out = read_json(ctx.job, OUTPUT)
    assert out == {"status": "disabled"}
    assert not ctx.warnings


def test_disabled_explicit_false_marks_status(tmp_path):
    ctx = _ctx(tmp_path, Config({"youtube_en": {"enabled": False}}))

    run(ctx)

    out = read_json(ctx.job, OUTPUT)
    assert out == {"status": "disabled"}


# ---------------------------------------------------------------------------
# 2. enabled 但 token 文件缺失（还没走一次性授权）→ no_login，只 warn 不 raise
# ---------------------------------------------------------------------------

def test_enabled_but_token_missing_warns_and_skips(tmp_path):
    token = tmp_path / "secrets_not_authed" / "token_en.json"   # 从不创建
    ctx = _ctx(tmp_path, Config({
        "youtube_en": {"enabled": True, "token": str(token)},
    }))

    run(ctx)   # provider 默认 youtube，走 token 存在性检查，不该触碰 googleapiclient

    out = read_json(ctx.job, OUTPUT)
    assert out == {"status": "no_login"}
    assert ctx.warnings
    assert any("auth_youtube" in w for w in ctx.warnings)
    assert any(str(token) in w for w in ctx.warnings)


# ---------------------------------------------------------------------------
# 3. happy path：provider=fake，上传参数/元数据/播放列表都对
# ---------------------------------------------------------------------------

def test_happy_path_uploads_source_mp4_with_unlisted_visibility(tmp_path, monkeypatch):
    instances, calls = _capture_fake_uploader(monkeypatch)
    ctx = _ctx(tmp_path, Config({
        "youtube_en": {
            "enabled": True, "provider": "fake", "visibility": "unlisted",
            "playlist": "Space / NASA Archive",
        },
    }))

    run(ctx)

    # 上传参数：视频是 source.mp4（英文原声原片），visibility 来自配置
    assert calls["video"] == ctx.job / "source.mp4"
    assert calls["visibility"] == "unlisted"
    assert calls["thumb"] == ctx.job / "thumb_en.jpg"     # 英文封面(标题+EN水印)已生成
    assert calls["thumb"].exists()

    # 输出 marker
    out = read_json(ctx.job, OUTPUT)
    assert out["status"] == "published"
    assert out["videoId"] == "FAKE12345"
    assert out["url"] == "https://youtu.be/FAKE12345"
    assert out["visibility"] == "unlisted"

    # 播放列表：记录在 FakeUploader 实例的 playlist_calls 上
    assert len(instances) == 1
    assert instances[0].playlist_calls == [("FAKE12345", "Space / NASA Archive")]
    assert out["playlist"] == "Space / NASA Archive"
    assert out["playlistId"] == "FAKEPL123"

    assert not ctx.warnings


def test_happy_path_metadata_shape_from_source_json(tmp_path, monkeypatch):
    _, calls = _capture_fake_uploader(monkeypatch)
    ctx = _ctx(tmp_path, Config({"youtube_en": {"enabled": True, "provider": "fake"}}))

    run(ctx)

    meta = calls["meta"]
    assert meta["title"] == "NASA Space Documentary"
    assert meta["description"] == (
        "Source: NASA, Public Domain.\n\n"
        "Public domain source: https://archive.org/details/nasadoc1\n\n"
        "This channel curates public-domain space & science films."
    )
    assert meta["tags"] == ["NASA", "space", "documentary", "public domain", "archive"]


def test_happy_path_default_visibility_is_unlisted_never_public(tmp_path, monkeypatch):
    """护栏#3：youtube_en 没配 visibility 时绝不能默认 public。"""
    _, calls = _capture_fake_uploader(monkeypatch)
    ctx = _ctx(tmp_path, Config({"youtube_en": {"enabled": True, "provider": "fake"}}))

    run(ctx)

    assert calls["visibility"] == "unlisted"
    out = read_json(ctx.job, OUTPUT)
    assert out["visibility"] == "unlisted"


def test_no_playlist_configured_skips_playlist_call(tmp_path, monkeypatch):
    instances, _ = _capture_fake_uploader(monkeypatch)
    ctx = _ctx(tmp_path, Config({"youtube_en": {"enabled": True, "provider": "fake"}}))

    run(ctx)

    assert instances[0].playlist_calls == []
    out = read_json(ctx.job, OUTPUT)
    assert "playlist" not in out
    assert "playlistId" not in out


# ---------------------------------------------------------------------------
# 4. 上传失败 → failed，只 warn，绝不 raise（追加发布不能拖累主链路）
# ---------------------------------------------------------------------------

def test_upload_failure_marks_failed_and_warns_without_raising(tmp_path, monkeypatch):
    def raising_upload(self, video, meta, thumb, visibility):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(youtube_mod.FakeUploader, "upload", raising_upload)
    ctx = _ctx(tmp_path, Config({"youtube_en": {"enabled": True, "provider": "fake"}}))

    run(ctx)   # 不该抛异常

    out = read_json(ctx.job, OUTPUT)
    assert out["status"] == "failed"
    assert "quota exceeded" in out["error"]
    assert ctx.warnings


def test_unknown_provider_raises_inside_marks_failed(tmp_path):
    """_uploader_for 对未知 provider 抛 ValueError；run() 必须吞掉它，只留 failed 标记。"""
    token = tmp_path / "token_en.json"
    token.write_text("{}", encoding="utf-8")   # 让 token 存在性检查先过，才能真正走到 _uploader_for
    ctx = _ctx(tmp_path, Config({
        "youtube_en": {"enabled": True, "provider": "bogus", "token": str(token)},
    }))

    run(ctx)   # 不该抛异常

    out = read_json(ctx.job, OUTPUT)
    assert out["status"] == "failed"
    assert "bogus" in out["error"] or "ValueError" in out["error"]
    assert ctx.warnings


def test_missing_source_json_marks_failed_without_raising(tmp_path, monkeypatch):
    """理论上 doc 管线总有 source.json（d1_source 早跑过），但防御一下缺失情形：
    读不到就是失败，不能让 run 崩掉。"""
    _capture_fake_uploader(monkeypatch)
    ctx = Ctx(slug="c", config=Config({"youtube_en": {"enabled": True, "provider": "fake"}}),
              root=tmp_path / "jobs", llm=FakeLLM())
    ctx.job.mkdir(parents=True)
    (ctx.job / "source.mp4").write_bytes(b"MP4")
    assert not (ctx.job / INPUT).exists()

    run(ctx)

    out = read_json(ctx.job, OUTPUT)
    assert out["status"] == "failed"
    assert ctx.warnings


# ---------------------------------------------------------------------------
# 5. 播放列表失败不能作废已成功的上传
# ---------------------------------------------------------------------------

def test_playlist_failure_keeps_published_status(tmp_path, monkeypatch):
    _capture_fake_uploader(monkeypatch)

    def raising_add_to_playlist(self, video_id, playlist_title):
        raise RuntimeError("insufficientPermissions")

    monkeypatch.setattr(youtube_mod.FakeUploader, "add_to_playlist", raising_add_to_playlist)
    ctx = _ctx(tmp_path, Config({
        "youtube_en": {"enabled": True, "provider": "fake", "playlist": "Space / NASA Archive"},
    }))

    run(ctx)   # 不该抛异常

    out = read_json(ctx.job, OUTPUT)
    assert out["status"] == "published"      # 上传本身成功，播放列表失败不作废它
    assert out["videoId"] == "FAKE12345"
    assert "insufficientPermissions" in out["playlist_error"]
    assert ctx.warnings


# ---------------------------------------------------------------------------
# 6. 标题超 100 字截断
# ---------------------------------------------------------------------------

def test_build_meta_clamps_title_to_100_chars():
    long_title = "A" * 150
    meta = _build_meta({"title": long_title, "identifier": "x", "attribution": "a"})
    assert len(meta["title"]) == 100
    assert meta["title"] == long_title[:100]


def test_build_meta_handles_missing_fields_gracefully():
    meta = _build_meta({})
    assert meta["title"] == ""
    assert meta["description"] == (
        "\n\nPublic domain source: \n\n"
        "This channel curates public-domain space & science films."
    )
    assert meta["tags"] == ["NASA", "space", "documentary", "public domain", "archive"]


def test_wrap_en_title_by_words():
    from src.stages.d9c_publish_en import _wrap_en_title
    lines = _wrap_en_title("The Space Shuttle (Narrated by William Shatner)")
    assert len(lines) == 2
    assert " ".join(lines) == "The Space Shuttle (Narrated by William Shatner)"  # 一个词都不丢
    assert _wrap_en_title("Short") == ["Short"]


def test_make_en_thumbnail_renders(tmp_path):
    from src.stages.d9c_publish_en import make_en_thumbnail
    from PIL import Image
    bg = tmp_path / "bg.png"
    Image.new("RGB", (1280, 720), (20, 30, 60)).save(bg)
    dest = tmp_path / "thumb_en.jpg"
    make_en_thumbnail(dest, "The Space Shuttle Narrated by William Shatner",
                      bg, Path("assets/watermark_en_tr.png"))
    assert dest.exists() and Image.open(dest).size == (1280, 720)
