"""P7 T15：d10_archive——发布后归档到 NAS。全离线（tmp_path 充当假 NAS），
不碰真实 `~/nas`（每个测试都显式传 archive.dir，绝不吃 DEFAULT_ARCHIVE_DIR 的
默认值，否则会真的在跑测试的机器家目录下 mkdir 出一棵 ~/nas/... 目录树）。
"""
from __future__ import annotations

import os
import stat

import pytest

from src.config import Config
from src.context import Ctx
from src.llm import FakeLLM
from src.util.jobs import read_json, write_json

import src.stages.d10_archive as d10_archive
from src.stages.d10_archive import OUTPUT, run


def _make_job(tmp_path, cfg: Config, slug: str = "c") -> Ctx:
    root = tmp_path / "jobs"
    ctx = Ctx(slug=slug, config=cfg, root=root, llm=FakeLLM())
    job = ctx.job
    job.mkdir(parents=True)

    # 大媒体（trim 候选）
    (job / "source.mp4").write_bytes(b"S" * 2000)
    (job / "video.mp4").write_bytes(b"V" * 1000)
    (job / "source_audio.wav").write_bytes(b"A" * 500)
    (job / "narration.wav").write_bytes(b"N" * 300)
    (job / "no_vocals.mp3").write_bytes(b"M" * 200)
    (job / "thumb_bg.png").write_bytes(b"BG" * 100)
    (job / "narration").mkdir()
    (job / "narration" / "seg_0.mp3").write_bytes(b"n0")
    (job / "narration" / "seg_1.mp3").write_bytes(b"n1n1")

    # 小台账（保留候选）
    (job / "subs.ass").write_text("[Script Info]\n", encoding="utf-8")
    (job / "thumb.jpg").write_bytes(b"JPG")
    write_json(job, "meta.json", {"title": "t", "description": "d", "tags": ["x"]})
    write_json(job, "publish.json", {"status": "private", "videoId": "V1"})

    return ctx


def _all_local_files(job):
    return sorted(p.relative_to(job) for p in job.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# 1. happy path：复制+校验+裁剪
# ---------------------------------------------------------------------------

def test_happy_path_archives_verifies_and_trims(tmp_path):
    archive_root = tmp_path / "nas"
    archive_root.mkdir()
    ctx = _make_job(tmp_path, Config({"archive": {"dir": str(archive_root)}}))
    before = _all_local_files(ctx.job)                     # archive.json 尚不存在，天然不含它
    sizes_before = {rel: (ctx.job / rel).stat().st_size for rel in before}

    run(ctx)

    dest = archive_root / ctx.slug
    for rel in before:
        dst = dest / rel
        assert dst.exists(), f"missing {rel}"
        assert dst.stat().st_size == sizes_before[rel]

    arc = read_json(ctx.job, OUTPUT)
    assert arc["status"] == "archived"
    assert arc["dest"] == str(dest)
    assert arc["files"] == len(before)
    assert arc["bytes"] == sum(sizes_before.values())

    # 大媒体本地被裁剪
    assert not (ctx.job / "source.mp4").exists()
    assert not (ctx.job / "video.mp4").exists()
    assert not (ctx.job / "source_audio.wav").exists()
    assert not (ctx.job / "narration.wav").exists()
    assert not (ctx.job / "no_vocals.mp3").exists()
    assert not (ctx.job / "thumb_bg.png").exists()
    assert not (ctx.job / "narration").exists()
    assert set(arc["trimmed"]) == {
        "source.mp4", "video.mp4", "source_audio.wav", "narration.wav",
        "no_vocals.mp3", "thumb_bg.png", "narration/",
    }

    # 小台账留在本地
    assert (ctx.job / "subs.ass").exists()
    assert (ctx.job / "thumb.jpg").exists()
    assert (ctx.job / "meta.json").exists()
    assert (ctx.job / "publish.json").exists()

    # NAS 上一份完整、内容正确
    assert (dest / "video.mp4").read_bytes() == b"V" * 1000
    assert (dest / "narration" / "seg_0.mp3").read_bytes() == b"n0"
    assert not (dest / OUTPUT).exists()    # archive.json 自身不该被搬过去


# ---------------------------------------------------------------------------
# 2. trim_local=false：只复制不删
# ---------------------------------------------------------------------------

def test_trim_local_false_copies_but_keeps_everything_local(tmp_path):
    archive_root = tmp_path / "nas"
    archive_root.mkdir()
    ctx = _make_job(tmp_path, Config({
        "archive": {"dir": str(archive_root), "trim_local": False},
    }))

    run(ctx)

    arc = read_json(ctx.job, OUTPUT)
    assert arc["status"] == "archived"
    assert arc["trimmed"] == []

    assert (ctx.job / "video.mp4").exists()
    assert (ctx.job / "source.mp4").exists()
    assert (ctx.job / "narration").is_dir()
    assert (archive_root / ctx.slug / "video.mp4").exists()


# ---------------------------------------------------------------------------
# 3. archive.dir 未配置 → skipped，不崩，不删本地
# ---------------------------------------------------------------------------

def test_archive_dir_unset_skips_without_crashing(tmp_path):
    ctx = _make_job(tmp_path, Config({"archive": {"dir": ""}}))

    run(ctx)

    arc = read_json(ctx.job, OUTPUT)
    assert arc["status"] == "skipped"
    assert ctx.warnings
    assert (ctx.job / "video.mp4").exists()    # 没删


def test_archive_dir_none_skips(tmp_path):
    ctx = _make_job(tmp_path, Config({"archive": {"dir": None}}))
    run(ctx)
    arc = read_json(ctx.job, OUTPUT)
    assert arc["status"] == "skipped"


# ---------------------------------------------------------------------------
# 4. archive.dir 不存在（dev 笔记本没挂 NFS 的情形）→ unreachable，绝不 mkdir 出假树
# ---------------------------------------------------------------------------

def test_archive_dir_missing_marks_unreachable_and_does_not_create_it(tmp_path):
    archive_root = tmp_path / "nas_not_mounted"   # 从不创建
    ctx = _make_job(tmp_path, Config({"archive": {"dir": str(archive_root)}}))

    run(ctx)

    arc = read_json(ctx.job, OUTPUT)
    assert arc["status"] == "unreachable"
    assert not archive_root.exists()       # 没有偷偷 mkdir 出一棵假目录树
    assert (ctx.job / "video.mp4").exists()  # 没有裁剪本地
    assert ctx.warnings


# ---------------------------------------------------------------------------
# 5. archive.dir 存在但不可写（mkdir 子目录失败）→ unreachable，不崩，不裁剪
# ---------------------------------------------------------------------------

def test_archive_dir_readonly_marks_unreachable(tmp_path):
    if os.name != "posix" or os.geteuid() == 0:
        pytest.skip("需要非 root 的 POSIX 权限语义才能测只读目录")

    archive_root = tmp_path / "readonly_nas"
    archive_root.mkdir()
    archive_root.chmod(stat.S_IRUSR | stat.S_IXUSR)   # r-x，不可写，无法在其中创建子目录

    ctx = _make_job(tmp_path, Config({"archive": {"dir": str(archive_root)}}))
    try:
        run(ctx)
    finally:
        archive_root.chmod(stat.S_IRWXU)              # 测试结束后恢复权限，方便 tmp_path 清理

    arc = read_json(ctx.job, OUTPUT)
    assert arc["status"] == "unreachable"
    assert (ctx.job / "video.mp4").exists()            # 没有裁剪本地
    assert ctx.warnings


# ---------------------------------------------------------------------------
# 6. 复制后校验发现大小不符（NAS 写坏/半路截断）→ failed，绝不裁剪本地
# ---------------------------------------------------------------------------

def test_size_mismatch_marks_failed_and_does_not_trim(tmp_path, monkeypatch):
    archive_root = tmp_path / "nas"
    archive_root.mkdir()
    ctx = _make_job(tmp_path, Config({"archive": {"dir": str(archive_root)}}))

    real_copy_file = d10_archive._copy_file

    def flaky_copy_file(src, dst):
        if src.name == "video.mp4":
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"TRUNCATED")   # 故意写得比源文件短，模拟拷贝损坏
        else:
            real_copy_file(src, dst)

    monkeypatch.setattr(d10_archive, "_copy_file", flaky_copy_file)

    run(ctx)

    arc = read_json(ctx.job, OUTPUT)
    assert arc["status"] == "failed"
    assert "video.mp4" in arc["reason"]

    # 本地一律没删——校验没过，本地是唯一完整副本
    assert (ctx.job / "video.mp4").exists()
    assert (ctx.job / "source.mp4").exists()
    assert (ctx.job / "narration").is_dir()
    assert ctx.warnings


# ---------------------------------------------------------------------------
# 7. P7 T16：已发布登记表也备份一份到 NAS 根目录
# ---------------------------------------------------------------------------

def test_published_registry_copied_to_archive_root_on_success(tmp_path):
    from src.util.jobs import write_json as _write_json

    archive_root = tmp_path / "nas"
    archive_root.mkdir()
    ctx = _make_job(tmp_path, Config({"archive": {"dir": str(archive_root)}}))
    # jobs_root/published.json 是 ctx.root 层级的文件，不是 job 目录里的
    _write_json(ctx.root, "published.json", [
        {"identifier": "id1", "title": "T1", "videoId": "V1", "slug": "c",
         "published_at": "2026-01-01T00:00:00+00:00"},
    ])

    run(ctx)

    arc = read_json(ctx.job, OUTPUT)
    assert arc["status"] == "archived"

    backup = archive_root / "published.json"
    assert backup.exists()
    assert read_json(archive_root, "published.json") == [
        {"identifier": "id1", "title": "T1", "videoId": "V1", "slug": "c",
         "published_at": "2026-01-01T00:00:00+00:00"},
    ]


def test_missing_published_registry_does_not_fail_archive(tmp_path):
    """jobs_root/published.json 压根不存在（原始管线/首次跑）——不该报错，
    也不该在 NAS 上凭空造出一个空文件。"""
    archive_root = tmp_path / "nas"
    archive_root.mkdir()
    ctx = _make_job(tmp_path, Config({"archive": {"dir": str(archive_root)}}))
    assert not (ctx.root / "published.json").exists()

    run(ctx)

    arc = read_json(ctx.job, OUTPUT)
    assert arc["status"] == "archived"
    assert not (archive_root / "published.json").exists()
