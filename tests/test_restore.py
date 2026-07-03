"""任务#50:restore_from_archive——d10 归档裁剪后各阶段自动从 NAS 找回大媒体。
全离线:归档目录就是 tmp_path 下的普通目录,不碰真 NFS。"""
from __future__ import annotations

from pathlib import Path

from src.config import Config
from src.util.restore import restore_from_archive


def _setup(tmp_path) -> tuple[Path, Config, list[str]]:
    job = tmp_path / "jobs" / "space-doc-99"
    job.mkdir(parents=True)
    archive = tmp_path / "nas" / "space-doc-99"
    archive.mkdir(parents=True)
    warns: list[str] = []
    cfg = Config({"archive": {"dir": str(tmp_path / "nas")}})
    return job, archive, cfg, warns


def test_local_file_present_returns_true_without_touching_archive(tmp_path):
    job, archive, cfg, warns = _setup(tmp_path)
    (job / "video.mp4").write_bytes(b"LOCAL")

    assert restore_from_archive(job, "video.mp4", cfg, warns.append) is True
    assert (job / "video.mp4").read_bytes() == b"LOCAL"   # 没被归档版覆盖
    assert not warns


def test_missing_local_restores_from_archive_and_warns(tmp_path):
    job, archive, cfg, warns = _setup(tmp_path)
    (archive / "video.mp4").write_bytes(b"FROM_NAS")

    assert restore_from_archive(job, "video.mp4", cfg, warns.append) is True
    assert (job / "video.mp4").read_bytes() == b"FROM_NAS"
    assert any("恢复" in w for w in warns)
    assert not (job / "video.mp4.part").exists()          # 临时文件不残留


def test_missing_everywhere_returns_false_silently(tmp_path):
    job, archive, cfg, warns = _setup(tmp_path)

    assert restore_from_archive(job, "video.mp4", cfg, warns.append) is False
    assert not (job / "video.mp4").exists()
    assert not warns   # 找不到不告警——调用方按"文件确实没有"的既有路径处理


def test_archive_dir_unset_returns_false(tmp_path):
    job, archive, _, warns = _setup(tmp_path)
    (archive / "video.mp4").write_bytes(b"FROM_NAS")
    cfg = Config({"archive": {"dir": ""}})

    assert restore_from_archive(job, "video.mp4", cfg, warns.append) is False
