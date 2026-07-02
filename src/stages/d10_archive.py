"""P7 T15：阶段⑩——发布后归档到 NAS。

d9_publish 跑完、视频已经传上 YouTube 之后，把整个 job 目录（源片、中间产物、
成片、字幕……约 1.5GB/集）搬一份到 NAS（zion 上 NFS 自动挂载的 tatara，写入根
`~/nas/Share/youtuber`），本地只留小文件（*.json/*.ass/thumb.jpg）当轻量台账，
腾出磁盘给下一集。

归档失败绝不能让 run 挂掉——视频已经发布了，归档只是「best-effort，失败也要
留个可辨认的标记」：
  - `archive.dir` 没配（空/None）→ status="skipped"，直接返回。
  - `archive.dir` 配了但目录本身不存在/不可写（dev 笔记本没挂 NFS 就是这种
    情况——`~/nas` 压根不存在）→ status="unreachable"。绝不 mkdir 出一棵假的
    本地目录树冒充 NAS——那样会把上百 MB 媒体悄悄写进开发机的家目录。
  - 复制完但校验（逐文件核对目标存在 + 大小一致）没过 → status="failed"，
    且**不裁剪本地文件**（本地此时是唯一还完整的副本）。
  - 复制 + 校验都过 → status="archived"；若 `archive.trim_local`（默认 true）
    则删本地大媒体文件，*.json/*.ass/thumb.jpg 留本地。

unreachable/failed 都会照常写 OUTPUT（不能不写——不写的话编排器的
skip-if-exists 会导致下次 `python -m src.run` 又把它当"没跑过"重新执行前面
所有阶段）；真要重试归档本身，用 `--from d10_archive --force`。
"""
from __future__ import annotations

import shutil
from pathlib import Path

from src.context import Ctx
from src.util.jobs import write_json

OUTPUT = "archive.json"
INPUT = "publish.json"

DEFAULT_ARCHIVE_DIR = "~/nas/Share/youtuber"

# 归档校验通过后才敢删的本地大媒体（NAS 上已有一份校验过的拷贝）。
# *.json/*.ass/thumb.jpg 之类小文件不在此列，留在本地当台账，不用挂 NAS 也能翻。
_TRIM_FILES = (
    "source.mp4", "video.mp4", "source_audio.wav", "narration.wav",
    "no_vocals.mp3", "thumb_bg.png",
)
_TRIM_DIRS = ("narration", "media", "audio")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_job_dir(job: Path, dest: Path) -> list[Path]:
    """把 job 目录内容（除 OUTPUT 自身外）复制到 dest，保留相对目录结构；
    返回复制过的相对路径列表（相对于 job），供随后校验 + 统计用。"""
    copied: list[Path] = []
    output_rel = Path(OUTPUT)
    for src in sorted(job.rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(job)
        if rel == output_rel:
            continue                       # 不把上一次归档留下的 archive.json 也搬过去
        _copy_file(src, dest / rel)
        copied.append(rel)
    return copied


def _verify(job: Path, dest: Path, files: list[Path]) -> str:
    """逐文件核对目标存在 + 大小一致；全过返回空串，否则返回失败原因。"""
    for rel in files:
        dst_p = dest / rel
        if not dst_p.exists():
            return f"目标缺失：{rel}"
        src_size = (job / rel).stat().st_size
        dst_size = dst_p.stat().st_size
        if src_size != dst_size:
            return f"大小不符：{rel}（源 {src_size} / 目标 {dst_size}）"
    return ""


def _trim_local(job: Path) -> list[str]:
    trimmed: list[str] = []
    for name in _TRIM_FILES:
        p = job / name
        if p.is_file():
            p.unlink()
            trimmed.append(name)
    for name in _TRIM_DIRS:
        p = job / name
        if p.is_dir():
            shutil.rmtree(p)
            trimmed.append(name + "/")
    return trimmed


def run(ctx: Ctx) -> None:
    raw_dir = ctx.config.get("archive.dir", DEFAULT_ARCHIVE_DIR)
    if not raw_dir:
        ctx.warn(f"{ctx.slug}: 归档未配置，跳过")
        write_json(ctx.job, OUTPUT, {"status": "skipped", "reason": "archive.dir 未配置"})
        return

    archive_root = Path(raw_dir).expanduser()
    if not archive_root.is_dir():
        ctx.warn(f"{ctx.slug}: 归档目录不可达({archive_root})，跳过归档"
                  "（不会自动创建假的本地目录冒充 NAS；--force 重跑可重试）")
        write_json(ctx.job, OUTPUT, {
            "status": "unreachable", "dest": str(archive_root / ctx.slug),
            "reason": f"archive.dir 不存在或不可达：{archive_root}",
        })
        return

    dest = archive_root / ctx.slug
    try:
        dest.mkdir(parents=True, exist_ok=True)
        copied = _copy_job_dir(ctx.job, dest)
    except OSError as e:
        ctx.warn(f"{ctx.slug}: 归档写入失败({type(e).__name__}: {e})，跳过（--force 重跑可重试）")
        write_json(ctx.job, OUTPUT, {
            "status": "unreachable", "dest": str(dest),
            "reason": f"{type(e).__name__}: {e}",
        })
        return

    fail_reason = _verify(ctx.job, dest, copied)
    if fail_reason:
        ctx.warn(f"{ctx.slug}: 归档校验失败({fail_reason})，不裁剪本地文件")
        write_json(ctx.job, OUTPUT, {
            "status": "failed", "dest": str(dest), "files": len(copied),
            "reason": fail_reason,
        })
        return

    total_bytes = sum((ctx.job / rel).stat().st_size for rel in copied)
    trimmed: list[str] = []
    if ctx.config.get("archive.trim_local", True):
        trimmed = _trim_local(ctx.job)

    write_json(ctx.job, OUTPUT, {
        "status": "archived",
        "dest": str(dest),
        "files": len(copied),
        "bytes": total_bytes,
        "trimmed": trimmed,
    })
