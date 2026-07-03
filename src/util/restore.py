"""从 NAS 归档恢复被 d10 `trim_local` 裁掉的本地大媒体文件(任务#50)。

d10_archive 校验归档后会删本地大媒体(source.mp4/video.mp4/vocals.mp3 等,
*.json/*.ass/thumb.jpg 留作台账)。之后任何重跑/重传(d7 重制、d9 重传、
d9b B站重投、d9c 英文版重混)都会缺这些文件——此前只能手工 cp 回来
(doc-05 d7 重制、英文版换乐批量都踩过)。本模块把"缺了就去归档找"变成
各阶段的标准前置动作。

失败语义:归档没配/文件不在归档/复制失败 → False,不 raise——调用方按
"文件确实没有"的既有路径走(该 fail 的 fail,该 warn 的 warn),恢复只是
锦上添花。复制先落 .part 临时名、校验大小后原子改名,避免半截文件被当成品。
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

# 与 d10_archive.DEFAULT_ARCHIVE_DIR 同值。刻意复制字面量而非 import:
# util 层不 import stages 层,避免反向依赖(生产配置反正都显式配了 archive.dir)。
DEFAULT_ARCHIVE_DIR = "~/nas/Share/youtuber"


def restore_from_archive(job: Path, filename: str, config,
                         warn: Callable[[str], None]) -> bool:
    """job/filename 本地存在 → True;缺失则尝试从 archive.dir/<slug>/ 复制回来。"""
    target = job / filename
    if target.exists():
        return True
    raw = config.get("archive.dir", DEFAULT_ARCHIVE_DIR) or ""
    if not raw:
        return False
    src = Path(raw).expanduser() / job.name / filename
    if not src.exists():
        return False
    try:
        tmp = target.with_name(target.name + ".part")
        shutil.copyfile(src, tmp)
        if tmp.stat().st_size != src.stat().st_size:   # NFS 半截读的保险
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(target)
    except OSError:
        return False
    warn(f"{filename} 本地缺失(归档已裁剪),已从 NAS 归档恢复")
    return True
