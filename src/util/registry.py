"""P7 T16：已发布纪录片登记表——防止 d1_source 同题材重复选片/重复发布。

背景：d1_source 按 downloads desc 排序取第一条过门候选，同一条 query 每次
run 起手拿到的候选列表几乎不变——不登记的话，同一部片会被反复选中、反复
下载/翻译/配音/发布，产出一堆重复视频。

登记表落在 `jobs_root/published.json`（一个 JSON 数组，dict 元素）。每条
记录一部已成功上传到 YouTube 的源纪录片：
    {"identifier", "title", "videoId", "slug", "published_at"}

两处联动：
  - `d1_source` 选片前用 `published_identifiers()` 排除已发布的，
    再用 `in_flight_identifiers()` 排除同批其它 job 目录里已下载但还没
    发布完的（`source.json` 存在，说明别的 case 已经挑过这部片，即便它
    还没跑到 d9_publish 也不该被再挑一次）。
  - `s9_publish` 上传成功后调用 `append_published()` 登记。
  - `d10_archive` 把 published.json 一并备份到 NAS（本地文件是唯一权威
    副本，NAS 备份只是灾备，读它不经过这个模块）。

`append_published()` 是 read-modify-write，不是并发安全的——本项目单进程
顺序跑 case，没有并发写场景。写入本身复用 `src.util.jobs.write_json` 的
tempfile+os.replace 原子替换语义，不会因为写到一半崩溃而留半个文件。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.util.jobs import write_json

REGISTRY_NAME = "published.json"

# 多机共享(用户定 2026-07-12):设 env YOUTUBER_REGISTRY=<NAS上published.json绝对路径>
# 则 zion/pluto 读写同一份 NAS 注册表,dedup 不再分裂。留空则退回本机 jobs_root/。
_ENV_REGISTRY = "YOUTUBER_REGISTRY"

# d1_source.META_OUTPUT 的字面值。此处不 import src.stages.d1_source ——
# 那样会和 d1_source 反过来 import 本模块形成循环 import。
_SOURCE_META = "source.json"


def registry_path(jobs_root: Path) -> Path:
    shared = os.environ.get(_ENV_REGISTRY, "").strip()
    if shared:
        return Path(shared).expanduser()
    return jobs_root / REGISTRY_NAME


def load_published(jobs_root: Path) -> list[dict]:
    """读登记表；文件缺失/损坏一律当空表处理，绝不因此中止 run。"""
    p = registry_path(jobs_root)
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict)]


def published_identifiers(jobs_root: Path) -> set[str]:
    return {e["identifier"] for e in load_published(jobs_root) if e.get("identifier")}


def append_published(jobs_root: Path, entry: dict) -> None:
    """read-modify-write 一条新记录进登记表；写入本身原子替换。
    共享模式(env YOUTUBER_REGISTRY)下写到 NAS 绝对路径,并 chmod 0666 让另一台
    机器(不同 uid,如 pluto=root/zion=fabric)也能读写同一份。"""
    path = registry_path(jobs_root)
    entries = load_published(jobs_root)
    entries.append(entry)
    write_json(path.parent, path.name, entries)     # tempfile 落在注册表同目录,原子替换有效
    try:
        os.chmod(path, 0o666)                        # 跨用户可读写(NAS 共享必需)
    except OSError:
        pass


def in_flight_identifiers(jobs_root: Path, exclude_slug: str) -> set[str]:
    """扫同级 job 目录（排除自己）的 source.json，收集它们已下载但未必
    已发布的 identifier——抓"已下载但还没登记发布"这类在途重复。目录不存在、
    单个 source.json 缺失/损坏一律跳过，不中止扫描。"""
    ids: set[str] = set()
    if not jobs_root.is_dir():
        return ids
    for child in jobs_root.iterdir():
        if not child.is_dir() or child.name == exclude_slug:
            continue
        src = child / _SOURCE_META
        if not src.is_file():
            continue
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("identifier"):
            ids.add(data["identifier"])
    return ids
