"""P7 T19：阶段⑨b——追加发布到 B 站（YouTube 之外的第二平台）。

d9_publish 把成片传到 YouTube 之后，用 `biliup` CLI（biliup-rs 风格，见
`~/bili-venv/bin/biliup --help`）把同一份视频再传一份到 B 站。B 站登录态是
CWD 下的 cookies.json（`~/youtuber-secrets/cookies.json`，用户已一次性
QR 扫码登录过），本阶段必须在该目录下起 subprocess——biliup 认 CWD 找
cookies.json，不能想着传别的目录当登录态。

内容定位：本频道把公有领域纪录片翻译配音发布，属于「转载」而非原创，固定传
`--copyright 2`（转载）+ `--source <archive.org 源片地址>`——诚实标注来源，
符合 B 站转载规范，也避免被误判原创引发版权纠纷。

失败语义（跟 d10_archive 一路：best-effort，绝不能让 B 站这条追加发布的问题
拖累已经成功的 YouTube 发布/登记表/归档流程——YouTube 发布已经成功了，B 站
只是锦上添花）：
  - `bilibili.enabled` 不是 true（默认 false，笔记本关；zion 生产开）→
    status="disabled"，直接返回，完全不碰 subprocess。
  - cookie 文件不存在（还没走一次性 QR 登录）→ status="no_login"，只
    warn 提示先登录，不 raise。
  - biliup 子进程起不来/超时/rc!=0（网络抖动、审核拒绝、参数错）→
    status="failed"，带 stderr 尾巴，只 warn，绝不 raise。
  - rc==0 → status="published"，尝试从 stdout 里挖 BV 号；挖不到就
    bvid=None（rc==0 但没匹配到号，算「提交成功但号未知」，不算失败）。
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from src.context import Ctx
from src.util.jobs import read_json, write_json

OUTPUT = "publish_bilibili.json"
INPUT = "meta.json"
SOURCE_META = "source.json"        # d1_source.META_OUTPUT——有 source_url 供 --source 标注转载来源

_BV_RE = re.compile(r"BV[0-9A-Za-z]{8,}")
_TITLE_MAX_CHARS = 80              # B 站标题上限
_TAG_MAX_CHARS = 20                # B 站单个 tag 上限
_TAGS_MAX_COUNT = 10               # B 站 tag 数量上限
_DEFAULT_TIMEOUT_SEC = 3600
_DEFAULT_SOURCE_URL = "https://archive.org"


def _clamp_title(title: str) -> str:
    return (title or "")[:_TITLE_MAX_CHARS]


def _clamp_tags(tags: list) -> list[str]:
    out: list[str] = []
    for t in tags or []:
        t = str(t).strip()[:_TAG_MAX_CHARS]
        if not t:
            continue
        out.append(t)
        if len(out) >= _TAGS_MAX_COUNT:
            break
    return out


def run(ctx: Ctx) -> None:
    if not ctx.config.get("bilibili.enabled", False):
        write_json(ctx.job, OUTPUT, {"status": "disabled"})
        return

    cookie_dir = Path(ctx.config.get("bilibili.cookie_dir", "~/youtuber-secrets")).expanduser()
    cookie_file = cookie_dir / "cookies.json"
    if not cookie_file.is_file():
        ctx.warn(
            f"{ctx.slug}: B 站未登录（缺 {cookie_file}），先在 {cookie_dir} 下跑一次 "
            "`biliup login` 完成扫码登录，本次跳过 B 站发布"
        )
        write_json(ctx.job, OUTPUT, {"status": "no_login"})
        return

    meta = read_json(ctx.job, INPUT)
    source = read_json(ctx.job, SOURCE_META) if (ctx.job / SOURCE_META).is_file() else {}

    # 重投场景:d10 归档后本地 video.mp4 被裁剪 → 自动从 NAS 找回(任务#50)
    from src.util.restore import restore_from_archive
    restore_from_archive(ctx.job, "video.mp4", ctx.config,
                         lambda m: ctx.warn(f"{ctx.slug}: {m}"))

    biliup_bin = Path(ctx.config.get("bilibili.biliup_bin", "~/bili-venv/bin/biliup")).expanduser()
    tid = ctx.config.get("bilibili.tid", 201)
    only_self = ctx.config.get("bilibili.only_self", True)
    submit = ctx.config.get("bilibili.submit", "web")   # app 接口对新号风控严(21566),web 接口通过率高
    line = ctx.config.get("bilibili.line", "")
    timeout_sec = ctx.config.get("bilibili.timeout_sec", _DEFAULT_TIMEOUT_SEC)

    title = _clamp_title(meta.get("title", ""))
    tags = _clamp_tags(meta.get("tags", []))
    # 优先 /details/ 页面(人可读,利于B站审核核验来源);无 identifier 才退直链
    identifier = source.get("identifier")
    source_url = (f"https://archive.org/details/{identifier}" if identifier
                  else source.get("source_url") or _DEFAULT_SOURCE_URL)

    # 绝对路径:biliup 以 cookie_dir 为 cwd 跑,相对路径会解析错位(真跑踩坑:封面 No such file)
    cmd = [
        str(biliup_bin), "upload", str((ctx.job / "video.mp4").resolve()),
        "--title", title,
        "--desc", meta.get("description", ""),
        "--tag", ",".join(tags),
        "--cover", str((ctx.job / "thumb.jpg").resolve()),
        "--tid", str(tid),
        "--copyright", "2",
        "--source", source_url,
        "--is-only-self", "1" if only_self else "0",
        "--submit", str(submit),
    ]
    if line:
        cmd += ["--line", str(line)]

    try:
        result = subprocess.run(
            cmd, cwd=str(cookie_dir), capture_output=True, text=True, timeout=timeout_sec,
        )
    except Exception as e:  # noqa: BLE001 — B 站失败绝不能拖累已成功的 YouTube 发布
        ctx.warn(f"{ctx.slug}: B 站上传子进程异常({type(e).__name__}: {e})")
        write_json(ctx.job, OUTPUT, {"status": "failed", "error": f"{type(e).__name__}: {e}"})
        return

    if result.returncode != 0:
        tail = (result.stderr or "")[-2000:]
        ctx.warn(f"{ctx.slug}: B 站上传失败(rc={result.returncode})：{tail}")
        write_json(ctx.job, OUTPUT, {"status": "failed", "error": tail})
        return

    match = _BV_RE.search(result.stdout or "")
    bvid = match.group(0) if match else None
    write_json(ctx.job, OUTPUT, {
        "status": "published",
        "bvid": bvid,
        "url": f"https://www.bilibili.com/video/{bvid}" if bvid else None,
        "only_self": bool(only_self),
    })
