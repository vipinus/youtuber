"""P7 阶段⑨d:中英视频简介双向互链(2026-07-06 用户定)。

独立成最后一步(在两个 YouTube 发布之后、归档之前),与发布顺序完全解耦——
不管中文先发还是英文先发,跑到这里 publish.json(中文配音版)+ publish_en.json
(英文原版)都已就位,各自把对方链接追加进简介:
  - 中文视频简介 += 英文原版链接
  - 英文视频简介 += 中文配音版链接

best-effort(同 d9b/d9c):任一 provider=fake/未授权/API 失败/缺台账都只 warn,
绝不中止 run,也不作废已成功的发布。幂等:链接已在简介里就跳过,重跑不重复追加。
"""
from __future__ import annotations

from src.context import Ctx
from src.util.jobs import read_json, write_json

OUTPUT = "crosslink.json"
ZH_PUB = "publish.json"
EN_PUB = "publish_en.json"
ZH_META = "meta.json"

_ZH_LINE = "\n\n🎬 英文原版 / English original：{url}"
_EN_LINE = "\n\n🎬 中文配音版 / Chinese dubbed version：{url}"


def _url(pub: dict) -> str:
    return pub.get("url") or (f"https://youtu.be/{pub['videoId']}" if pub.get("videoId") else "")


def _both_published(ctx: Ctx) -> tuple[dict, dict] | None:
    if not (ctx.job / ZH_PUB).exists() or not (ctx.job / EN_PUB).exists():
        return None
    try:
        zh, en = read_json(ctx.job, ZH_PUB), read_json(ctx.job, EN_PUB)
    except Exception:  # noqa: BLE001
        return None
    if not zh.get("videoId") or not en.get("videoId"):
        return None
    if zh.get("status") in ("retracted", "disabled") or en.get("status") in ("disabled", "no_login", "failed"):
        return None
    return zh, en


def run(ctx: Ctx) -> None:
    pair = _both_published(ctx)
    if pair is None:
        write_json(ctx.job, OUTPUT, {"status": "skipped", "reason": "缺中文或英文发布"})
        return
    zh, en = pair
    zh_url, en_url = _url(zh), _url(en)
    result: dict = {"status": "done", "zh_url": zh_url, "en_url": en_url, "linked": []}

    # 中文视频 += 英文链接(主频道 uploader)
    try:
        from src.youtube import get_uploader
        zh_up = get_uploader(ctx.config)
        base = read_json(ctx.job, ZH_META).get("description", "") if (ctx.job / ZH_META).exists() else ""
        if en_url and en_url not in base:
            zh_up.update_description(zh["videoId"], base + _ZH_LINE.format(url=en_url))
            result["linked"].append("zh")
    except Exception as e:  # noqa: BLE001
        ctx.warn(f"{ctx.slug}: 回填中文视频简介失败({type(e).__name__}: {e})")
        result["zh_error"] = f"{type(e).__name__}: {e}"

    # 英文视频 += 中文链接(第二频道 uploader,形状同 d9c._uploader_for)
    try:
        from src.stages.d9c_publish_en import _build_meta, _uploader_for
        en_up = _uploader_for(ctx.config)
        en_base = _build_meta(read_json(ctx.job, "source.json")).get("description", "") \
            if (ctx.job / "source.json").exists() else ""
        if zh_url and zh_url not in en_base:
            en_up.update_description(en["videoId"], en_base + _EN_LINE.format(url=zh_url))
            result["linked"].append("en")
    except Exception as e:  # noqa: BLE001
        ctx.warn(f"{ctx.slug}: 回填英文视频简介失败({type(e).__name__}: {e})")
        result["en_error"] = f"{type(e).__name__}: {e}"

    write_json(ctx.job, OUTPUT, result)
