from __future__ import annotations

from datetime import datetime, timezone

from src.context import Ctx
from src.util.jobs import read_json, write_json
from src.util.registry import append_published
from src.youtube import get_uploader

OUTPUT = "publish.json"
INPUT = "meta.json"
SOURCE_META = "source.json"        # d1_source.META_OUTPUT——存在即 doc 管线


def run(ctx: Ctx) -> None:
    video = ctx.job / "video.mp4"
    meta_path = ctx.job / INPUT

    if not ctx.config.get("publish.auto_upload", False):        # 护栏#3: 冷启动安全档
        ctx.warn(f"{ctx.slug}: publish.auto_upload=false，跳过上传")
        write_json(ctx.job, OUTPUT, {
            "status": "skipped",
            "video": str(video),
            "meta": str(meta_path),
        })
        return

    meta = read_json(ctx.job, INPUT)
    visibility = ctx.config.get("publish.visibility", "private")  # 护栏#3: 从不默认公开
    uploader = get_uploader(ctx.config)
    result = uploader.upload(video, meta, ctx.job / "thumb.jpg", visibility)

    playlist_title = ctx.config.get("publish.playlist", "")
    if playlist_title:
        try:
            playlist_id = uploader.add_to_playlist(result["videoId"], playlist_title)
            result["playlist"] = playlist_title
            result["playlistId"] = playlist_id
        except Exception as e:  # noqa: BLE001 — 播放列表失败不能作废已成功的上传
            msg = str(e)
            # scope 从 .upload 换成完整 youtube 后，老 token 会 403 insufficientPermissions；
            # 这种情况提示用户重新走一次授权，而不是让 run 直接挂掉。
            hint = ("；权限不足，请重新执行 python -m src.auth_youtube 重新授权（补齐播放列表权限）"
                    if "insufficient" in msg.lower() else "")
            result["playlist_error"] = msg + hint
            ctx.warn(f"{ctx.slug}: 加入播放列表 {playlist_title!r} 失败：{msg}{hint}")

    if result.get("videoId") and (ctx.job / SOURCE_META).is_file():
        # P7 T16：doc 管线（有 source.json）上传成功后登记进已发布表，供
        # d1_source 选片时去重；原始 s1-s9 管线没有 source.json，静默跳过。
        try:
            source = read_json(ctx.job, SOURCE_META)
            append_published(ctx.root, {
                "identifier": source.get("identifier"),
                "title": source.get("title"),
                "videoId": result["videoId"],
                "slug": ctx.slug,
                "published_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:  # noqa: BLE001 — 登记失败不能作废已成功的上传
            ctx.warn(f"{ctx.slug}: 登记已发布表失败({type(e).__name__}: {e})，不影响本次上传结果")

    write_json(ctx.job, OUTPUT, result)
