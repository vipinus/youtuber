from __future__ import annotations

from src.context import Ctx
from src.util.jobs import read_json, write_json
from src.youtube import get_uploader

OUTPUT = "publish.json"
INPUT = "meta.json"


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

    write_json(ctx.job, OUTPUT, result)
