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
    write_json(ctx.job, OUTPUT, result)
