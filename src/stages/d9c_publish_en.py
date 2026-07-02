"""P7 T20：阶段⑨c——追加发布英文原声源片（未配音的 `source.mp4`，非中文配音的
`video.mp4`）到第二个 YouTube 账号/频道（英文原片档案频道）。

跟 d9b_bilibili 同一个 best-effort 失败隔离思路：这是「主发布之外」的追加发布，
绝不能拖累已经成功的中文配音版 YouTube 发布（s9_publish/d9_publish）、B 站发布
（d9b_bilibili）或归档（d10_archive）——三者都已跑完/在跑，这条只是锦上添花。

第二账号 = 第二份 OAuth token 文件（默认 `secrets/token_en.json`），跟主频道
（`youtube.token`，默认 `secrets/token.json`）用同一份 `client_secret.json`，
只是浏览器授权时选了另一个 Google 账号：
    python -m src.auth_youtube secrets/token_en.json

刻意不用 `src.youtube.get_uploader`——那个读的是 `youtube.*`（主频道配置），
这里要读独立的 `youtube_en.*`，所以本模块自己有一个小的 `_uploader_for`，
形状照抄 get_uploader（`youtube_en.provider` 默认 "youtube"，"fake" 走
FakeUploader 供测试离线注入，不联网）。

失败语义：
  - `youtube_en.enabled` 不是 true（默认 false，笔记本关；zion 生产按需开）→
    status="disabled"，直接返回，不碰任何文件/网络。
  - token 文件不存在（还没走一次性授权，provider=youtube 时才检查；
    provider=fake 测试不需要 token 文件）→ status="no_login"，只 warn 提示
    先跑 `python -m src.auth_youtube secrets/token_en.json` 选第二个 Google
    账号，不 raise。
  - 上传失败（网络抖动/配额/权限/source.json 缺失等任何异常）→
    status="failed"，带 error 摘要，只 warn，绝不 raise——注意这跟
    s9_publish 不同：s9_publish 是主发布，失败要让 run 中止；这里是 d9b_bilibili
    那种「追加发布」，失败只留标记。
  - 成功 → status="published"，带 videoId/url/visibility；playlist 配了则
    尝试加入，失败走 playlist_error（不作废已成功的上传，同 s9_publish 的模式）。

护栏#3：visibility 从不默认 public——本模块默认 unlisted（英文源片频道走
档案/引流定位，比中文配音频道更保守也不到 private）。
"""
from __future__ import annotations

from pathlib import Path

from src.config import Config
from src.context import Ctx
from src.util.jobs import read_json, write_json
from src.youtube import FakeUploader, Uploader, YouTubeUploader, load_creds

OUTPUT = "publish_en.json"
INPUT = "source.json"          # d1_source.META_OUTPUT

_TITLE_MAX_CHARS = 100         # YouTube 标题上限
_DEFAULT_TAGS = ["NASA", "space", "documentary", "public domain", "archive"]


def _uploader_for(config: Config) -> Uploader:
    """形状照抄 src.youtube.get_uploader，但读 youtube_en.* 而非 youtube.*。"""
    provider = config.get("youtube_en.provider", "youtube")
    if provider == "fake":
        return FakeUploader()
    if provider == "youtube":
        creds = load_creds(
            config.get("youtube_en.token", "secrets/token_en.json"),
            config.get("youtube_en.client_secret", "secrets/client_secret.json"),
        )
        return YouTubeUploader(creds)
    raise ValueError(f"未知 youtube_en provider: {provider}")


def _build_meta(source: dict) -> dict:
    title = (source.get("title") or "")[:_TITLE_MAX_CHARS]
    identifier = source.get("identifier")
    attribution = source.get("attribution", "")
    archive_url = f"https://archive.org/details/{identifier}" if identifier else ""
    description = (
        f"{attribution}\n\n"
        f"Public domain source: {archive_url}\n\n"
        "This channel curates public-domain space & science films."
    )
    return {"title": title, "description": description, "tags": list(_DEFAULT_TAGS)}


def run(ctx: Ctx) -> None:
    if not ctx.config.get("youtube_en.enabled", False):
        write_json(ctx.job, OUTPUT, {"status": "disabled"})
        return

    provider = ctx.config.get("youtube_en.provider", "youtube")
    token = ctx.config.get("youtube_en.token", "secrets/token_en.json")
    if provider != "fake" and not Path(token).exists():
        ctx.warn(
            f"{ctx.slug}: 英文原片频道未授权（缺 {token}），先跑 "
            f"`python -m src.auth_youtube {token}` 选第二个 Google 账号完成一次性授权，"
            "本次跳过英文原片发布"
        )
        write_json(ctx.job, OUTPUT, {"status": "no_login"})
        return

    visibility = ctx.config.get("youtube_en.visibility", "unlisted")  # 护栏#3: 从不默认公开

    try:
        source = read_json(ctx.job, INPUT)
        meta = _build_meta(source)
        uploader = _uploader_for(ctx.config)
        video = ctx.job / "source.mp4"                # 英文原声原片，非中文配音的 video.mp4
        thumb = ctx.job / "thumb_bg.png"               # AI 背景图(无中文字)；不存在时 uploader 自动跳过封面
        result = uploader.upload(video, meta, thumb, visibility)
    except Exception as e:  # noqa: BLE001 — 追加发布失败绝不能拖累主发布/B站发布/归档链条
        ctx.warn(f"{ctx.slug}: 英文原片上传失败({type(e).__name__}: {e})")
        write_json(ctx.job, OUTPUT, {"status": "failed", "error": f"{type(e).__name__}: {e}"})
        return

    result["status"] = "published"
    result["visibility"] = visibility

    playlist_title = ctx.config.get("youtube_en.playlist", "")
    if playlist_title:
        try:
            playlist_id = uploader.add_to_playlist(result["videoId"], playlist_title)
            result["playlist"] = playlist_title
            result["playlistId"] = playlist_id
        except Exception as e:  # noqa: BLE001 — 播放列表失败不能作废已成功的上传
            result["playlist_error"] = str(e)
            ctx.warn(f"{ctx.slug}: 英文频道加入播放列表 {playlist_title!r} 失败：{e}")

    write_json(ctx.job, OUTPUT, result)
