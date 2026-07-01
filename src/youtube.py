from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.config import Config

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


class Uploader(Protocol):
    def upload(self, video: Path, meta: dict, thumb: Path, visibility: str) -> dict: ...


class FakeUploader:
    """离线：确定性返回假 videoId，不联网。"""

    def upload(self, video: Path, meta: dict, thumb: Path, visibility: str) -> dict:
        return {"videoId": "FAKE12345", "url": "https://youtu.be/FAKE12345",
                "status": visibility}


def load_creds(token: str, client_secret: str, scopes=SCOPES):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    if not Path(token).exists():
        raise FileNotFoundError(
            f"未找到 YouTube 授权文件 {token}；请先跑一次性授权："
            f"python -m src.auth_youtube（详见 docs/youtube-oauth-setup.md）")
    creds = Credentials.from_authorized_user_file(token, scopes)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        Path(token).write_text(creds.to_json(), encoding="utf-8")   # 回写刷新后的 token，次日/长期无人值守仍可用
    return creds


class YouTubeUploader:
    """YouTube Data API v3。惰性 import googleapiclient。"""

    def __init__(self, creds):
        from googleapiclient.discovery import build

        self._yt = build("youtube", "v3", credentials=creds)

    def upload(self, video: Path, meta: dict, thumb: Path, visibility: str) -> dict:
        from googleapiclient.http import MediaFileUpload

        body = {
            "snippet": {"title": meta.get("title", ""),
                        "description": meta.get("description", ""),
                        "tags": meta.get("tags", []), "categoryId": "22"},
            "status": {"privacyStatus": visibility, "selfDeclaredMadeForKids": False},
        }
        req = self._yt.videos().insert(
            part="snippet,status", body=body,
            media_body=MediaFileUpload(str(video), resumable=True))
        resp = req.execute()
        vid = resp["id"]
        result = {"videoId": vid, "url": f"https://youtu.be/{vid}", "status": visibility}
        # 视频已上传成功；封面失败（如频道未手机验证→403）不能吞掉已成功的 videoId，
        # 否则阶段⑨ 写不出 publish.json，就丢失了这条已经存在于 YouTube 的视频。
        if thumb and Path(thumb).exists():
            try:
                self._yt.thumbnails().set(
                    videoId=vid, media_body=MediaFileUpload(str(thumb))).execute()
            except Exception as e:  # noqa: BLE001 — 封面失败不应作废整次上传
                result["thumbnail_error"] = str(e)
        return result


def get_uploader(config: Config) -> Uploader:
    provider = config.get("youtube.provider", "youtube")
    if provider == "fake":
        return FakeUploader()
    if provider == "youtube":
        creds = load_creds(config.get("youtube.token", "secrets/token.json"),
                           config.get("youtube.client_secret", "secrets/client_secret.json"))
        return YouTubeUploader(creds)
    raise ValueError(f"未知 youtube provider: {provider}")
