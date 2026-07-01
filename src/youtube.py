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

    creds = Credentials.from_authorized_user_file(token, scopes)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
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
        if thumb and Path(thumb).exists():
            self._yt.thumbnails().set(
                videoId=vid, media_body=MediaFileUpload(str(thumb))).execute()
        return {"videoId": vid, "url": f"https://youtu.be/{vid}", "status": visibility}


def get_uploader(config: Config) -> Uploader:
    provider = config.get("youtube.provider", "youtube")
    if provider == "fake":
        return FakeUploader()
    if provider == "youtube":
        creds = load_creds(config.get("youtube.token", "secrets/token.json"),
                           config.get("youtube.client_secret", "secrets/client_secret.json"))
        return YouTubeUploader(creds)
    raise ValueError(f"未知 youtube provider: {provider}")
