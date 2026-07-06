from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.config import Config

# 需要完整的 youtube scope 才能管理播放列表（playlists/playlistItems）；
# 仅 .upload 做不到。该完整 scope 已覆盖上传权限，所以是替换而非追加。
# 换 scope 后老 token 权限不够，playlist 调用会 403 insufficientPermissions——
# 见 src/stages/s9_publish.py 的容错与 docs/youtube-oauth-setup.md 的重新授权说明。
SCOPES = ["https://www.googleapis.com/auth/youtube"]


class Uploader(Protocol):
    def upload(self, video: Path, meta: dict, thumb: Path, visibility: str) -> dict: ...
    def add_to_playlist(self, video_id: str, playlist_title: str) -> str: ...
    def update_description(self, video_id: str, description: str) -> None: ...


class FakeUploader:
    """离线：确定性返回假 videoId/playlistId，不联网。"""

    def __init__(self):
        self.playlist_calls: list[tuple[str, str]] = []
        self.description_updates: list[tuple[str, str]] = []

    def upload(self, video: Path, meta: dict, thumb: Path, visibility: str) -> dict:
        return {"videoId": "FAKE12345", "url": "https://youtu.be/FAKE12345",
                "status": visibility}

    def add_to_playlist(self, video_id: str, playlist_title: str) -> str:
        self.playlist_calls.append((video_id, playlist_title))
        return "FAKEPL123"

    def update_description(self, video_id: str, description: str) -> None:
        self.description_updates.append((video_id, description))


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

    def add_to_playlist(self, video_id: str, playlist_title: str) -> str:
        """把视频加入标题匹配的播放列表；不存在则创建（播放列表本身公开，
        列表里的视频是否可见仍取决于视频自身的 privacyStatus）。返回 playlistId。"""
        playlist_id = self._find_playlist(playlist_title)
        if playlist_id is None:
            resp = self._yt.playlists().insert(
                part="snippet,status",
                body={"snippet": {"title": playlist_title},
                      "status": {"privacyStatus": "public"}},
            ).execute()
            playlist_id = resp["id"]
        self._yt.playlistItems().insert(
            part="snippet",
            body={"snippet": {"playlistId": playlist_id,
                               "resourceId": {"kind": "youtube#video", "videoId": video_id}}},
        ).execute()
        return playlist_id

    def update_description(self, video_id: str, description: str) -> None:
        """改视频简介(互链回填用):update 要求 snippet 带 title+categoryId,
        先 list 取现有 snippet 再改 description,避免把标题/分类冲掉。"""
        resp = self._yt.videos().list(part="snippet", id=video_id).execute()
        items = resp.get("items", [])
        if not items:
            return
        snip = items[0]["snippet"]
        snip["description"] = description
        self._yt.videos().update(
            part="snippet",
            body={"id": video_id, "snippet": {
                "title": snip.get("title", ""), "description": description,
                "categoryId": snip.get("categoryId", "22"),
                "tags": snip.get("tags", [])}},
        ).execute()

    def _find_playlist(self, title: str) -> str | None:
        """在自己的播放列表里按标题精确匹配（翻页找全，别漏了 50 个之后的）。"""
        page_token = None
        while True:
            kw = {"part": "snippet", "mine": True, "maxResults": 50}
            if page_token:
                kw["pageToken"] = page_token
            resp = self._yt.playlists().list(**kw).execute()
            for item in resp.get("items", []):
                if item.get("snippet", {}).get("title") == title:
                    return item["id"]
            page_token = resp.get("nextPageToken")
            if not page_token:
                return None


def get_uploader(config: Config) -> Uploader:
    provider = config.get("youtube.provider", "youtube")
    if provider == "fake":
        return FakeUploader()
    if provider == "youtube":
        creds = load_creds(config.get("youtube.token", "secrets/token.json"),
                           config.get("youtube.client_secret", "secrets/client_secret.json"))
        return YouTubeUploader(creds)
    raise ValueError(f"未知 youtube provider: {provider}")
