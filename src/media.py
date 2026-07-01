from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.config import Config


class MediaSource(Protocol):
    def search(self, query: str, limit: int = 5) -> list[dict]: ...
    def fetch(self, candidate: dict, dest: Path) -> None: ...


class FakeMedia:
    """离线：确定性返回一个 PD 候选，fetch 写一张合法的最小 JPEG。

    注意：必须写真正可解码的图片（而非占位字节），否则下游 s7 的 ffmpeg
    `-loop 1` 会对非法数据反复重试解码卡死、s8 的 PIL make_thumbnail 会崩，
    使 media=fake 无法离线跑通整条流水线。
    """

    def search(self, query: str, limit: int = 5) -> list[dict]:
        return [{
            "manifest": f"fake://{query}", "ext": "jpg",
            "license": "public-domain", "attribution": f"Fake/{query}", "source": "fake",
        }]

    def fetch(self, candidate: dict, dest: Path) -> None:
        from PIL import Image

        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (18, 20, 40)).save(dest, "JPEG")


class NASAMedia:
    """NASA Images API（公有领域）。惰性 import requests。"""

    SEARCH = "https://images-api.nasa.gov/search"

    def search(self, query: str, limit: int = 5) -> list[dict]:
        import requests

        r = requests.get(self.SEARCH, params={"q": query, "media_type": "image"}, timeout=20)
        r.raise_for_status()
        items = r.json().get("collection", {}).get("items", [])[:limit]
        out = []
        for it in items:
            data = (it.get("data") or [{}])[0]
            out.append({
                "manifest": it.get("href", ""), "ext": "jpg",
                "license": "public-domain",
                "attribution": f"NASA / {data.get('title', 'image')}",
                "source": "nasa",
            })
        return [c for c in out if c["manifest"]]

    def fetch(self, candidate: dict, dest: Path) -> None:
        import requests

        man = requests.get(candidate["manifest"], timeout=20)
        man.raise_for_status()
        urls = [u for u in man.json() if isinstance(u, str) and u.lower().endswith(".jpg")]
        url = next((u for u in urls if "~orig" in u), urls[0])
        img = requests.get(url, timeout=30)
        img.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(img.content)


def get_media_source(config: Config) -> MediaSource:
    provider = config.get("media.provider", "nasa")
    if provider == "fake":
        return FakeMedia()
    if provider == "nasa":
        return NASAMedia()
    raise ValueError(f"未知 media provider: {provider}")
