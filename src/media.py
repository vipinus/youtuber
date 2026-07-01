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


_ARCHIVE_PD_COLLECTIONS = {
    "prelinger", "fedflix", "usgovfilms", "us-national-archives",
    "nasa", "publicmovies", "openbeelden",
}


def _norm_license(licenseurl: str | None, collection: str | list[str] | None) -> str | None:
    """archive.org 素材护栏#2：把 licenseurl/collection 归一成 s6 白名单 token
    ({public-domain, cc0, cc-by, cc-by-sa} 之一)，无法确认合法 -> None（拒绝，
    根本不作为候选，不是兜底）。

    顺序很重要：显式 by-nc/by-nd 一律硬拒——即便 collection 命中 PD 白名单也不能
    覆盖（NC 不在允许集合里；ND 禁止我们剪辑改作）。
    """
    lic = (licenseurl or "").strip().lower()
    if "by-nc" in lic or "by-nd" in lic:
        return None
    if "publicdomain" in lic or "/zero/" in lic:
        return "public-domain"
    if "/licenses/by-sa" in lic:
        return "cc-by-sa"
    if "/licenses/by/" in lic:
        return "cc-by"
    if lic:
        # licenseurl 存在但不是已知许可(如 rightsstatements.org/InC = 明确"有版权")
        # → 一律拒。绝不让 collection 白名单覆盖显式许可声明——这是封号级红线。
        return None

    # 仅当完全没有 licenseurl 时，才凭 collection 判定(公认全 PD 的集合)
    cols = collection if isinstance(collection, list) else ([collection] if collection else [])
    cols_norm = {str(c).strip().lower() for c in cols if c}
    pd_norm = {c.lower() for c in _ARCHIVE_PD_COLLECTIONS}
    if cols_norm & pd_norm:
        return "public-domain"
    return None


class ArchiveVideo:
    """archive.org 公有领域/宽松CC影片片段（mediatype:movies）。惰性 import requests。

    archive.org 的影片搜索一半是坑：mirrortube/altcensored/PeriscopeFilm 等集合是镜像的
    YouTube 视频、通常无 licenseurl，很可能有版权 —— 必须靠 `_norm_license` 严判后拒绝，
    这是护栏#2，保护频道不被 Content-ID 打击，不能削弱。
    """

    SEARCH = "https://archive.org/advancedsearch.php"
    METADATA = "https://archive.org/metadata"

    _VIDEO_FORMATS = {"h.264", "MPEG4", "512Kb MPEG4", "h.264 HD", "Ogg Video"}
    _VIDEO_EXTS = (".mp4", ".ogv", ".m4v")
    # live 探针实测坑：有些条目带 "Download.mp4"/"Download (1).mp4" 这类几KB的占位/损坏
    # 衍生文件(实测 0.04s 时长)。"选最小 mp4" 若不设下限会挑中它们而不是真片。
    _MIN_PLAUSIBLE_VIDEO_BYTES = 200_000

    def search(self, query: str, limit: int = 5) -> list[dict]:
        import requests

        rows = max(limit * 4, limit)          # 过量取，扛得住过滤掉的坏结果
        params = {
            "q": f"({query}) AND mediatype:movies",
            "fl[]": ["identifier", "title", "licenseurl", "collection"],
            "rows": rows,
            "output": "json",
        }
        r = requests.get(self.SEARCH, params=params, timeout=20)
        r.raise_for_status()
        docs = (r.json().get("response") or {}).get("docs", [])

        out: list[dict] = []
        for d in docs:
            identifier = d.get("identifier")
            if not identifier:
                continue
            token = _norm_license(d.get("licenseurl"), d.get("collection"))
            if token is None:                  # 护栏#2：没过白名单 -> 根本不作为候选
                continue
            out.append({
                "manifest": identifier, "ext": "mp4", "license": token,
                "attribution": f"Internet Archive / {d.get('title', identifier)}",
                "source": "archive",
            })
            if len(out) >= limit:
                break
        return out

    def _pick_file(self, files: list[dict]) -> dict:
        def is_video(f: dict) -> bool:
            name = str(f.get("name", ""))
            return f.get("format") in self._VIDEO_FORMATS or name.lower().endswith(self._VIDEO_EXTS)

        def size_of(f: dict) -> int:
            try:
                return int(f.get("size") or 0)
            except (TypeError, ValueError):
                return 0

        videos = [f for f in files if is_video(f)]
        if not videos:
            raise ValueError("archive.org: 无可用视频文件")

        # 只截几秒就用，优先选最小的 mp4（避免为几秒画面下 59MB 全片）；
        # 没有 mp4 才退到 ogv/m4v。
        mp4s = [f for f in videos if str(f.get("name", "")).lower().endswith(".mp4")]
        pick_from = mp4s or videos
        # 过滤掉可疑的极小占位/损坏文件；若全体都可疑（唯一选项），退回全体而非报错。
        plausible = [f for f in pick_from if size_of(f) >= self._MIN_PLAUSIBLE_VIDEO_BYTES]
        pick_from = plausible or pick_from
        return min(pick_from, key=size_of)

    def fetch(self, candidate: dict, dest: Path) -> None:
        import requests

        identifier = candidate["manifest"]
        r = requests.get(f"{self.METADATA}/{identifier}", timeout=20)
        r.raise_for_status()
        meta = r.json()

        # 纵深防御：metadata 是权威来源，即便 search 阶段放行过，这里也要重新核验，
        # 免得索引过期/candidate 被篡改，绕过护栏#2。
        info = meta.get("metadata") or {}
        token = _norm_license(info.get("licenseurl"), info.get("collection"))
        if token is None:
            raise ValueError(
                f"archive.org {identifier}: metadata 未通过护栏#2 二次核验"
                f"(licenseurl={info.get('licenseurl')!r}, collection={info.get('collection')!r})"
            )

        best = self._pick_file(meta.get("files") or [])
        url = f"https://archive.org/download/{identifier}/{best['name']}"
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    fh.write(chunk)


def get_media_source(config: Config) -> MediaSource:
    provider = config.get("media.provider", "nasa")
    if provider == "fake":
        return FakeMedia()
    if provider == "nasa":
        return NASAMedia()
    if provider == "archive":
        return ArchiveVideo()
    raise ValueError(f"未知 media provider: {provider}")
