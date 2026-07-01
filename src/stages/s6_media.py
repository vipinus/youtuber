from __future__ import annotations

from pathlib import Path

from src.context import Ctx
from src.media import get_media_source
from src.util.jobs import read_json, write_json

OUTPUT = "credits.json"
INPUT = "script.json"
DEFAULT_ALLOWED = ["public-domain", "cc0", "cc-by", "cc-by-sa"]


def _terms(query: str) -> list[str]:
    """把 visual_query 拆成多个候选检索词——LLM 常给逗号拼接的复合串，
    整串搜图库 0 命中，拆成单个短词逐个试命中率高得多。"""
    parts = [t.strip() for t in query.split(",") if t.strip()]
    return parts or ([query.strip()] if query.strip() else [])


def _fallback(ctx: Ctx, media_dir: Path, sid: int, reason: str) -> dict:
    name = f"seg_{sid:03d}.jpg"                       # 兜底恒为静图
    ctx.warn(f"{ctx.slug} seg {sid}: {reason}，用兜底 PD 图")
    fb = Path(ctx.config.get("paths.fallback_image", "assets/fallback.jpg"))
    (media_dir / name).write_bytes(fb.read_bytes() if fb.exists() else b"")
    return {"id": sid, "file": name, "source": "fallback",
            "license": "public-domain", "attribution": "fallback"}


def run(ctx: Ctx) -> None:
    script = read_json(ctx.job, INPUT)
    source = get_media_source(ctx.config)
    allowed = set(ctx.config.get("guardrails.allowed_licenses", DEFAULT_ALLOWED))
    media_dir = ctx.job / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    for seg in script.get("segments", []):
        sid = int(seg.get("id", 0))
        try:
            c = None
            for term in _terms(seg.get("visual_query", "")):
                cands = [x for x in source.search(term)
                         if x.get("license") in allowed]          # 护栏#2: license 门
                if cands:
                    c = cands[sid % len(cands)]                   # 同词不同段轮选，避免整片同图
                    break
            if c is None:
                items.append(_fallback(ctx, media_dir, sid, "无合法素材"))
                continue
            ext = str(c.get("ext") or "jpg").lstrip(".").lower()  # 视频源 mp4 / 图源 jpg
            name = f"seg_{sid:03d}.{ext}"
            source.fetch(c, media_dir / name)
            items.append({"id": sid, "file": name, "source": c["source"],
                          "license": c["license"], "attribution": c["attribution"]})
        except Exception as e:  # 网络/畸形素材 → fail-safe 兜底
            items.append(_fallback(ctx, media_dir, sid, f"取素材失败({type(e).__name__})"))

    write_json(ctx.job, OUTPUT, {"items": items})
