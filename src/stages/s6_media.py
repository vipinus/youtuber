from __future__ import annotations

from pathlib import Path

from src.context import Ctx
from src.media import get_media_source
from src.util.jobs import read_json, write_json

OUTPUT = "credits.json"
INPUT = "script.json"
DEFAULT_ALLOWED = ["public-domain", "cc0", "cc-by", "cc-by-sa"]


def _fallback(ctx: Ctx, media_dir: Path, name: str, sid: int, reason: str) -> dict:
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
        name = f"seg_{sid:03d}.jpg"
        try:
            cands = [c for c in source.search(seg.get("visual_query", ""))
                     if c.get("license") in allowed]              # 护栏#2: license 门
            if not cands:
                items.append(_fallback(ctx, media_dir, name, sid, "无合法素材"))
                continue
            c = cands[0]
            source.fetch(c, media_dir / name)
            items.append({"id": sid, "file": name, "source": c["source"],
                          "license": c["license"], "attribution": c["attribution"]})
        except Exception as e:  # 网络/畸形素材 → fail-safe 兜底
            items.append(_fallback(ctx, media_dir, name, sid, f"取素材失败({type(e).__name__})"))

    write_json(ctx.job, OUTPUT, {"items": items})
