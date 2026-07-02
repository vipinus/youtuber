from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.context import Ctx
from src.util.jobs import read_json, write_json
from src.util.llmjson import extract_json

OUTPUT = "meta.json"
INPUT = "script.json"
TITLE_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc"

SYSTEM = (
    "你是「探索飞碟与外星人」频道的 YouTube 运营。为给定脚本写 SEO 元数据。"
    "只输出一个 JSON 对象，不要额外文字。"
)


def build_prompt(script: dict) -> str:
    hint = (script.get("title_options") or ["未解之谜"])[0]
    body = "".join(s.get("text", "") for s in script.get("segments", []))[:800]
    return (
        "为下面的悬疑解说写 YouTube 元数据，输出 JSON，字段："
        "title(中文,吸引点击,≤40字), description(中文,2-4句), tags(中文/英文字符串数组,5-12个)。\n"
        f"标题线索：{hint}\n脚本节选：{body}"
    )


def _font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:                                       # 字体缺失(换机器)→兜底默认字体，别崩最后一阶段
        try:
            return ImageFont.load_default(size)
        except TypeError:                                # 老版 Pillow 的 load_default 不吃 size
            return ImageFont.load_default()


def _fit(path: str, text: str, max_w: int, start: int = 90, lo: int = 40):
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    s = start
    while s > lo:
        f = _font(path, s)
        if probe.textbbox((0, 0), text, font=f, stroke_width=3)[2] <= max_w:
            return f
        s -= 3
    return _font(path, lo)


def make_thumbnail(dest: Path, title: str, bg: Path, font_path: str | None = None) -> None:
    """font_path：P7 T14 可配置字体(config `paths.title_font`)；不传时用
    模块级 TITLE_FONT(笔记本硬编码路径)兜底，保持老调用方/老测试不变。"""
    W, H = 1280, 720
    img = None
    if bg.exists() and bg.stat().st_size > 0:
        try:
            img = Image.open(bg).convert("RGB").resize((W, H))
        except Exception:                                    # 素材非法/损坏(非图字节)→回退纯色底，别崩在最后一阶段
            img = None
    if img is None:
        img = Image.new("RGB", (W, H), (10, 12, 28))
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 110))          # 压暗保证字清晰
    img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    d = ImageDraw.Draw(img)
    line = title if len(title) <= 9 else title[:9]
    font = _fit(font_path or TITLE_FONT, line, int(W * 0.86))
    d.text((W // 2, H // 2), line, font=font, fill=(255, 255, 255),
           anchor="mm", stroke_width=6, stroke_fill=(8, 12, 30))
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, "JPEG", quality=88, optimize=True)


def run(ctx: Ctx) -> None:
    script = read_json(ctx.job, INPUT)
    meta = extract_json(ctx.llm.generate(SYSTEM, build_prompt(script)))
    try:
        credits = read_json(ctx.job, "credits.json").get("items", [])
    except FileNotFoundError:
        credits = []
    attrib = sorted({c["attribution"] for c in credits if c.get("source") != "fallback"})
    if attrib:
        meta["description"] = meta.get("description", "") + "\n\n素材来源：" + "；".join(attrib)
    title = meta.get("title") or (script.get("title_options") or ["未解之谜"])[0]
    font_path = ctx.config.get("paths.title_font", TITLE_FONT)
    make_thumbnail(ctx.job / "thumb.jpg", title, ctx.job / "media" / "seg_001.jpg",
                    font_path=font_path)
    write_json(ctx.job, OUTPUT, meta)
