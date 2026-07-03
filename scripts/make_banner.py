"""频道 banner 合成:SDXL 太空底图 + 高清金山(从 800px 头像亮度键控抠出) + 金色金属字。

用法:
    python scripts/make_banner.py --brand zh --bg banner_bg.png --out banner_zh.png \
        [--font /path/NotoSansCJK-Black.ttc]

尺寸按 YouTube 官方:2048x1152,<=6MB;所有品牌元素锁在中央安全区 1546x423
(手机/TV 都不裁)。山标不用 308px 的小水印放大(会糊),而是从 800px 的
assets/avatar_zh.png 亮度键控抠出后**下采样**;文字同理现渲不缩放。
"""
from __future__ import annotations

import argparse

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 2048, 1152
SAFE_W, SAFE_H = 1546, 423
SAFE_X, SAFE_Y = (W - SAFE_W) // 2, (H - SAFE_H) // 2

# 金色渐变(采自台标金),上亮下深
GOLD_STOPS = [(0.0, (246, 227, 170)), (0.45, (227, 181, 84)), (1.0, (140, 100, 28))]

DEFAULT_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc"


def _lerp(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def _gold_at(t: float):
    for (t1, c1), (t2, c2) in zip(GOLD_STOPS, GOLD_STOPS[1:]):
        if t <= t2:
            return _lerp(c1, c2, (t - t1) / (t2 - t1) if t2 > t1 else 0)
    return GOLD_STOPS[-1][1]


def extract_mountain(avatar_path: str) -> Image.Image:
    """从 800px 头像抠金山:金亮/蓝暗对比,亮度软阈值成 alpha,只取山体区域。"""
    img = Image.open(avatar_path).convert("RGB")
    crop = img.crop((130, 170, 680, 540))          # 山体 bbox(避开下方文字)
    px = crop.load()
    mask = Image.new("L", crop.size, 0)
    mpx = mask.load()
    for y in range(crop.size[1]):
        for x in range(crop.size[0]):
            r, g, b = px[x, y]
            lum = 0.5 * r + 0.4 * g + 0.1 * b       # 压蓝抬金,山体边缘更干净
            mpx[x, y] = max(0, min(255, int((lum - 55) * 6)))
    mask = mask.filter(ImageFilter.GaussianBlur(1.2))
    out = crop.convert("RGBA")
    out.putalpha(mask)
    return out.crop(out.getbbox())


def gold_text_layer(lines: list[tuple[str, int, int]], font_path: str) -> Image.Image:
    """金色金属字图层:每行 (文本, 字号, 字间距px)。带描边+软阴影,行间居中排版。"""
    fonts = [ImageFont.truetype(font_path, size) for _, size, _ in lines]

    def line_size(text, font, tracking):
        ws = [font.getbbox(ch)[2] for ch in text]
        return sum(ws) + tracking * (len(text) - 1), font.size

    sizes = [line_size(t, f, tr) for (t, _, tr), f in zip(lines, fonts)]
    gap = 18
    total_h = sum(int(f.size * 1.25) for f in fonts) + gap * (len(lines) - 1)
    lw = max(w for w, _ in sizes)
    layer = Image.new("RGBA", (lw + 80, total_h + 80), (0, 0, 0, 0))

    y = 40
    for (text, _, tracking), font, (w, _) in zip(lines, fonts, sizes):
        lh = int(font.size * 1.25)
        # 单行画到独立画布:字形(白) -> 阴影/描边/渐变填充
        line_img = Image.new("L", (lw + 80, lh + 40), 0)
        d = ImageDraw.Draw(line_img)
        x = (lw + 80 - w) // 2
        for ch in text:
            d.text((x, 8), ch, font=font, fill=255)
            x += font.getbbox(ch)[2] + tracking
        # 软阴影
        sh = line_img.filter(ImageFilter.GaussianBlur(6))
        shadow = Image.new("RGBA", line_img.size, (0, 0, 0, 0))
        shadow.putalpha(sh.point(lambda v: int(v * 0.8)))
        layer.alpha_composite(shadow, (0, y - 8 + 5))
        # 深色描边(膨胀)
        edge = line_img.filter(ImageFilter.MaxFilter(5))
        outline = Image.new("RGBA", line_img.size, (40, 26, 6, 255))
        outline.putalpha(edge)
        layer.alpha_composite(outline, (0, y - 8))
        # 垂直金渐变填充
        grad = Image.new("RGBA", line_img.size)
        gd = ImageDraw.Draw(grad)
        for gy in range(lh + 40):
            gd.line([(0, gy), (lw + 80, gy)], fill=_gold_at(gy / (lh + 40)) + (255,))
        grad.putalpha(line_img)
        layer.alpha_composite(grad, (0, y - 8))
        y += lh + gap
    return layer.crop(layer.getbbox())


def compose(bg_path: str, brand: str, out_path: str, font_path: str) -> None:
    bg = Image.open(bg_path).convert("RGB").resize((W, H)).convert("RGBA")

    # 中央径向暗角:压暗徽记后方,保文字对比
    vig = Image.new("L", (W, H), 0)
    vd = ImageDraw.Draw(vig)
    cx, cy, rad = W // 2, H // 2 + 30, 620
    for r in range(rad, 0, -6):
        vd.ellipse([cx - r, cy - r * 0.62, cx + r, cy + r * 0.62],
                   fill=int(110 * (1 - r / rad)))
    vig = vig.filter(ImageFilter.GaussianBlur(40))
    dark = Image.new("RGBA", (W, H), (4, 5, 14, 0))
    dark.putalpha(vig)
    bg.alpha_composite(dark)

    mountain = extract_mountain("assets/avatar_zh.png")

    if brand == "zh":
        m_h, text = 195, [("尖锋纪录片", 128, 14)]
    else:
        m_h, text = 185, [("SUMMIT", 118, 26), ("DOCUMENTARIES", 46, 22)]
    m_w = int(mountain.width * m_h / mountain.height)
    mountain = mountain.resize((m_w, m_h), Image.LANCZOS)
    txt = gold_text_layer(text, font_path)

    total = m_h + 14 + txt.height
    top = SAFE_Y + (SAFE_H - total) // 2
    bg.alpha_composite(mountain, ((W - m_w) // 2, top))
    bg.alpha_composite(txt, ((W - txt.width) // 2, top + m_h + 14))

    bg.convert("RGB").save(out_path, "PNG", optimize=True)
    print(f"banner -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", choices=["zh", "en"], required=True)
    ap.add_argument("--bg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--font", default=DEFAULT_FONT)
    args = ap.parse_args()
    compose(args.bg, args.brand, args.out, args.font)


if __name__ == "__main__":
    main()
