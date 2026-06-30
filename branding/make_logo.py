"""生成频道头像 + 品牌水印（与横幅同款飞碟徽标，保持品牌一致）。
- avatar.png   : 800x800 PNG，深色圆形徽章（YouTube 头像，建议 >=150x150，<=1MB）
- watermark.png: 320x320 PNG，透明背景的飞碟徽标 + 暗色光晕（视频品牌水印，<=1MB）
纯 Pillow，无需 GPU/AI。
"""
from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

HERE = Path(__file__).parent
random.seed(7)


def soft(size, cx, cy, r, color) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return layer.filter(ImageFilter.GaussianBlur(max(1, r // 2)))


def radial_badge(S, inner, outer) -> Image.Image:
    s0 = 256
    g = Image.new("RGB", (s0, s0))
    px = g.load()
    c = s0 / 2
    for y in range(s0):
        for x in range(s0):
            d = min(1.0, math.hypot(x - c, y - c) / c)
            px[x, y] = tuple(int(inner[i] + (outer[i] - inner[i]) * d) for i in range(3))
    return g.resize((S, S)).convert("RGBA")


def circle_mask(S) -> Image.Image:
    m = Image.new("L", (S, S), 0)
    ImageDraw.Draw(m).ellipse([0, 0, S - 1, S - 1], fill=255)
    return m


def draw_saucer(img, cx, cy, w, under_glow=True) -> None:
    h = int(w * 0.30)
    if under_glow:
        img.alpha_composite(soft(img.size, cx, int(cy + h * 0.85), int(w * 0.40), (70, 255, 170, 120)))
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    lw = max(2, w // 110)
    dw, dh = w * 0.46, h * 2.0
    d.ellipse([cx - dw / 2, cy - dh * 0.70, cx + dw / 2, cy + dh * 0.30],
              fill=(130, 225, 255, 175), outline=(215, 246, 255, 235), width=lw)
    d.ellipse([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], fill=(128, 140, 166, 255))
    d.ellipse([cx - w / 2, cy - h / 2, cx + w / 2, cy + h * 0.12], fill=(184, 196, 216, 255))
    d.ellipse([cx - w / 2, cy + h * 0.06, cx + w / 2, cy + h / 2], fill=(80, 90, 114, 255))
    cols = [(255, 110, 110), (120, 255, 180), (130, 200, 255)]
    n = 7
    for i in range(n):
        lx = cx - w * 0.40 + (i / (n - 1)) * w * 0.80
        ly = cy + h * 0.14
        r = max(2, w // 80)
        d.ellipse([lx - r, ly - r, lx + r, ly + r], fill=cols[i % 3] + (255,))
    img.alpha_composite(layer)


def make_avatar(S=800) -> Path:
    img = radial_badge(S, inner=(30, 34, 70), outer=(8, 8, 20))
    d = ImageDraw.Draw(img)
    for _ in range(40):  # faint stars inside the badge
        a = random.uniform(0, 2 * math.pi)
        rr = random.uniform(0.18, 0.46) * S
        x, y = S / 2 + rr * math.cos(a), S / 2 + rr * math.sin(a)
        b = random.randint(140, 235)
        d.ellipse([x, y, x + 2, y + 2], fill=(b, b, 255, 255))
    draw_saucer(img, S // 2, int(S * 0.54), int(S * 0.60))
    d.ellipse([S * 0.035, S * 0.035, S * 0.965, S * 0.965],
              outline=(95, 205, 255, 200), width=max(3, S // 90))
    img.putalpha(ImageChops.multiply(img.split()[3], circle_mask(S)))  # round, transparent corners
    out = HERE / "avatar.png"
    img.save(out, "PNG", optimize=True)
    return out


def make_watermark(S=320) -> Path:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    # dark halo so the bright emblem stays visible over light video frames
    img.alpha_composite(soft((S, S), S // 2, int(S * 0.5), int(S * 0.40), (0, 0, 0, 150)))
    draw_saucer(img, S // 2, int(S * 0.52), int(S * 0.74))
    out = HERE / "watermark.png"
    img.save(out, "PNG", optimize=True)
    return out


def report(p: Path) -> None:
    w, h = Image.open(p).size
    mb = p.stat().st_size / 1_048_576
    print(f"{p.name:14s} {w}x{h}  {mb:.3f} MB")
    assert mb <= 1.0, f"{p.name} over 1MB"


if __name__ == "__main__":
    report(make_avatar())
    report(make_watermark())
