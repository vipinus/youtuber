"""生成兜底 PD 图（无匹配素材时用）：1920x1080 深空星野，纯 Pillow、自制、公有领域。"""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

W, H = 1920, 1080
OUT = Path(__file__).parent / "fallback.jpg"
random.seed(11)


def vgrad(top, bot) -> Image.Image:
    col = Image.new("RGB", (1, H))
    for y in range(H):
        t = y / (H - 1)
        col.putpixel((0, y), tuple(int(a + (b - a) * t) for a, b in zip(top, bot)))
    return col.resize((W, H)).convert("RGBA")


def blob(cx, cy, r, color) -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(layer).ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return layer.filter(ImageFilter.GaussianBlur(r // 2))


def main() -> None:
    img = vgrad((4, 6, 18), (16, 12, 36))
    img.alpha_composite(blob(int(W * 0.25), int(H * 0.30), 400, (40, 120, 150, 80)))
    img.alpha_composite(blob(int(W * 0.80), int(H * 0.75), 460, (90, 44, 120, 85)))
    d = ImageDraw.Draw(img)
    for _ in range(600):
        x = random.randint(0, W)
        y = int(random.triangular(0, H, 0))
        b = random.randint(120, 255)
        s = random.choice([1, 1, 1, 2])
        d.ellipse([x, y, x + s, y + s], fill=(b, b, min(255, b + 20), 255))
    img.convert("RGB").save(OUT, "JPEG", quality=88, optimize=True, progressive=True)
    w, h = Image.open(OUT).size
    print(f"saved {OUT}  {w}x{h}  {OUT.stat().st_size // 1024}KB")
    assert (w, h) == (W, H)


if __name__ == "__main__":
    main()
