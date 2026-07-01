"""生成 YouTube 频道横幅（channel art）。
规格：2048x1152 px，JPG，<6MB；标题落在 YouTube「全设备安全区」(中央 1235x338)。
纯 Pillow 程序化绘制（无需 GPU / AI）。确定性（固定随机种子）。
"""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 2048, 1152
OUT = Path(__file__).parent / "channel_banner.jpg"
TITLE_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc"
TAG_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

TITLE = "探索飞碟与外星人"
TAGLINE = "真实 UFO 档案  ·  未解之谜探秘"

random.seed(7)


def vgradient(top, bottom) -> Image.Image:
    col = Image.new("RGB", (1, H))
    for y in range(H):
        t = y / (H - 1)
        col.putpixel((0, y), tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)))
    return col.resize((W, H)).convert("RGBA")


def soft_blob(cx, cy, r, color) -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(layer).ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return layer.filter(ImageFilter.GaussianBlur(r // 2))


def add_stars(img: Image.Image) -> None:
    d = ImageDraw.Draw(img)
    for _ in range(520):
        x = random.randint(0, W)
        y = int(random.triangular(0, H, 0))  # denser near top
        b = random.randint(120, 255)
        s = random.choice([1, 1, 1, 2])
        d.ellipse([x, y, x + s, y + s], fill=(b, b, min(255, b + 20), 255))
    # bright glow stars
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for _ in range(28):
        x, y = random.randint(40, W - 40), random.randint(20, int(H * 0.55))
        r = random.randint(3, 6)
        gd.ellipse([x - r, y - r, x + r, y + r], fill=(180, 220, 255, 230))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(6)))


def draw_saucer(img: Image.Image, cx, cy, w) -> None:
    h = int(w * 0.28)
    # under-glow
    img.alpha_composite(
        soft_blob(cx, int(cy + h * 0.9), int(w * 0.42), (70, 255, 170, 110))
    )
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    # dome
    dw, dh = w * 0.46, h * 2.1
    d.ellipse([cx - dw / 2, cy - dh * 0.72, cx + dw / 2, cy + dh * 0.28],
              fill=(130, 225, 255, 160), outline=(210, 245, 255, 220), width=3)
    # body
    d.ellipse([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], fill=(120, 132, 158, 255))
    d.ellipse([cx - w / 2, cy - h / 2, cx + w / 2, cy + h * 0.12], fill=(176, 188, 208, 255))
    d.ellipse([cx - w / 2, cy + h * 0.06, cx + w / 2, cy + h / 2], fill=(78, 88, 110, 255))
    d.ellipse([cx - w * 0.5, cy - h * 0.05, cx + w * 0.5, cy + h * 0.05],
              outline=(225, 235, 250, 200), width=2)
    # rim lights
    cols = [(255, 110, 110), (120, 255, 180), (130, 200, 255)]
    n = 9
    for i in range(n):
        lx = cx - w * 0.42 + (i / (n - 1)) * w * 0.84
        ly = cy + h * 0.16
        c = cols[i % 3]
        d.ellipse([lx - 6, ly - 6, lx + 6, ly + 6], fill=c + (255,))
    img.alpha_composite(layer)


def fit_font(path, text, max_w, start=190, lo=70) -> ImageFont.FreeTypeFont:
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    s = start
    while s > lo:
        f = ImageFont.truetype(path, s)
        bb = probe.textbbox((0, 0), text, font=f, stroke_width=4)
        if bb[2] - bb[0] <= max_w:
            return f
        s -= 4
    return ImageFont.truetype(path, lo)


def text_with_glow(img, xy, text, font, fill, glow, gr=16, sw=4, sf=(6, 10, 26)):
    gl = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(gl).text(xy, text, font=font, fill=glow + (255,),
                            anchor="mm", stroke_width=sw, stroke_fill=glow + (255,))
    img.alpha_composite(gl.filter(ImageFilter.GaussianBlur(gr)))
    ImageDraw.Draw(img).text(xy, text, font=font, fill=fill + (255,),
                             anchor="mm", stroke_width=sw, stroke_fill=sf + (255,))


def main() -> None:
    img = vgradient((4, 6, 18), (20, 14, 44))
    img.alpha_composite(soft_blob(int(W * 0.18), int(H * 0.22), 520, (40, 130, 150, 90)))
    img.alpha_composite(soft_blob(int(W * 0.84), int(H * 0.8), 560, (96, 44, 130, 95)))
    add_stars(img)

    # vignette (darken edges, keep center subject bright)
    vig = Image.new("L", (W, H), 0)
    ImageDraw.Draw(vig).ellipse(
        [int(W * 0.04), int(H * 0.02), int(W * 0.96), int(H * 0.98)], fill=255
    )
    vig = vig.filter(ImageFilter.GaussianBlur(170))
    img = Image.composite(img, Image.new("RGBA", (W, H), (0, 0, 0, 255)), vig)

    draw_saucer(img, W // 2, int(H * 0.27), 520)

    cx, cy = W // 2, int(H * 0.55)            # title centre, inside safe area
    title_font = fit_font(TITLE_FONT, TITLE, max_w=1180, start=200)
    text_with_glow(img, (cx, cy), TITLE, title_font, fill=(255, 255, 255),
                   glow=(90, 200, 255), gr=18, sw=5)
    tag_font = ImageFont.truetype(TAG_FONT, 58)
    text_with_glow(img, (cx, int(H * 0.70)), TAGLINE, tag_font, fill=(196, 224, 255),
                   glow=(40, 90, 150), gr=8, sw=2)

    out = img.convert("RGB")
    q = 92
    OUT.parent.mkdir(parents=True, exist_ok=True)
    while True:
        out.save(OUT, "JPEG", quality=q, optimize=True, progressive=True)
        mb = OUT.stat().st_size / 1_048_576
        if mb <= 6 or q <= 60:
            break
        q -= 6
    w, h = Image.open(OUT).size
    print(f"saved {OUT}  {w}x{h}  {mb:.2f} MB  q={q}")
    assert (w, h) == (W, H), "dimension mismatch"
    assert mb <= 6, "over 6MB"


if __name__ == "__main__":
    main()
