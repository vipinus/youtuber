"""中文频道改名重做台标(2026-07-05 用户定:尖锋纪录片→发现纪录片)。

保留原金山图形(watermark_tr.png 顶部 y0-178 已是干净透明金山),只替换下方
金字。复用 make_banner.gold_text_layer 的金属渐变字。产出:
  - assets/watermark_tr.png  透明台标(正片烧角标 + 缩略图水印共用)
  - assets/avatar_zh.png     800x800 深色底头像
旧文件先备份 .bak。用法:python scripts/rebrand_zh.py "发现纪录片"
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_banner import gold_text_layer  # noqa: E402  金属金渐变字

ASSETS = Path(__file__).resolve().parents[1] / "assets"
NAME = sys.argv[1] if len(sys.argv) > 1 else "发现纪录片"


def build_watermark(name: str) -> Image.Image:
    """透明台标:原金山(顶部)+ 新金字(底部)。加左右/顶部内边距,避免文字顶框
    (用户反馈 2026-07-05:图标偏大、发/片贴边)。山峰适度缩小留呼吸。"""
    src = ASSETS / "watermark_tr.png.bak"             # 从旧图取干净金山(重跑幂等)
    if not src.exists():
        src = ASSETS / "watermark_tr.png"
    old = Image.open(src).convert("RGBA")
    OW, _ = old.size
    mtn_src = old.crop((0, 0, OW, 180)).crop(         # 裁到山峰实际 bbox 再统一缩放
        Image.open(src).convert("RGBA").crop((0, 0, OW, 180)).getbbox())

    MARGIN = 26                                       # 四周留白,元素不再贴边
    inner_w = 300                                     # 内容区宽度
    W = inner_w + MARGIN * 2
    mtn_w = int(inner_w * 0.72)                       # 山峰只占内容宽的 72%(原来近满宽)
    mtn = mtn_src.resize((mtn_w, int(mtn_src.height * mtn_w / mtn_src.width)), Image.LANCZOS)

    txt = gold_text_layer([(name, 74, 5)], str(_font()))
    tw = int(inner_w * 0.94)                          # 文字占内容宽 94%,两侧仍有留白
    txt = txt.resize((tw, int(txt.height * tw / txt.width)), Image.LANCZOS)

    gap = 8
    H = MARGIN + mtn.height + gap + txt.height + MARGIN
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.alpha_composite(mtn, ((W - mtn.width) // 2, MARGIN))
    out.alpha_composite(txt, ((W - txt.width) // 2, MARGIN + mtn.height + gap))
    return out


def build_avatar(name: str) -> Image.Image:
    """800x800 头像:深蓝径向渐变底 + 金山 + 金字(沿用旧头像视觉)。"""
    S = 800
    img = Image.new("RGBA", (S, S), (10, 14, 34, 255))
    vig = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(vig)
    for r in range(S // 2, 0, -6):
        d.ellipse([S//2 - r, S//2 - r, S//2 + r, S//2 + r],
                  fill=int(70 * (1 - r / (S // 2))))
    glow = Image.new("RGBA", (S, S), (40, 60, 120, 0))
    glow.putalpha(vig.filter(ImageFilter.GaussianBlur(60)))
    img = Image.alpha_composite(img, glow)

    wm = Image.open(ASSETS / "watermark_tr.png").convert("RGBA")
    mtn = wm.crop((0, 0, wm.width, 180))
    mh = 360
    mw = int(mtn.width * mh / mtn.height)
    mtn = mtn.resize((mw, mh), Image.LANCZOS)
    img.alpha_composite(mtn, ((S - mw) // 2, 150))

    txt = gold_text_layer([(name, 150, 12)], str(_font()))
    tw = min(txt.width, int(S * 0.82))
    txt = txt.resize((tw, int(txt.height * tw / txt.width)), Image.LANCZOS)
    img.alpha_composite(txt, ((S - txt.width) // 2, 560))
    return img


def _font() -> Path:
    for c in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
              "/usr/share/fonts/truetype/noto/NotoSansCJK-Black.ttc"):
        if Path(c).exists():
            return Path(c)
    return Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc")


def main() -> None:
    # 先构建(读旧金山),再备份旧文件,最后落新——顺序颠倒会把源文件删掉
    new_wm = build_watermark(NAME)
    new_av = build_avatar(NAME)
    for f in ("watermark_tr.png", "avatar_zh.png"):
        p = ASSETS / f
        if p.exists():
            p.replace(p.with_suffix(".png.bak"))
    new_wm.save(ASSETS / "watermark_tr.png")
    new_av.save(ASSETS / "avatar_zh.png")
    print(f"台标/头像已重做为「{NAME}」")


if __name__ == "__main__":
    main()
