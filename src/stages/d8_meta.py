"""P7 T8：阶段⑧——AI 缩略图 + 吸引力简介。

流程：
  1. 用 LLM 读中文脚本(zh_script.json) + 源片信息(source.json)，一次性生成
     一个 JSON **对象**（避坑同 d3_translate：extract_json 只解析对象，绝不能
     让 LLM 直接回顶层数组）：
     {"title","description","tags","image_prompt"}。
  2. `image_prompt`（英文、无中文字要求）交给 src.remote.get_remote(config)
     去 zion 跑 SDXL，生成一张 1280x720 的无字背景图 thumb_bg.png。
  3. 本地 Pillow 合成：背景图 + 底部渐变(保标题可读) + 中文大标题(自动缩字号
     避免溢出，思路同 s8_meta._fit) + 频道水印(半透明深色圆角面板，右上角)
     → thumb.jpg（YouTube 缩略图标准 1280x720）。
  4. 追加源片署名到简介（如 "素材来源：NASA / Apollo 11（公有领域）"），写
     meta.json = {"title","description","tags"}——字段和 s9_publish 读的
     meta.json（INPUT）完全兼容，s9_publish 本身不用改。
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from src.context import Ctx
from src.remote import get_remote
from src.stages.s8_meta import TITLE_FONT, _fit
from src.util.jobs import exists, read_json, write_json
from src.util.llmjson import extract_json

OUTPUT = "meta.json"
INPUT = "zh_script.json"
SOURCE_INPUT = "source.json"
DEFAULT_WATERMARK = "assets/watermark_tr.png"
DEFAULT_IMAGE_PROMPT = (
    "cinematic dramatic space documentary scene, nebula and starfield, "
    "epic volumetric lighting, ultra detailed, 8k, no text"
)

_LICENSE_LABELS = {
    "public-domain": "公有领域",
    "cc0": "CC0",
    "cc-by": "CC BY",
    "cc-by-sa": "CC BY-SA",
}

SYSTEM = (
    "你是「发现纪录片频道」的 YouTube 运营。频道把公有领域英文纪录片翻译配音成"
    "中文重新发布。为给定的中文配音脚本写吸引点击的元数据。"
    "只输出一个 JSON 对象，不要额外文字、不要 markdown 代码块。"
)


def build_prompt(zh_script: dict, source: dict) -> str:
    body = "".join(seg.get("zh", "") for seg in zh_script.get("segments", []))[:1200]
    title_hint = source.get("title", "")
    return (
        "为下面这部翻译成中文配音的纪录片写 YouTube 元数据，输出一个 JSON 对象"
        '（不是数组），格式必须是 '
        '{"title": "...", "description": "...", "tags": ["...", ...], '
        '"image_prompt": "..."}。\n'
        "title: 中文标题，吸引点击，不超过约20字。\n"
        "description: 中文简介，2-4句，有吸引力，呼应脚本内容。\n"
        "tags: 5-12个中文/英文标签字符串数组。\n"
        "image_prompt: 一句英文的 Stable Diffusion 场景提示词，描述一个和本片"
        "主题相关、电影感强烈的太空/纪录片画面，要求 dramatic, no text, "
        "no watermark, no logo。\n"
        f"原片标题线索：{title_hint}\n"
        f"中文脚本节选：{body}"
    )


def _attribution_line(source: dict) -> str:
    attribution = (source or {}).get("attribution")
    if not attribution:
        return ""
    label = _LICENSE_LABELS.get(source.get("license"), source.get("license") or "")
    return f"素材来源：{attribution}（{label}）" if label else f"素材来源：{attribution}"


def _panel_with_watermark(watermark: Image.Image, target_w: int = 170, pad: int = 16) -> Image.Image:
    """水印缩到 target_w 宽，贴在一块半透明深色圆角面板上（白色 60-alpha 细描边）。"""
    w, h = watermark.size
    scale = target_w / w
    wm = watermark.resize((target_w, max(1, round(h * scale))), Image.LANCZOS)

    panel_w, panel_h = wm.width + pad * 2, wm.height + pad * 2
    panel = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle(
        [0, 0, panel_w - 1, panel_h - 1], radius=18,
        fill=(8, 12, 26, 150), outline=(255, 255, 255, 60), width=2,
    )
    panel.alpha_composite(wm, (pad, pad))
    return panel


def _wrap_title(title: str, max_per_line: int = 13) -> list[str]:
    """短标题单行；长标题折成两行(不再硬截断丢字)——优先在中点附近的标点后断，
    否则按中点断。最多两行。"""
    title = title.strip()
    if len(title) <= max_per_line:
        return [title]
    puncts = "：:！!？?，,、；;·"
    mid = len(title) // 2
    best = None
    for i, ch in enumerate(title[:-1]):
        if ch in puncts and (best is None or abs(i - mid) < abs(best - mid)):
            best = i
    cut = (best + 1) if best is not None else mid
    return [title[:cut].strip(), title[cut:].strip()]


def _metallic_text_layer(lines: list[str], font, canvas_w: int, canvas_h: int,
                         y0: int, line_h: int) -> "Image.Image":
    """金属质感标题层(用户给了参考图定的风格)：银白渐变字面 + 厚黑描边 +
    错位模糊投影 + 轻微右倾(仿斜)。返回可直接 alpha_composite 的 RGBA 层。"""
    from PIL import ImageFilter

    layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    # 投影层：黑字模糊后错位叠底
    sh = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    dsh = ImageDraw.Draw(sh)
    for i, ln in enumerate(lines):
        dsh.text((canvas_w // 2 + 7, y0 + i * line_h + 9), ln, font=font,
                 fill=(0, 0, 0, 200), anchor="mm", stroke_width=8, stroke_fill=(0, 0, 0, 200))
    layer.alpha_composite(sh.filter(ImageFilter.GaussianBlur(5)))

    # 描边层：厚黑描边(墨蓝黑,近似参考图)
    outline = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    dol = ImageDraw.Draw(outline)
    for i, ln in enumerate(lines):
        dol.text((canvas_w // 2, y0 + i * line_h), ln, font=font,
                 fill=(10, 14, 24, 255), anchor="mm", stroke_width=8, stroke_fill=(10, 14, 24, 255))
    layer.alpha_composite(outline)

    # 字面：银白金属纵向渐变(亮→灰→暗→回亮,模拟高光反射带)
    mask = Image.new("L", (canvas_w, canvas_h), 0)
    dmk = ImageDraw.Draw(mask)
    for i, ln in enumerate(lines):
        dmk.text((canvas_w // 2, y0 + i * line_h), ln, font=font, fill=255, anchor="mm")
    stops = [(0.00, (252, 253, 255)), (0.38, (216, 222, 232)), (0.55, (128, 136, 150)),
             (0.68, (238, 241, 246)), (1.00, (170, 176, 188))]
    grad = Image.new("RGB", (1, canvas_h))
    for y in range(canvas_h):
        t = y / max(canvas_h - 1, 1)
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t0 <= t <= t1:
                k = (t - t0) / max(t1 - t0, 1e-6)
                grad.putpixel((0, y), tuple(int(a + (b - a) * k) for a, b in zip(c0, c1)))
                break
    face = grad.resize((canvas_w, canvas_h)).convert("RGBA")
    face.putalpha(mask)
    layer.alpha_composite(face)

    # 轻微右倾(仿斜),贴近参考图的动感
    shear = 0.10
    return layer.transform(
        (canvas_w, canvas_h), Image.AFFINE,
        (1, shear, -shear * canvas_h * 0.5, 0, 1, 0), resample=Image.BICUBIC)


def make_thumbnail(dest: Path, title: str, bg: Path, watermark: Path) -> None:
    """背景图(或纯色兜底) + 底部渐变 + 大字中文标题(自动缩字号) + 右上角水印面板。"""
    W, H = 1280, 720
    img = None
    if bg and Path(bg).exists() and Path(bg).stat().st_size > 0:
        try:
            img = Image.open(bg).convert("RGB").resize((W, H))
        except Exception:                          # 素材非法/损坏(SDXL 挂了/占位字节)→回退纯色底
            img = None
    if img is None:
        img = Image.new("RGB", (W, H), (10, 12, 28))
    img = img.convert("RGBA")

    # 底部渐变遮罩：从画面 45% 处开始渐暗到底部，保证白字在任何背景上都清晰
    grad = Image.new("L", (1, H), 0)
    for y in range(H):
        t = max(0.0, (y - H * 0.45) / (H * 0.55))
        grad.putpixel((0, y), int(200 * min(1.0, t)))
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow.putalpha(grad.resize((W, H)))
    img = Image.alpha_composite(img, shadow)

    lines = _wrap_title(title)                        # 长标题折两行，不丢字
    font = _fit(TITLE_FONT, max(lines, key=len), int(W * 0.86), start=100, lo=40)
    lh = font.size + 14
    y0 = H - 92 - lh * (len(lines) - 1)               # 多行整体贴底部
    img.alpha_composite(_metallic_text_layer(lines, font, W, H, y0, lh))

    if watermark and Path(watermark).exists():
        try:
            wm_src = Image.open(watermark).convert("RGBA")
            panel = _panel_with_watermark(wm_src)
            margin = 24
            img.alpha_composite(panel, (W - margin - panel.width, margin))
        except Exception:               # 水印缺失/损坏不该崩最后一阶段，缩略图照样出
            pass

    dest.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(dest, "JPEG", quality=88, optimize=True)


def run(ctx: Ctx) -> None:
    zh_script = read_json(ctx.job, INPUT)
    source = read_json(ctx.job, SOURCE_INPUT) if exists(ctx.job, SOURCE_INPUT) else {}

    meta = extract_json(ctx.llm.generate(SYSTEM, build_prompt(zh_script, source)))

    attrib = _attribution_line(source)
    if attrib:
        meta["description"] = (meta.get("description") or "") + "\n\n" + attrib

    image_prompt = meta.get("image_prompt") or DEFAULT_IMAGE_PROMPT
    remote = get_remote(ctx.config)
    try:
        bg_path = remote.generate_image(image_prompt, ctx.job / "thumb_bg.png")
    except Exception as e:                # zion 挂/SDXL 失败不该中止整条 run；回退纯色底，视频照发
        ctx.warn(f"{ctx.slug}: d8_meta SDXL 封面背景生成失败({type(e).__name__})，改用纯色底")
        bg_path = ctx.job / "thumb_bg_missing.png"   # 不存在 → make_thumbnail 自动回退纯色底

    title = meta.get("title") or source.get("title") or "未解之谜"
    watermark = Path(ctx.config.get("p7.watermark", DEFAULT_WATERMARK))
    make_thumbnail(ctx.job / "thumb.jpg", title, bg_path, watermark)

    write_json(ctx.job, OUTPUT, {
        "title": title,
        "description": meta.get("description", ""),
        "tags": meta.get("tags", []),
    })
