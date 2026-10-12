"""P7 阶段⑦b：给中文成片加品牌片头 + 片尾 CTA(点赞/订阅/提醒,用户定 2026-07-12)。

d7_assemble 产出 video.mp4(中文配音版,含烧字幕+台标+双音轨)后:
  1. 探测实际 分辨率/帧率(各片不同),片头尾按此动态生成才能无缝拼;
  2. 片头(台标+频道名) + 片尾(台标+"感谢观看"+**图标带**:👍点赞 🔴订阅 🔔打开提醒),
     各带 2 条音轨匹配主片双音轨;图标用 PIL 渲染彩色 emoji+中文标签成 PNG 叠加
     (ffmpeg drawtext 渲染不了彩色 emoji 位图字体);
  3. concat -c copy 拼 片头+主片+片尾 → video.mp4;**原片留 video_core.mp4** 给
     d11_shorts 按原时间戳切 Short(加片头会整体后移)。

只作用于 video.mp4(中文版);英文版发 source.mp4 不受影响。best-effort:任一步失败只
warn+保原片。幂等:video_core.mp4 已存在就跳过。默认关(p7.bookend.enabled)。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.context import Ctx
from src.util.jobs import write_json

OUTPUT = "bookend.json"
VIDEO = "video.mp4"
CORE = "video_core.mp4"

_BG = "0x0b0f16"
_GOLD = "0xd9ad5c"
_WHITE = "0xe8edf6"
_MUTED = "0x8a97ad"
_EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"


def _probe(video: Path) -> tuple[int, int, str]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", str(video)],
        capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    return int(s["width"]), int(s["height"]), str(s.get("r_frame_rate", "30/1"))


def _n_audio(video: Path) -> int:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    return len([x for x in r.stdout.splitlines() if x.strip()])


def _icon_strip(out_png: Path, items: list[tuple[str, str]], cjk_font: str, emoji_font: str,
                target_w: int) -> bool:
    """PIL 渲染 图标带(每列:彩色emoji + 金色中文标签),透明底 PNG。返回是否成功
    (PIL 缺失/emoji 字体找不到 → False,片尾优雅降级为无图标带,不中止)。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:  # noqa: BLE001
        return False
    if not emoji_font or not Path(emoji_font).exists():
        return False
    ef = ImageFont.truetype(emoji_font, 109)          # 彩色emoji原生109px
    lab_px = max(40, int(target_w / 26))
    lf = ImageFont.truetype(cjk_font, lab_px)
    icon_px = int(lab_px * 2.8)
    W = target_w
    H = icon_px + int(lab_px * 1.9)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cols = W // len(items)
    for i, (emo, lab) in enumerate(items):
        cx = cols * i + cols // 2
        tmp = Image.new("RGBA", (140, 140), (0, 0, 0, 0))
        ImageDraw.Draw(tmp).text((0, 0), emo, font=ef, embedded_color=True)
        tmp = tmp.resize((icon_px, icon_px), Image.LANCZOS)
        img.paste(tmp, (cx - icon_px // 2, 0), tmp)
        bb = d.textbbox((0, 0), lab, font=lf)
        d.text((cx - (bb[2] - bb[0]) // 2, icon_px + int(lab_px * 0.5)), lab, font=lf,
               fill=(217, 173, 92, 255))
    img.save(out_png)
    return True


def _fps_arg(fps: str) -> str:
    return fps if "/" in fps else fps.split("/")[0]


def _text(prev: str, out_label: str, font: str, textfile: Path, color: str, fs: int, y: str) -> str:
    return (f"[{prev}]drawtext=fontfile='{font}':textfile='{textfile}':fontcolor={color}:"
            f"fontsize={fs}:x=(w-tw)/2:y={y}[{out_label}]")


def _card(out: Path, dur: float, w: int, h: int, fps: str, font: str, logo: Path | None,
          bgm: Path | None, texts: list[tuple[str, str, int, str]], overlay: Path | None,
          overlay_y: str, n_audio: int) -> None:
    """通用卡片：暗底 + 可选台标 + 多行文字(各带绝对y) + 可选叠加图(图标带) + BGM,音轨数匹配主片。"""
    scale = h / 1080.0
    inputs = ["-f", "lavfi", "-t", f"{dur}", "-i", f"color=c={_BG}:s={w}x{h}:r={fps}"]
    filt = []
    v = "0:v"
    idx = 1
    if logo and logo.exists():
        inputs += ["-loop", "1", "-t", f"{dur}", "-i", str(logo)]
        filt.append(f"[{idx}:v]scale=-1:{int(300 * scale)}[logo]")
        filt.append(f"[{v}][logo]overlay=(W-w)/2:(H-h)/2-{int(300 * scale)}[bl]")
        v = "bl"; idx += 1
    ov_idx = None
    if overlay and overlay.exists():
        inputs += ["-loop", "1", "-t", f"{dur}", "-i", str(overlay)]
        ov_idx = idx; idx += 1
    tmpfiles = []
    for i, (text, color, fs, y) in enumerate(texts):
        tf = out.parent / f".bk_{out.stem}_{i}.txt"
        tf.write_text(text, encoding="utf-8"); tmpfiles.append(tf)
        nv = f"t{i}"
        filt.append(_text(v, nv, font, tf, color, int(fs * scale), y))
        v = nv
    if ov_idx is not None:
        strip_w = int(w * 0.62)
        filt.append(f"[{ov_idx}:v]scale={strip_w}:-1[icons]")
        filt.append(f"[{v}][icons]overlay=(W-w)/2:{overlay_y}[vo]")
        v = "vo"
    filt.append(f"[{v}]fade=t=in:d=0.4,fade=t=out:st={dur - 0.6:.2f}:d=0.6[v]")
    if bgm and bgm.exists():
        inputs += ["-t", f"{dur}", "-i", str(bgm)]
        filt.append(f"[{idx}:a]afade=t=in:d=0.5,afade=t=out:st={dur - 1:.2f}:d=1,volume=0.6,"
                    f"asplit={n_audio}" + "".join(f"[a{j}]" for j in range(n_audio)))
    else:
        filt.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{dur},asplit={n_audio}"
                    + "".join(f"[a{j}]" for j in range(n_audio)))
    maps = ["-map", "[v]"] + sum(([f"-map", f"[a{j}]"] for j in range(n_audio)), [])
    subprocess.run(
        ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filt), *maps,
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
         "-r", _fps_arg(fps), "-c:a", "aac", "-ar", "48000", "-ac", "2", "-shortest", str(out)],
        check=True, capture_output=True, text=True)
    for tf in tmpfiles:
        tf.unlink(missing_ok=True)


def run(ctx: Ctx) -> None:
    cfg = ctx.config
    if not cfg.get("p7.bookend.enabled", False):
        return
    if (ctx.job / CORE).exists():
        return
    video = ctx.job / VIDEO
    if not video.exists():
        ctx.warn(f"{ctx.slug}: d7b_bookend 无 {VIDEO}，跳过")
        return

    intro_sec = float(cfg.get("p7.bookend.intro_sec", 3))
    outro_sec = float(cfg.get("p7.bookend.outro_sec", 6))
    font = cfg.get("paths.title_font_bold", cfg.get("paths.title_font", ""))
    logo = Path(cfg.get("p7.bookend.logo", "assets/avatar_zh.png"))
    bgm = Path(cfg.get("p7.bgm_dir", "assets/bgm")) / cfg.get("p7.bookend.bgm", "Morning.mp3")
    channel = cfg.get("p7.bookend.channel_name", "发现纪录片")
    thanks = cfg.get("p7.bookend.thanks", "感谢观看")
    tail = cfg.get("p7.bookend.tail", "每期解锁一部 NASA 中文配音纪录片")
    icons = [("\U0001F44D", cfg.get("p7.bookend.like_label", "点赞")),
             ("\U0001F534", cfg.get("p7.bookend.sub_label", "订阅")),
             ("\U0001F514", cfg.get("p7.bookend.notify_label", "打开提醒"))]

    try:
        w, h, fps = _probe(video)
        n_audio = max(1, _n_audio(video))
        scale = h / 1080.0
        intro = ctx.job / "_intro.mp4"
        outro = ctx.job / "_outro.mp4"
        strip = ctx.job / "_iconstrip.png"
        # 片头:台标 + 频道名(居中偏下)
        _card(intro, intro_sec, w, h, fps, font, logo, bgm,
              [(channel, _GOLD, 88, f"H/2+{int(80 * scale)}")], None, "", n_audio)
        # 片尾:台标 + 感谢观看 + 图标带 + 副标题
        emoji_font = cfg.get("paths.emoji_font", _EMOJI_FONT)
        has_icons = _icon_strip(strip, icons, font, emoji_font, int(w * 0.62))
        _card(outro, outro_sec, w, h, fps, font, logo, bgm,
              [(thanks, _WHITE, 60, f"H/2+{int(40 * scale)}"),
               (tail, _MUTED, 42, f"H/2+{int(500 * scale)}")],
              strip if has_icons else None, f"H/2+{int(170 * scale)}", n_audio)

        core = ctx.job / CORE
        video.rename(core)
        listf = ctx.job / "_concat.txt"
        listf.write_text("".join(f"file '{p.resolve()}'\n" for p in (intro, core, outro)),
                         encoding="utf-8")
        lang = ["chi", "eng", "und", "und"]
        meta = sum(([f"-metadata:s:a:{j}", f"language={lang[j]}"] for j in range(n_audio)), [])
        r = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listf),
             "-map", "0:v", "-map", "0:a", "-c", "copy", *meta,
             "-movflags", "+faststart", str(video)],
            capture_output=True, text=True)
        if r.returncode != 0 or not video.exists():
            ctx.warn(f"{ctx.slug}: d7b_bookend concat copy 失败({r.stderr[-160:]})，改重编码")
            aud = "".join(f"[{i}:a:{j}]" for i in range(3) for j in range(n_audio))
            fc = ("".join(f"[{i}:v]scale={w}:{h},fps={fps},setsar=1[v{i}];" for i in range(3))
                  + f"[v0][v1][v2]{aud}concat=n=3:v=1:a={n_audio}[v]"
                  + "".join(f"[a{j}]" for j in range(n_audio)))
            maps = ["-map", "[v]"] + sum(([f"-map", f"[a{j}]"] for j in range(n_audio)), [])
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(intro), "-i", str(core), "-i", str(outro),
                 "-filter_complex", fc, *maps, *meta, "-c:v", "libx264", "-preset", "medium",
                 "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000",
                 "-movflags", "+faststart", str(video)],
                check=True, capture_output=True, text=True)
        for f in (intro, outro, strip, listf):
            f.unlink(missing_ok=True)
        write_json(ctx.job, OUTPUT, {"intro_sec": intro_sec, "outro_sec": outro_sec,
                                     "core": CORE, "icons": [i[1] for i in icons],
                                     "resolution": f"{w}x{h}"})
    except subprocess.CalledProcessError as e:
        ctx.warn(f"{ctx.slug}: d7b_bookend 失败({str(e.stderr)[-160:]})，保持原片")
        if (ctx.job / CORE).exists() and not video.exists():
            (ctx.job / CORE).rename(video)
    except Exception as e:  # noqa: BLE001
        ctx.warn(f"{ctx.slug}: d7b_bookend 异常({type(e).__name__}: {str(e)[:80]})，保持原片")
        if (ctx.job / CORE).exists() and not video.exists():
            (ctx.job / CORE).rename(video)
