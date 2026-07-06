"""P7 T20：阶段⑨c——追加发布英文原声源片（未配音的 `source.mp4`，非中文配音的
`video.mp4`）到第二个 YouTube 账号/频道（英文原片档案频道）。

跟 d9b_bilibili 同一个 best-effort 失败隔离思路：这是「主发布之外」的追加发布，
绝不能拖累已经成功的中文配音版 YouTube 发布（s9_publish/d9_publish）、B 站发布
（d9b_bilibili）或归档（d10_archive）——三者都已跑完/在跑，这条只是锦上添花。

第二账号 = 第二份 OAuth token 文件（默认 `secrets/token_en.json`），跟主频道
（`youtube.token`，默认 `secrets/token.json`）用同一份 `client_secret.json`，
只是浏览器授权时选了另一个 Google 账号：
    python -m src.auth_youtube secrets/token_en.json

刻意不用 `src.youtube.get_uploader`——那个读的是 `youtube.*`（主频道配置），
这里要读独立的 `youtube_en.*`，所以本模块自己有一个小的 `_uploader_for`，
形状照抄 get_uploader（`youtube_en.provider` 默认 "youtube"，"fake" 走
FakeUploader 供测试离线注入，不联网）。

失败语义：
  - `youtube_en.enabled` 不是 true（默认 false，笔记本关；zion 生产按需开）→
    status="disabled"，直接返回，不碰任何文件/网络。
  - token 文件不存在（还没走一次性授权，provider=youtube 时才检查；
    provider=fake 测试不需要 token 文件）→ status="no_login"，只 warn 提示
    先跑 `python -m src.auth_youtube secrets/token_en.json` 选第二个 Google
    账号，不 raise。
  - 上传失败（网络抖动/配额/权限/source.json 缺失等任何异常）→
    status="failed"，带 error 摘要，只 warn，绝不 raise——注意这跟
    s9_publish 不同：s9_publish 是主发布，失败要让 run 中止；这里是 d9b_bilibili
    那种「追加发布」，失败只留标记。
  - 成功 → status="published"，带 videoId/url/visibility；playlist 配了则
    尝试加入，失败走 playlist_error（不作废已成功的上传，同 s9_publish 的模式）。

护栏#3：visibility 从不默认 public——本模块默认 unlisted（英文源片频道走
档案/引流定位，比中文配音频道更保守也不到 private）。
"""
from __future__ import annotations

from pathlib import Path

from src.config import Config
from src.context import Ctx
from src.util.jobs import read_json, write_json
from src.youtube import FakeUploader, Uploader, YouTubeUploader, load_creds

OUTPUT = "publish_en.json"
INPUT = "source.json"          # d1_source.META_OUTPUT

_TITLE_MAX_CHARS = 100         # YouTube 标题上限
_DEFAULT_TAGS = ["NASA", "space", "documentary", "public domain", "archive"]


def _uploader_for(config: Config) -> Uploader:
    """形状照抄 src.youtube.get_uploader，但读 youtube_en.* 而非 youtube.*。"""
    provider = config.get("youtube_en.provider", "youtube")
    if provider == "fake":
        return FakeUploader()
    if provider == "youtube":
        creds = load_creds(
            config.get("youtube_en.token", "secrets/token_en.json"),
            config.get("youtube_en.client_secret", "secrets/client_secret.json"),
        )
        return YouTubeUploader(creds)
    raise ValueError(f"未知 youtube_en provider: {provider}")


def _wrap_en_title(title: str, max_per_line: int = 26) -> list[str]:
    """英文标题按词折行(最多两行)——中文的按字折行对英文会切在单词中间。"""
    words = (title or "").split()
    if not words:
        return [""]
    # 第一行贪心装到 max_per_line,剩余全部进第二行(字号自动缩,绝不丢词——
    # 中文封面当初 title[:14] 丢词的教训,英文这里不再犯)
    cur, i = "", 0
    for i, w in enumerate(words):
        cand = f"{cur} {w}".strip()
        if len(cand) > max_per_line and cur:
            break
        cur = cand
        i += 1
    rest = " ".join(words[i:]).strip()
    return [cur, rest] if rest else [cur]


def make_en_thumbnail(dest: Path, title: str, bg: Path, watermark: Path,
                      font_path: str | None = None) -> None:
    """英文封面:AI 背景 + 底部渐变 + 金属英文标题(词级折行) + 右上英文水印面板。
    复用 d8_meta 的金属字/水印面板组件,只有折行策略换成英文词级。"""
    from PIL import Image
    from src.stages.d8_meta import _metallic_text_layer, _panel_with_watermark
    from src.stages.s8_meta import TITLE_FONT, _fit

    W, H = 1280, 720
    img = None
    if bg and Path(bg).exists() and Path(bg).stat().st_size > 0:
        try:
            img = Image.open(bg).convert("RGB").resize((W, H))
        except Exception:
            img = None
    if img is None:
        img = Image.new("RGB", (W, H), (10, 12, 28))
    img = img.convert("RGBA")

    grad = Image.new("L", (1, H), 0)
    for y in range(H):
        t = max(0.0, (y - H * 0.45) / (H * 0.55))
        grad.putpixel((0, y), int(200 * min(1.0, t)))
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow.putalpha(grad.resize((W, H)))
    img = Image.alpha_composite(img, shadow)

    lines = _wrap_en_title(title)
    font = _fit(font_path or TITLE_FONT, max(lines, key=len), int(W * 0.88), start=88, lo=36)
    lh = font.size + 12
    y0 = H - 88 - lh * (len(lines) - 1)
    img.alpha_composite(_metallic_text_layer(lines, font, W, H, y0, lh))

    if watermark and Path(watermark).exists():
        try:
            wm = Image.open(watermark).convert("RGBA")
            panel = _panel_with_watermark(wm)
            img.alpha_composite(panel, (W - 24 - panel.width, 24))
        except Exception:
            pass
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(dest, "JPEG", quality=88, optimize=True)


def _build_clean_source(ctx) -> tuple[Path, dict] | None:
    """英文版「换乐不换声」:demucs 分离出的 vocals.mp3(英文旁白原声) + CC-BY BGM
    重混,视频流 `-c:v copy` 不重编码。为什么:source.mp4 原声里内嵌片方外购的
    商业配乐,直传第二频道会跟中文版当年一样吃 Content ID 认领(中文版在 d7 已换
    BGM 所以干净);英文版没有配音要替换,但配乐同样得换。

    选曲用 pick_bgm_track(slug crc32 确定性)——跟 d7 中文版选同一首,双语版
    听感一致。vocals 缺失/选曲失败/ffmpeg 失败 → warn + None,回退直传原片
    (老行为),绝不挡上传。成功返回 (clean_path, 曲目条目) 供上传+简介署名。"""
    import subprocess

    from src.stages.d7_assemble import infer_bgm_mood, pick_bgm_track
    from src.util.ffprobe import duration_sec

    from src.util.restore import restore_from_archive

    source = ctx.job / "source.mp4"
    vocals = ctx.job / "vocals.mp3"
    _w = lambda m: ctx.warn(f"{ctx.slug}: {m}")  # noqa: E731
    restore_from_archive(ctx.job, "vocals.mp3", ctx.config, _w)   # 任务#50:归档裁剪后自动找回
    if not vocals.exists():
        ctx.warn(f"{ctx.slug}: vocals.mp3 缺失(归档后没恢复?),英文版直传原声原片(可能吃版权认领)")
        return None
    src_meta = read_json(ctx.job, INPUT) if (ctx.job / INPUT).exists() else {}
    mood = infer_bgm_mood(str(src_meta.get("title", "")), str(src_meta.get("attribution", "")))
    entry = pick_bgm_track(   # 与 d7 同输入同 slug → 中英版永远同一首曲子
        Path(ctx.config.get("p7.bgm_dir", "assets/bgm")), ctx.slug, ctx.warn, mood=mood)
    if entry is None:
        return None
    try:
        dur = duration_sec(source)
        out = ctx.job / "source_clean.mp4"
        audio_fc = (
            f"[1:a]volume={ctx.config.get('p7.bg_gain', 0.5)}[bg];"
            "[bg][2:a]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=300[ducked];"
            "[ducked][2:a]amix=inputs=2:normalize=0,alimiter=limit=0.95[aout]"
        )
        cmd = ["ffmpeg", "-y", "-i", str(source),
               "-stream_loop", "-1", "-t", f"{dur:.3f}", "-i", str(entry["path"]),
               "-i", str(vocals),
               "-filter_complex", audio_fc,
               "-map", "0:v", "-map", "[aout]",
               "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)]
        subprocess.run(cmd, capture_output=True, timeout=3600, check=True)
        if not out.exists() or out.stat().st_size == 0:
            raise RuntimeError("输出为空")
        return out, entry
    except Exception as e:  # noqa: BLE001 — 重混失败退回原片,不挡上传
        ctx.warn(f"{ctx.slug}: 英文版换 BGM 失败({type(e).__name__}),直传原声原片")
        return None


def _extract_frame_bg(video: Path, dest: Path) -> bool:
    """thumb_bg.png 缺失(通常被 d10 归档裁剪)时从源片抽一帧当封面背景。
    取片长 22% 处——躲开片头字幕卡/logo,又不至于深入正片撞上采访人脸。
    任何失败(源片也没了/ffmpeg 缺失)→ False,调用方落回纯色兜底,不 raise。"""
    import subprocess
    if not video.exists():
        return False
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, timeout=60, check=True)
        t = max(1.0, float(probe.stdout.strip()) * 0.22)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.1f}", "-i", str(video),
             "-frames:v", "1", "-vf", "scale=1280:720", str(dest)],
            capture_output=True, timeout=120, check=True)
        return dest.exists() and dest.stat().st_size > 0
    except Exception:  # noqa: BLE001 — 抽帧只是锦上添花
        return False


def _build_meta(source: dict) -> dict:
    title = (source.get("title") or "")[:_TITLE_MAX_CHARS]
    identifier = source.get("identifier")
    attribution = source.get("attribution", "")
    if identifier and not str(identifier).startswith("yt:"):   # yt: 源(youtube_gov)没有 archive 详情页
        archive_url = f"https://archive.org/details/{identifier}"
    else:
        archive_url = source.get("source_url") or ""
    description = (
        f"{attribution}\n\n"
        f"Public domain source: {archive_url}\n\n"
        "This channel curates public-domain space & science films."
    )
    return {"title": title, "description": description, "tags": list(_DEFAULT_TAGS)}


def run(ctx: Ctx) -> None:
    if not ctx.config.get("youtube_en.enabled", False):
        write_json(ctx.job, OUTPUT, {"status": "disabled"})
        return

    provider = ctx.config.get("youtube_en.provider", "youtube")
    token = ctx.config.get("youtube_en.token", "secrets/token_en.json")
    if provider != "fake" and not Path(token).exists():
        ctx.warn(
            f"{ctx.slug}: 英文原片频道未授权（缺 {token}），先跑 "
            f"`python -m src.auth_youtube {token}` 选第二个 Google 账号完成一次性授权，"
            "本次跳过英文原片发布"
        )
        write_json(ctx.job, OUTPUT, {"status": "no_login"})
        return

    visibility = ctx.config.get("youtube_en.visibility", "unlisted")  # 护栏#3: 从不默认公开

    try:
        source = read_json(ctx.job, INPUT)
        meta = _build_meta(source)
        uploader = _uploader_for(ctx.config)
        video = ctx.job / "source.mp4"                # 英文原声原片，非中文配音的 video.mp4
        from src.util.restore import restore_from_archive
        restore_from_archive(ctx.job, "source.mp4", ctx.config,
                             lambda m: ctx.warn(f"{ctx.slug}: {m}"))  # 任务#50
        clean_built = None
        if ctx.config.get("youtube_en.replace_bgm", True):   # 默认换乐:原片内嵌商业配乐会吃 Content ID
            built = _build_clean_source(ctx)
            if built is not None:
                clean_built, bgm = built
                video = clean_built
                attribution = (bgm.get("entry") or {}).get("attribution") or ""
                if attribution:
                    meta["description"] += (
                        "\n\nOriginal soundtrack replaced with royalty-free music:\n" + attribution)
        # 英文封面:AI 背景 + 金属英文标题 + 英文水印(生成失败退回裸背景图,再不行跳过封面)
        thumb = ctx.job / "thumb_en.jpg"
        bg = ctx.job / "thumb_bg.png"
        if not bg.exists() and _extract_frame_bg(video, bg):
            ctx.warn(f"{ctx.slug}: thumb_bg.png 缺失,已从源片抽帧补背景")
        try:
            make_en_thumbnail(
                thumb, meta["title"], bg,
                Path(ctx.config.get("youtube_en.watermark", "assets/watermark_en_tr.png")),
                font_path=ctx.config.get("paths.title_font", None),
            )
        except Exception as e:  # noqa: BLE001 — 封面失败不挡上传
            ctx.warn(f"{ctx.slug}: 英文封面生成失败({type(e).__name__})，退回背景图")
            thumb = ctx.job / "thumb_bg.png"
        result = uploader.upload(video, meta, thumb, visibility)
    except Exception as e:  # noqa: BLE001 — 追加发布失败绝不能拖累主发布/B站发布/归档链条
        ctx.warn(f"{ctx.slug}: 英文原片上传失败({type(e).__name__}: {e})")
        write_json(ctx.job, OUTPUT, {"status": "failed", "error": f"{type(e).__name__}: {e}"})
        return

    result["status"] = "published"
    result["visibility"] = visibility
    if clean_built is not None:
        result["replaced_bgm"] = True
        clean_built.unlink(missing_ok=True)   # 可再生的临时产物(NAS 有 source+vocals),传完即清

    playlist_title = ctx.config.get("youtube_en.playlist", "")
    if playlist_title:
        try:
            playlist_id = uploader.add_to_playlist(result["videoId"], playlist_title)
            result["playlist"] = playlist_title
            result["playlistId"] = playlist_id
        except Exception as e:  # noqa: BLE001 — 播放列表失败不能作废已成功的上传
            result["playlist_error"] = str(e)
            ctx.warn(f"{ctx.slug}: 英文频道加入播放列表 {playlist_title!r} 失败：{e}")

    write_json(ctx.job, OUTPUT, result)
