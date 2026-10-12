from __future__ import annotations

import json
import re
from pathlib import Path

from src.context import Ctx
from src.media import _ARCHIVE_SEARCH_COLLECTIONS, _is_blocked, _norm_license
from src.util.jobs import write_json
from src.util.registry import in_flight_identifiers, published_identifiers

OUTPUT = "source.mp4"
META_OUTPUT = "source.json"

SEARCH_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata"

DEFAULT_QUERY = "space OR apollo OR NASA OR planet"
DEFAULT_MIN_SEC = 300
DEFAULT_MAX_SEC = 1800
DEFAULT_MAX_BYTES = 500_000_000       # ~500MB：整片要清晰但不要单片吃满磁盘/带宽
DEFAULT_ROWS = 30

# live 探针实测坑（同 s6 ArchiveVideo）：几KB的占位/损坏衍生文件（如 "Download.mp4"）
# 若不设下限会被误当"可用文件"选中。
_MIN_PLAUSIBLE_VIDEO_BYTES = 200_000

# archive.org 对没带 User-Agent 的请求（含 urllib 默认 UA）经常直接回空 {}——
# 不是 404/500，是"看起来成功但没有数据"，必须显式带一个像样的 UA。
_UA = {"User-Agent": "Mozilla/5.0 (compatible; ufo-doc-pipeline/1.0; +https://archive.org/details/prelinger)"}


def _get(url: str, *, params: dict | None = None, timeout: int = 20, stream: bool = False):
    """统一带 UA 的 GET（archive.org 需要真实 User-Agent，否则可能回空 {}）。"""
    import requests

    r = requests.get(url, params=params, headers=_UA, timeout=timeout, stream=stream)
    r.raise_for_status()
    return r


def _search(query: str, collections: tuple[str, ...], rows: int = DEFAULT_ROWS) -> list[dict]:
    """限定在可信 PD 集合内搜（护栏#2 的前置门）——不限定会捞到 mis-tag 的版权内容
    （实测：开放搜索能捞到被标 PD 的 GTA 游戏录屏）。按下载量降序，优先热门（更可能是
    完整正片而非用户上传的碎片/预告）。"""
    cols = " OR ".join(collections)
    params = {
        "q": f"({query}) AND mediatype:movies AND collection:({cols})",
        "fl[]": ["identifier", "title", "licenseurl", "collection", "subject", "description"],
        "sort[]": "downloads desc",
        "rows": rows,
        "output": "json",
    }
    r = _get(SEARCH_URL, params=params)
    return (r.json().get("response") or {}).get("docs", [])


def _fetch_metadata(identifier: str) -> dict:
    r = _get(f"{METADATA_URL}/{identifier}")
    return r.json()


def _join(v) -> str:
    return " ".join(v) if isinstance(v, list) else str(v or "")


def _parse_duration(length) -> float:
    """metadata.files[].length 有的是纯秒数字符串("187.32")，有的是 HH:MM:SS/MM:SS——
    两种都要能解析；解析失败一律当 0（后续时长门会自然拒绝）。"""
    if length is None:
        return 0.0
    s = str(length).strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        pass
    try:
        parts = [float(p) for p in s.split(":")]
    except ValueError:
        return 0.0
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def _size_of(f: dict) -> int:
    try:
        return int(f.get("size") or 0)
    except (TypeError, ValueError):
        return 0


def _pixels(f: dict) -> int:
    try:
        return int(f.get("width") or 0) * int(f.get("height") or 0)
    except (TypeError, ValueError):
        return 0


def _resolution_of(f: dict) -> str:
    w, h = f.get("width"), f.get("height")
    if w and h:
        return f"{w}x{h}"
    return "unknown"


def _pick_video_file(
    files: list[dict], min_sec: float, max_sec: float, max_bytes: int,
    min_height: int = 0,
) -> tuple[dict, float] | None:
    """整片要清晰：mp4 优先，按分辨率(退化到体积)从高到低尝试；超过 max_bytes 的
    直接跳过尝试下一档而非拒绝整条 item(常见：附带体积超大的母版文件旁边有合适的
    mp4 派生档)。占位/损坏小文件与时长不在范围内的一律跳过。全不过 -> None。"""
    mp4s = [f for f in files if str(f.get("name", "")).lower().endswith(".mp4")]
    if not mp4s:
        return None

    ranked = sorted(mp4s, key=lambda f: (_pixels(f), _size_of(f)), reverse=True)
    for f in ranked:
        try:
            h = int(f.get("height") or 0)
        except (TypeError, ValueError):
            h = 0
        if min_height and h < min_height:      # 画质门(p7.min_height):zion生产档只收HD
            continue
        size = _size_of(f)
        if size < _MIN_PLAUSIBLE_VIDEO_BYTES or size > max_bytes:
            continue
        dur = _parse_duration(f.get("length"))
        if not (min_sec <= dur <= max_sec):
            continue
        return f, dur
    return None


def _attribution(title: str, collection) -> str:
    cols = collection if isinstance(collection, list) else ([collection] if collection else [])
    cols_norm = {str(c).strip().lower() for c in cols if c}
    if "nasa" in cols_norm:
        return f"NASA / {title}"
    return f"Internet Archive / {title}"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = _get(url, timeout=60, stream=True)
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(chunk_size=1 << 16):
            if chunk:
                fh.write(chunk)


# 无解说/纯素材信号：这类只有画面+背景乐(NASA UHD/4K 飞越、fisheye 穿梭等)，
# 转录会得 0 段→没得翻译→出没中文的废片。宁可跳过，也不选这种源。
_NON_NARRATED = (
    "fly-through", "flythrough", "fly through", "fisheye", "b-roll", "b roll",
    "timelapse", "time-lapse", "time lapse", "montage", "no narration", "no audio",
    "silent", "music only", "screensaver", "ambient", "ultra high definition",
    "ultra-high", "uhd", "4k", "8k", "flyover", "aerials", "raw footage",
    "resource reel", "stock footage", "highlight reel",   # NASA 给创作者的素材卷，故意无旁白
    "onboard film", "onboard camera", "16mm",             # 机载胶片(真跑踩坑:阿波罗11机载胶片连音轨都没有)
    "home movie", "home movies", "stock shots",           # 2K胶扫无声家庭影像/素材卷(自然类HD普查:
    "outtakes", "screen tests", "camera test",            # Prelinger 高清池几乎全是这类,画质够但没解说)
)

# 会议/发布会/直播形式(用户定 2026-07-08:降权,不排除)。这类是长时间松散讲话
# (town hall/发布会/听证/网研会),不是影像叙事纪录片——排到点赞排序最后,有
# 制作旁白的纪录片优先命中。命中不拦截,只把 sort 优先级压到底。
_MEETING_FORMAT = (
    "town hall", "briefing", "press conference", "news conference", "teleconference",
    "hearing", "webinar", "media availability", "roundtable", "fireside", "panel",
    "q&a", "ama", "town-hall", "all-hands", "town hall meeting", "presser",
    "state of nasa", "advisory council", "committee meeting", "public meeting",
    # 采访/对谈形式(用户定 2026-07-09,再确认 2026-07-14:不要会议/发布会/采访等)——
    # 同属松散讲话非叙事片
    "interview", "采访", "in conversation", "conversation with", "sit-down",
    "one-on-one", "fireside chat", "ask the", "office hours",
    "发布会", "记者会", "座谈", "对谈", "listening session", "testimony",
    "press briefing", "media roundtable",
)


def _is_meeting_format(title: str) -> bool:
    t = (title or "").lower()
    return any(k in t for k in _MEETING_FORMAT)


def _has_audio(path: Path) -> bool:
    """下载后实测有没有音轨——标题启发式永远猜不全(真跑:APOLLO 11 16MM ONBOARD
    FILM 无任何音轨,走到 d2 提音频才炸)。ffprobe 数 audio 流,0 条即拒。"""
    import subprocess

    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def _looks_narrated(*texts: str) -> bool:
    """标题/主题/描述命中无解说信号则判为纯素材(返回 False)。启发式，配合 d2 的
    0 段兜底：即便这里漏判，d2 转出 0 段也会 StageBlocked 拦住，不会静默出废片。"""
    hay = " ".join(t for t in texts if t).lower()
    return not any(k in hay for k in _NON_NARRATED)


_YTDLP_EXTRA: list[str] = []          # yt-dlp 附加参数(cookie 等),_run_youtube_gov 从配置设


def _set_ytdlp_extra(cfg) -> None:
    """从 p7.ytdlp_cookies_browser 配置 cookie 来源(2026-07-08:YouTube 反爬需登录
    cookie,zion 桌面机从 Firefox 提)。空/未配则不带 cookie(archive 源不需要)。"""
    global _YTDLP_EXTRA
    extra: list[str] = []
    b = cfg.get("p7.ytdlp_cookies_browser", "")
    cf = cfg.get("p7.ytdlp_cookies_file", "")
    if b:
        extra += ["--cookies-from-browser", str(b)]
    elif cf:
        # 无浏览器的无头机(如 pluto,2026-07-11):用导出的 Netscape cookie 文件
        extra += ["--cookies", str(Path(str(cf)).expanduser())]
    # YouTube n-challenge(2026-07-08):需 deno + EJS 解算器才能拿到视频流,否则只有
    # 图片。ejs:github 拉远程解算脚本(zion 已装 deno)。留空关闭。
    rc = cfg.get("p7.ytdlp_remote_components", "ejs:github")
    if rc:
        extra += ["--remote-components", str(rc)]
    _YTDLP_EXTRA = extra


def _yt_videos_url(channel: str) -> str:
    """@handle / 完整URL 都归一成频道 Videos 标签页 URL。"""
    if channel.startswith("http"):
        return channel if channel.rstrip("/").endswith("/videos") else channel.rstrip("/") + "/videos"
    return f"https://www.youtube.com/{channel}/videos"


def _yt_list_channel(ytdlp: str, channel: str, limit: int) -> list[dict]:
    """yt-dlp flat-playlist 列频道视频:[{id,title,duration,channel},...]。真网络,测试必打桩。"""
    import subprocess
    out = subprocess.run(
        [ytdlp, *_YTDLP_EXTRA, "--flat-playlist", "-J", "--playlist-end", str(limit), _yt_videos_url(channel)],
        capture_output=True, text=True, timeout=600, check=True)
    data = json.loads(out.stdout)
    return [e for e in (data.get("entries") or []) if e]


def _yt_fetch_entry(ytdlp: str, video_id: str) -> dict:
    """按 id 抓单个视频的选片元数据(定点指定用,视频可能超出频道列表深度)。"""
    import subprocess
    out = subprocess.run(
        [ytdlp, *_YTDLP_EXTRA, "-J", "--no-playlist", f"https://www.youtube.com/watch?v={video_id}"],
        capture_output=True, text=True, timeout=120, check=True)
    d = json.loads(out.stdout)
    return {"id": d.get("id"), "title": d.get("title"), "duration": d.get("duration"),
            "channel": d.get("channel") or d.get("uploader"),
            "uploader_id": d.get("uploader_id"), "channel_id": d.get("channel_id"),
            "view_count": d.get("view_count"), "like_count": d.get("like_count"),
            "live_status": d.get("live_status")}


def _yt_video_stats(ytdlp: str, video_id: str) -> tuple[int, int, int]:
    """单条视频 (播放量, 点赞数, 上传时间戳epoch)。频道扁平列表这些恒为空(实测
    2026-07-03),只能逐条 --print,不下载。任一取不到记 0(时间戳 0=未知,排近期时垫底)。"""
    import subprocess
    try:
        out = subprocess.run(
            [ytdlp, *_YTDLP_EXTRA, "-s", "--print", "%(view_count)s|%(like_count)s|%(timestamp)s",
             "--no-playlist", f"https://www.youtube.com/watch?v={video_id}"],
            capture_output=True, text=True, timeout=60, check=True)
        parts = out.stdout.strip().splitlines()[-1].split("|")
        toint = lambda x: int(x) if x.strip().lstrip("-").isdigit() else 0  # noqa: E731
        return toint(parts[0]), toint(parts[1] if len(parts) > 1 else ""), \
            toint(parts[2] if len(parts) > 2 else "")
    except Exception:
        return 0, 0, 0


def _yt_download(ytdlp: str, video_id: str, dest: Path, min_height: int, max_bytes: int) -> None:
    """按高度门槛下载合并成 mp4;无满足门槛的格式时 yt-dlp 非零退出,调用方跳过该片。"""
    import subprocess
    fmt = (f"bestvideo[height>={min_height}][ext=mp4]+bestaudio[ext=m4a]"
           f"/best[height>={min_height}][ext=mp4]")
    subprocess.run(
        [ytdlp, *_YTDLP_EXTRA, "-f", fmt, "--max-filesize", str(max_bytes), "--merge-output-format", "mp4",
         "-o", str(dest), "--no-playlist", f"https://www.youtube.com/watch?v={video_id}"],
        capture_output=True, text=True, timeout=7200, check=True)


def _yt_download_subs(ytdlp: str, video_id: str, job: Path) -> bool:
    """尽力下载英文字幕(人工优先,自动兜底)为 job/source_subs.json3。
    平台字幕是 youtube_gov 片源的转录真值来源(见 d2/ytsubs);拿不到不报错,
    d2 自动回退 whisper。"""
    import subprocess
    for flag, langs in (("--write-subs", "en.*,en"), ("--write-auto-subs", "en-orig,en")):
        try:
            subprocess.run(
                [ytdlp, *_YTDLP_EXTRA, "--skip-download", flag, "--sub-langs", langs,
                 "--sub-format", "json3", "-o", str(job / "source_subs.%(ext)s"),
                 "--no-playlist", f"https://www.youtube.com/watch?v={video_id}"],
                capture_output=True, text=True, timeout=300, check=True)
        except Exception:  # noqa: BLE001
            continue
        hits = sorted(job.glob("source_subs*.json3"))
        if hits:
            if hits[0] != job / "source_subs.json3":
                hits[0].replace(job / "source_subs.json3")
                for extra in hits[1:]:
                    extra.unlink(missing_ok=True)
            return True
    return False


def _probe_height(path: Path) -> int:
    """ffprobe 实测视频高度(下载后纵深校验,不信任元数据)。"""
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=height", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60, check=True)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def _run_youtube_gov(ctx: Ctx) -> None:
    """P7 题材2供给:官方**美国联邦政府**频道白名单选片(NPS/USFWS/NOAA...)。

    合法性前提(护栏#2 的 youtube 形态):联邦政府雇员职务作品法定公有领域
    (17 USC §105),**不论 YouTube 页面标什么许可**。因此本分支的许可门 =
    `p7.yt_channels` 白名单本身——只能放官方联邦机构频道;州立机构/承包商/
    个人频道不适用该法条,加进来就是破门,review 时必须拦。

    其余门禁与 archive 分支完全同款:登记表去重(yt:<id>)、标题黑名单、
    无解说关键词、时长窗口、HD 地板(下载前格式过滤+下载后 ffprobe 实测)、
    音轨探测。"""
    cfg = ctx.config
    ytdlp = str(Path(str(cfg.get("p7.ytdlp_bin", "yt-dlp"))).expanduser())
    _set_ytdlp_extra(cfg)          # YouTube 反爬:带上 cookie(见 _set_ytdlp_extra)
    channels = list(cfg.get("p7.yt_channels", []))
    if not channels:
        raise RuntimeError("d1_source: p7.source=youtube_gov 但 p7.yt_channels 为空")
    min_sec = cfg.get("p7.min_sec", DEFAULT_MIN_SEC)
    max_sec = cfg.get("p7.max_sec", DEFAULT_MAX_SEC)
    max_bytes = cfg.get("p7.max_bytes", DEFAULT_MAX_BYTES)
    limit = int(cfg.get("p7.rows", DEFAULT_ROWS))
    floor = max(int(cfg.get("p7.min_height", 700)), 700)   # 用户硬约束:HD 地板同 archive 分支

    skip_ids = published_identifiers(ctx.root) | in_flight_identifiers(ctx.root, ctx.slug)

    # 用户定(2026-07-03):全部白名单频道汇总后按播放量降序选——流量最高的先做。
    # 顺序:先过零成本门(去重/黑名单/无解说/时长),幸存者才逐条拉真实播放量再排序。
    pool: list[tuple[dict, str]] = []
    for channel in channels:
        try:
            pool += [(e, channel) for e in _yt_list_channel(ytdlp, channel, limit)]
        except Exception as e:
            ctx.warn(f"d1_source: 频道 {channel} 列表失败({type(e).__name__})，跳过")

    # 题材硬排除(p7.avoid_keywords,用户定 2026-07-10):标题含排除词的整片剔除,不进候选
    # (如自然线已连发多部鱼片→avoid:[salmon,fish]换哺乳类)。留空不启用;与 prefer_keywords 互补。
    avoid = [str(k).lower() for k in (cfg.get("p7.avoid_keywords", []) or [])]
    # 题材硬要求(p7.require_keywords,用户定 2026-07-10):标题必须含至少一个要求词,否则剔除
    # (定向要某一类如"鸟类"时用;prefer 只软排序,require 是硬门)。留空不启用。
    require = [str(k).lower() for k in (cfg.get("p7.require_keywords", []) or [])]

    survivors: list[tuple[dict, str]] = []
    for e, channel in pool:
        vid = e.get("id")
        if not vid or f"yt:{vid}" in skip_ids:
            continue
        title = str(e.get("title") or vid)
        # 用户定(2026-07-03):直播回放一律不收——live_status 硬判 + 标题词兜底
        # (直播档是多小时松散解说,不是成片纪录片;NASA 发射直播还常年霸播放量榜)
        if e.get("live_status") in ("is_live", "was_live", "is_upcoming", "post_live"):
            continue
        if re.search(r"\blive\b|\blivestream\b|\breplay\b", title, re.IGNORECASE):
            continue
        if _is_blocked(title, "", "") or not _looks_narrated(title, ""):
            continue
        # 会议/发布会/采访硬排除(用户定 2026-07-09,p7.exclude_meeting=true 时启用):
        # 默认仍是降权(排序压后),开关打开则整类剔除,不进候选。
        if cfg.get("p7.exclude_meeting", False) and _is_meeting_format(title):
            continue
        if avoid and any(k in title.lower() for k in avoid):
            continue
        if require and not any(k in title.lower() for k in require):
            continue
        dur = float(e.get("duration") or 0)
        if not (min_sec <= dur <= max_sec):
            continue
        survivors.append((e, channel))
    for e, _ in survivors:
        if not e.get("view_count") or e.get("like_count") is None or not e.get("timestamp"):
            v, l, ts = _yt_video_stats(ytdlp, e["id"])
            e["view_count"] = e.get("view_count") or v
            e["like_count"] = e.get("like_count") if e.get("like_count") is not None else l
            e["timestamp"] = e.get("timestamp") or ts
    # 近期时间窗(p7.max_age_days,用户定 2026-07-14):只留上传在 N 天内的片。配合
    # sort_by=views 即"近期热点"(近期窗内最高播放);0/未配则不限时。timestamp 未知(0)一律剔除。
    max_age_days = int(cfg.get("p7.max_age_days", 0) or 0)
    if max_age_days > 0:
        import time
        cutoff = time.time() - max_age_days * 86400
        kept = [(e, ch) for e, ch in survivors if (e.get("timestamp") or 0) >= cutoff]
        ctx.warn(f"d1_source: 近期窗 {max_age_days} 天 → {len(kept)}/{len(survivors)} 片入选")
        survivors = kept

    # 题材偏好(p7.prefer_keywords,用户定 2026-07-09):标题含偏好词的整体排最前
    # (自然线设动物词→优先动物片,不再撞 NPS 频道混入的历史/文化片)。留空不启用。
    prefer = [str(k).lower() for k in (cfg.get("p7.prefer_keywords", []) or [])]

    def _prefer_hit(title: str) -> bool:
        return bool(prefer) and any(k in title.lower() for k in prefer)

    # 用户定(2026-07-04):好评(点赞)优先,播放量次之。
    # 会议/发布会形式降权(用户定 2026-07-08):非会议片整体排在会议片前面,组内再
    # 按点赞——即 town hall 这类哪怕点赞高也排到所有纪录片之后,不排除只压后。
    # 排序优先级:题材偏好命中 > 非会议 > 点赞 > 播放量(都降序)。
    # p7.sort_by(用户定 2026-07-10/14):"views"=播放量优先;"recent"=上传时间优先(近期热点,
    # 按 timestamp 降序,新片在前,同时点赞次之);默认(空)=点赞优先。
    _sort = str(cfg.get("p7.sort_by", "") or "").lower()

    def _metric(e: dict) -> tuple:
        v, l, ts = e.get("view_count") or 0, e.get("like_count") or 0, e.get("timestamp") or 0
        if _sort == "recent":
            return (ts, l)
        if _sort == "views":
            return (v, l)
        return (l, v)

    survivors.sort(key=lambda t: (_prefer_hit(str(t[0].get("title", ""))),
                                  not _is_meeting_format(str(t[0].get("title", ""))),
                                  *_metric(t[0])),
                   reverse=True)

    # 定点指定(p7.yt_pin_ids):人工点名的视频强制排队首(如系列片续集),仍过
    # 去重/时长门;不在频道列表里(超出 rows 深度)则直接按 id 抓元数据补进来。
    pins = [str(v) for v in (cfg.get("p7.yt_pin_ids", []) or [])]
    if pins:
        by_id = {e.get("id"): (e, ch) for e, ch in survivors}
        pinned: list[tuple[dict, str]] = []
        for vid in pins:
            if f"yt:{vid}" in skip_ids:
                ctx.warn(f"d1_source: 定点 {vid} 已发布/在途，跳过")
                continue
            if vid in by_id:
                pinned.append(by_id.pop(vid))
            else:
                try:
                    ent = _yt_fetch_entry(ytdlp, vid)
                    # 许可门(不可松):定点视频也必须来自白名单联邦频道,否则 PD 署名
                    # 就是假的(同名山寨号一堆)。uploader_id(@handle)对白名单校验。
                    up = str(ent.get("uploader_id") or "").lstrip("@").lower()
                    wl = {c.lstrip("@").lower() for c in channels}
                    if up not in wl:
                        ctx.warn(f"d1_source: 定点 {vid} 来源频道 @{up} 不在白名单，"
                                 "拒绝(非联邦官方=非法定PD)")
                    elif min_sec <= float(ent.get("duration") or 0) <= max_sec:
                        pinned.append((ent, ent.get("channel") or "pin"))
                    else:
                        ctx.warn(f"d1_source: 定点 {vid} 时长超范围，跳过")
                except Exception as ex:  # noqa: BLE001
                    ctx.warn(f"d1_source: 定点 {vid} 抓取失败({type(ex).__name__})，跳过")
        survivors = pinned + [t for t in survivors if t[0].get("id") in by_id]

    for e, channel in survivors:
        vid = e["id"]
        identifier = f"yt:{vid}"
        title = str(e.get("title") or vid)
        dur = float(e.get("duration") or 0)
        try:
            _yt_download(ytdlp, vid, ctx.job / OUTPUT, floor, max_bytes)
        except Exception as ex:
            ctx.warn(f"d1_source: {identifier} 下载失败({type(ex).__name__}，可能无≥{floor}p格式)，跳过")
            continue
        height = _probe_height(ctx.job / OUTPUT)
        if height < floor or not _has_audio(ctx.job / OUTPUT):
            ctx.warn(f"d1_source: {identifier} 实测不达标(height={height})或无音轨，弃用")
            (ctx.job / OUTPUT).unlink(missing_ok=True)
            continue
        has_subs = _yt_download_subs(ytdlp, vid, ctx.job)
        if not has_subs:
            ctx.warn(f"d1_source: {identifier} 无可用英文字幕,d2 将用 whisper 转录")
        label = str(e.get("channel") or e.get("uploader") or channel)
        write_json(ctx.job, META_OUTPUT, {
            "identifier": identifier,
            "title": title,
            "license": "public-domain",   # 联邦政府作品(17 USC §105);白名单即许可门
            "attribution": f"Source: {label} (U.S. government work, public domain).",
            "duration": dur,
            "view_count": e.get("view_count"),
            "like_count": e.get("like_count"),
            "source_url": f"https://www.youtube.com/watch?v={vid}",
            "resolution": f"{height}p",
        })
        return
    raise RuntimeError(
        f"d1_source: youtube_gov 频道白名单里没有符合条件的片(channels={channels}, "
        f"{min_sec}-{max_sec}s, >={floor}p) — 加频道/放宽时长后重试")


def run(ctx: Ctx) -> None:
    cfg = ctx.config
    ctx.job.mkdir(parents=True, exist_ok=True)   # 下载落地目录(youtube_gov 分支没有 write_json 先行建目录)
    if cfg.get("p7.source", "archive") == "youtube_gov":
        return _run_youtube_gov(ctx)
    query = cfg.get("p7.query", DEFAULT_QUERY)
    min_sec = cfg.get("p7.min_sec", DEFAULT_MIN_SEC)
    max_sec = cfg.get("p7.max_sec", DEFAULT_MAX_SEC)
    max_bytes = cfg.get("p7.max_bytes", DEFAULT_MAX_BYTES)
    collections = tuple(cfg.get("p7.collections", list(_ARCHIVE_SEARCH_COLLECTIONS)))
    rows = int(cfg.get("p7.rows", DEFAULT_ROWS))   # HD门槛+登记去重会滤掉大量热门,池子要够深

    docs = _search(query, collections, rows=rows)
    # 定点白名单(p7.include_identifiers):人工核验过的条目排最前面优先考虑——
    # 仍要走下面全部门禁(许可二次核验/黑名单/无解说/HD地板/音轨),白名单≠免检
    include = [{"identifier": i} for i in cfg.get("p7.include_identifiers", []) or []]
    docs = include + [d for d in docs if d.get("identifier") not in
                      {i["identifier"] for i in include}]

    # P7 T16：登记表去重——已发布过的、以及同批其它 job 已下载但还没发布完
    # 的（source.json 存在），一律跳过，不然同一条 query 每次都会重挑到
    # 同一部片（downloads desc 排序几乎不变）。
    skip_ids = published_identifiers(ctx.root) | in_flight_identifiers(ctx.root, ctx.slug)

    for d in docs:
        identifier = d.get("identifier")
        if not identifier:
            continue

        if identifier in skip_ids:
            ctx.warn(f"d1_source: {identifier} 已发布/在途，跳过")
            continue

        try:
            meta = _fetch_metadata(identifier)
        except Exception as e:
            ctx.warn(f"d1_source: {identifier} metadata 获取失败({type(e).__name__})，跳过")
            continue

        files = meta.get("files") or []
        if not files:
            # darkened/下架条目 metadata.files 为空——即便 search 阶段过了白名单也要跳过
            ctx.warn(f"d1_source: {identifier} 无可用 files（可能已下架），跳过")
            continue

        info = meta.get("metadata") or {}
        # 纵深防御：护栏#2 在 metadata 上二次核验，不信任 search 阶段的结果
        token = _norm_license(info.get("licenseurl"), info.get("collection"))
        if token is None:
            continue

        title = info.get("title") or d.get("title") or identifier
        subject, desc = _join(info.get("subject")), _join(info.get("description"))
        if _is_blocked(str(title), subject, desc):
            continue
        if not _looks_narrated(str(title), subject, desc):
            ctx.warn(f"d1_source: {identifier} 疑似无解说素材(纯画面/音乐)，跳过")
            continue

        # 用户约束(2026-07-03):所有选片最低 HD——代码级地板 700,配置只能往上调不能往下破
        # (自然类 480p 首发后用户定的硬门槛;配置里写低了也按 700 执行)
        picked = _pick_video_file(files, min_sec, max_sec, max_bytes,
                                   min_height=max(int(cfg.get("p7.min_height", 700)), 700))
        if picked is None:
            continue
        f, duration = picked

        url = f"https://archive.org/download/{identifier}/{f['name']}"
        _download(url, ctx.job / OUTPUT)

        if not _has_audio(ctx.job / OUTPUT):
            ctx.warn(f"d1_source: {identifier} 下载后实测无音轨(纯胶片/无声素材)，弃用换下一部")
            (ctx.job / OUTPUT).unlink(missing_ok=True)
            continue

        write_json(ctx.job, META_OUTPUT, {
            "identifier": identifier,
            "title": title,
            "license": token,
            "attribution": _attribution(title, info.get("collection")),
            "duration": duration,
            "source_url": url,
            "resolution": _resolution_of(f),
        })
        return

    raise RuntimeError(
        f"d1_source: 未找到符合条件的公有领域纪录片"
        f"(query={query!r}, {min_sec}-{max_sec}s, collections={collections}) — "
        "换题材/放宽 p7.min_sec/p7.max_sec 后重试"
    )
