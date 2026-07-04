"""P7 T6：阶段⑦——把 source.mp4(源片视频) + no_vocals.mp3(demucs 去人声后的背景床)
+ narration.json(逐段中文旁白，按 start 时间锚点摆放) + subs.ass(中文字幕) 合成成片
video.mp4。扩展 s7_assemble，思路取自已跑通的
.superpowers/sdd/reference-build_sample.py 第 4 节。

两遍 ffmpeg（比单条塞 100+ 路 -i 的巨型 filter_complex 更稳，样片已验证）：
  1) 旁白轨：每段配音 adelay 到其 start，amix 叠成一路，apad/atrim 补齐/裁到
     源片时长 → narration.wav。
  2) 成片：源片视频原样(不做无谓放大)烧字幕；背景先 volume=bg_gain 压成床，
     再用 sidechaincompress 让旁白路过时自动压低背景(说话时床音量降、间隙
     恢复)——比样片里恒定 volume=0.30 的背景垫更聪明。ducked 背景与旁白
     amix 后 alimiter 防削波，与烧好字幕的视频一起编码。

0 段旁白(narration.json 缺失或 segments 为空)：跳过第①遍，第②遍直接用
背景(压 bg_gain，无旁白可 duck)+字幕合成，warn 不中断——保证任何 job 都能
出片。

P7 T14：`p7.encoder` 编码器开关(默认 `libx264`，笔记本上没有 NVENC)。zion
原生跑管线时设成 `nvenc`（见 config-zion.yaml），第②遍改用
`h264_nvenc -preset p5 -cq 23`——ffmpeg 的 nvenc 编码器要读
`/run/opengl-driver/lib` 下 NixOS 的 NVIDIA 驱动库(已在 zion 上验证)，走
`p7.encoder_lib_path` PREPEND 进 `LD_LIBRARY_PATH`(合并 os.environ，不替换)。
nvenc 编码失败(如驱动没装对/显存被占)不该丢了这条片子——warn 后自动回退
libx264 重试一次，绝不因为 GPU 编码挂了就把整段渲染报废。

背景床来源(`p7.bgm`，默认 replace)：真跑首次上传发现 NASA 公有领域片里常内嵌
片方外购的商业库背景乐，demucs 的 no_vocals 背景床把它原样保留，触发了 4 条
YouTube Content ID 版权认领(不下架但拦变现)。replace 模式改用
`assets/bgm/`(`p7.bgm_dir`) 里的免版税 CC-BY 曲目：按 slug 的 crc32 确定性选一
条(同 slug 永远同曲，换 slug 可能换)，`-stream_loop -1 -t {video_dur}` 循环铺满
源片时长，接同一条 volume=bg_gain + sidechaincompress 侧链——no_vocals.mp3 只是
不再被用作背景床(d2 仍照常产出，留作素材)。选中曲目写进 job 的 bgm.json，供
d8_meta 追加 CC-BY 署名。曲目清单/目录缺失或清单为空 → warn 后回退 original
模式(背景床=no_vocals.mp3)，绝不中止 run。

P7 T18：`p7.video_watermark`(默认开)把频道台标(`p7.watermark`，跟 d8 缩略图共用
同一张透明底 PNG)烧进正片全程——电视台角标样式，右上(TR)/右下(BR)按
`p7.wm_min_sec`~`p7.wm_max_sec`(默认 5~10min)随机间隔切换，从右上开始。切换点
由 slug 的 crc32 派生一个确定性 LCG 生成(`watermark_windows`，跟 pick_bgm_track
选曲同一"不用 random/时间戳"的思路)——同 slug 重跑窗口/位置序列完全一致。水印
输入缩到 140px 宽、转 rgba、调 85% 不透明度，叠在字幕烧完的画面上；BR 位
`H-h-130` 特意抬高，避开烧字幕常驻的底部约 120px 区域，不挡字。视频短于
`wm_min_sec`(整部片子不够切一次)时天然只出一个覆盖全片的 TR 窗口，只叠一路，
省一次 split/overlay。开关关闭、或水印文件缺失/读不到 → 按老逻辑拼命令(无水印
输入、无 overlay 段)，warn 后继续，绝不因为水印功能中止渲染。
"""
from __future__ import annotations

import os
import subprocess
import zlib
from pathlib import Path
from typing import Callable

from src.config import Config
from src.context import Ctx
from src.util.ffprobe import duration_sec
from src.util.jobs import exists, read_json, write_json

OUTPUT = "video.mp4"
NARRATION = "narration.json"
SOURCE = "source.mp4"
BACKGROUND = "no_vocals.mp3"
SUBS = "subs.ass"
NARRATION_TRACK = "narration.wav"          # 中间产物：第①遍拼好的旁白轨
BGM_MANIFEST = "bgm.json"                  # p7.bgm_dir 下的曲目清单文件名
JOB_BGM = "bgm.json"                       # job 目录：记录本次选中曲目(供 d8 署名)
DEFAULT_BGM_DIR = "assets/bgm"
DEFAULT_ENCODER_LIB_PATH = "/run/opengl-driver/lib"    # NVENC 需要的 NixOS NVIDIA 驱动库路径(zion 已验证)
DEFAULT_WATERMARK = "assets/watermark_tr.png"          # P7 T18：跟 d8_meta 缩略图共用同一张透明底台标


NARRATION_GAP_SEC = 0.15      # 相邻旁白间留的喘息间隙
MAX_TEMPO = 1.25              # 按角色分块变速的封顶(用户定 2026-07-04:1.4 密集段听不清;块内统一速率见 _plan)

WM_MIN_SEC = 300.0            # P7 T18：水印切换间隔下限(5min)，config p7.wm_min_sec 覆盖
WM_MAX_SEC = 600.0            # 水印切换间隔上限(10min)，config p7.wm_max_sec 覆盖
WM_TR = "TR"                  # 右上角标位
WM_BR = "BR"                  # 右下角标位(抬高 130px 避开烧字幕区)


def _speaker_key(seg: dict) -> str:
    """"角色"分块键:speaker(未来接分话人) > gender(现有基频检测) > voice。
    没有真正的说话人分离,同性别连续段近似为同一角色的一段话。"""
    return str(seg.get("speaker") or seg.get("gender") or seg.get("voice") or "")


def enrich_speaker_fields(segments: list[dict], job: Path) -> list[dict]:
    """旧 narration.json 没有 gender(2026-07-04 前的 d4 产物)——从 zh_script.json
    按 id 回填,供 _speaker_key 分块。新数据自带 gender,原样返回。"""
    need_gender = not any(s.get("gender") or s.get("speaker") for s in segments)
    need_end = not any(s.get("end") for s in segments)
    if not (need_gender or need_end):
        return segments
    try:
        zh = read_json(job, "zh_script.json").get("segments", [])
        m = {int(x.get("id", -1)): x for x in zh}
        for s in segments:
            x = m.get(int(s.get("id", -1)), {})
            if need_gender:
                s["gender"] = x.get("gender")
            if need_end:
                s["end"] = x.get("end")
    except Exception:  # noqa: BLE001 — 回填失败退化为 voice 键分块/起点差近似间隙,不新增故障
        pass
    return segments


PARA_GAP_SEC = 2.5            # 原片句间隙超过它=自然段落停顿,保留为硬锚点(不吞掉)
SENT_PAUSE_SEC = 0.35         # 段落内重排后的句间呼吸(用户反馈:原空隙太长听着断)


def _orig_gap(prev: dict, seg: dict) -> float:
    """原片里 prev 说完到 seg 开口的间隙。prev 无 end(旧数据未回填)时用
    prev.start+tts_dur 近似(中文时长≈量级参考,只影响分段粗细,不影响正确性)。"""
    prev_end = prev.get("end")
    if prev_end is None:
        prev_end = float(prev.get("start", 0.0)) + float(prev.get("tts_dur", 0.0) or 0.0)
    return float(seg.get("start", 0.0)) - float(prev_end)


def place_segments(segments: list[dict], video_dur: float) -> list[dict]:
    """段落级排布 v2(用户反馈迭代 2026-07-04):

    v1 把句子从段落锚点向左压实,富余全堆在段落尾——画面后半还在说话配音已
    结束,接一段死寂(用户:图像在说话没声音/段落间空白太长)。v2 改为**贴原点
    排布**:每句尽量回到自己的原片起点(空白按原位置留在句间),仅在与前句冲突
    时向后让(至少 SENT_PAUSE 呼吸),或为给后面句子留够空间向前提。效果:
    画面同步以句为粒度恢复,死寂消散,装得下的段落语速恒为 1.0x(用户:尽量≤1x);
    只有真正超密的段落整段提速(封顶 MAX_TEMPO),仍不够只截末句尾。

    段落 = 同角色且原句间隙≤PARA_GAP 的连续句;段落窗口到下一段落锚点为止。
    返回逐段 {"start","factor","dur","cut"};d5 字幕共用,永远贴配音。"""
    n = len(segments)
    if n == 0:
        return []
    paras: list[list[int]] = [[0]]
    for i in range(1, n):
        if (_speaker_key(segments[i]) != _speaker_key(segments[i - 1])
                or _orig_gap(segments[i - 1], segments[i]) > PARA_GAP_SEC):
            paras.append([i])
        else:
            paras[-1].append(i)

    placements: list[dict] = [None] * n  # type: ignore[list-item]
    for pi, idxs in enumerate(paras):
        p_start = float(segments[idxs[0]].get("start", 0.0))
        nxt = (float(segments[paras[pi + 1][0]].get("start", video_dur))
               if pi + 1 < len(paras) else video_dur)
        p_end = max(nxt - NARRATION_GAP_SEC, p_start + 0.3)
        m = len(idxs)
        speech = sum(float(segments[i].get("tts_dur", 0.0) or 0.0) for i in idxs)
        avail = max(p_end - p_start - SENT_PAUSE_SEC * (m - 1), 0.3)
        need = speech / avail if speech > 0 else 1.0
        factor = 1.0 if need <= 1.02 else min(need, MAX_TEMPO)

        durs = [float(segments[i].get("tts_dur", 0.0) or 0.0) / factor for i in idxs]
        # suffix[k] = 第k句起到段落末所需的最小总时长(含句间呼吸)
        suffix = [0.0] * (m + 1)
        for k in range(m - 1, -1, -1):
            suffix[k] = durs[k] + (SENT_PAUSE_SEC + suffix[k + 1] if k + 1 < m else 0.0)
        cur = p_start
        for k, i in enumerate(idxs):
            orig = float(segments[i].get("start", 0.0))
            latest = p_end - suffix[k]                 # 再晚就挤不下后面的句子
            start_i = min(max(orig, cur), max(latest, cur))
            dur = durs[k]
            cut = None
            if k == m - 1:                             # 末句:封顶仍越窗才截尾
                allowed = p_end - start_i
                if dur > allowed + 0.05:
                    cut = max(allowed, 0.3)
            placements[i] = {"start": round(start_i, 3), "factor": factor,
                             "dur": round(min(dur, cut) if cut else dur, 3), "cut": cut}
            cur = start_i + (min(dur, cut) if cut else dur) + SENT_PAUSE_SEC
    return placements


def _plan(segments: list[dict], video_dur: float) -> list[tuple[float, float]]:
    """兼容旧签名:(factor, 该段可播时长)。新逻辑见 place_segments(段落重排)。"""
    return [(p["factor"], p["dur"] if p["cut"] is None else p["cut"])
            for p in place_segments(segments, video_dur)]


def tempo_factors(segments: list[dict], video_dur: float) -> list[float]:
    """逐段算 atempo 压缩比：配音若长过「到下一条旁白开播」的窗口就压回去(留间隙)，
    否则 1.0 保持原速。真跑发现 -18% 慢速让 85% 段溢出槽位、叠到下一条上——
    字幕由 d5 按同一 _plan 跟配音窗口对齐(2026-07-04 修:原"字幕锚原轴"的假设在时长不匹配时错位)。"""
    return [f for f, _ in _plan(segments, video_dur)]


def build_narration_filter(segments: list[dict], video_dur: float) -> str:
    """按 place_segments 的重排结果摆放:adelay 到新起点(段落内顺排),整段落统一
    atempo,只有段落末句可能截尾+淡出(中间句重排后必然装得下)。"""
    n = len(segments)
    placements = place_segments(segments, video_dur)
    parts: list[str] = []
    for idx, (seg, pl) in enumerate(zip(segments, placements)):
        ms = int(round(pl["start"] * 1000))
        chain = f"atempo={pl['factor']:.4f}," if pl["factor"] > 1.0 else ""
        if pl["cut"] is not None:
            fade_start = max(pl["cut"] - 0.2, 0.0)
            chain += f"atrim=0:{pl['cut']:.3f},afade=t=out:st={fade_start:.3f}:d=0.2,"
        parts.append(f"[{idx}:a]{chain}adelay={ms}|{ms}[d{idx}]")
    parts.append("".join(f"[d{idx}]" for idx in range(n)) + f"amix=inputs={n}:normalize=0[nar]")
    # 旁白响度归一到 -16 LUFS(用户反馈 2026-07-04:配音音量太小)——F5 克隆输出
    # 响度不齐且普遍偏轻,归一后旁白稳坐主声部,背景床(bg_gain+侧链)相对自然退后。
    parts.append(f"[nar]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,"
                 f"apad=whole_dur={video_dur},atrim=0:{video_dur}[narp]")
    return ";".join(parts)


def build_narration_cmd(job: Path, segments: list[dict], video_dur: float, out_path: Path) -> list[str]:
    cmd: list[str] = ["ffmpeg", "-y"]
    for seg in segments:
        cmd += ["-i", str(job / seg.get("file", ""))]
    cmd += ["-filter_complex", build_narration_filter(segments, video_dur), "-map", "[narp]", str(out_path)]
    return cmd


_MOOD_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # 顺序即优先级:更具体的场景先命中(ocean 片名往往也含 nature 词)
    ("ocean",     ("ocean", "sea", "whale", "orca", "marine", "fish", "reef", "coral",
                   "underwater", "shark", "dolphin", "tide")),
    ("nature",    ("wildlife", "national park", "forest", "bird", "bear", "salmon", "river",
                   "wilderness", "refuge", "nature", "animal", "wolf", "wolves", "eagle",
                   "everglades", "yellowstone", "conservation")),
    ("space",     ("space", "nasa", "apollo", "orbit", "mars", "moon", "planet", "telescope",
                   "shuttle", "astronaut", "kepler", "galaxy", "cosmos", "satellite", "rocket")),
    ("somber",    ("war", "memorial", "disaster", "tragedy", "accident", "crash", "extinct",
                   "abandoned", "ruins")),
    ("uplifting", ("discovery", "celebration", "anniversary", "achievement", "triumph")),
)


def infer_bgm_mood(*texts: str) -> str | None:
    """从片名/主题/描述关键词推断配乐情绪桶(用户定 2026-07-03:BGM 按影片内容换,
    不再全线同一首)。刻意不走 LLM:确定性、零失败面、同片重跑同曲。命不中 → None
    (选曲时退回全池)。"""
    hay = " ".join(t or "" for t in texts).lower()
    for mood, keys in _MOOD_KEYWORDS:
        if any(k in hay for k in keys):
            return mood
    return None


def pick_bgm_track(bgm_dir: Path, slug: str, warn: Callable[[str], None],
                   mood: str | None = None) -> dict | None:
    """从 `bgm_dir/bgm.json` 清单里挑一条:先按 mood 过滤(条目 moods 含它才入围;
    没配 moods 的条目视作百搭、mood=None 或桶空则全池),再按 slug 的 crc32 确定性
    选(同 slug 永远同曲，换 slug 可能换——不用 Date/random，保证同一 job 重跑
    选曲不变)。

    清单/目录缺失、清单为空、或选中曲目的文件本身不存在，一律 warn 后返回
    None，调用方回退 original 模式——never crash, never silent-fail。"""
    if not exists(bgm_dir, BGM_MANIFEST):
        warn(f"d7 BGM 清单缺失({bgm_dir / BGM_MANIFEST})，回退用 no_vocals 背景床")
        return None
    try:
        manifest = read_json(bgm_dir, BGM_MANIFEST)
    except Exception as e:
        warn(f"d7 BGM 清单解析失败({type(e).__name__}: {e})，回退用 no_vocals 背景床")
        return None
    tracks = manifest.get("tracks") or []
    if not tracks:
        warn(f"d7 BGM 清单为空({bgm_dir / BGM_MANIFEST})，回退用 no_vocals 背景床")
        return None
    if mood:
        bucket = [t for t in tracks if not t.get("moods") or mood in t["moods"]]
        if bucket:
            tracks = bucket
    idx = zlib.crc32(slug.encode("utf-8")) % len(tracks)
    entry = tracks[idx]
    path = bgm_dir / entry.get("file", "")
    if not path.exists():
        warn(f"d7 BGM 选中曲目文件缺失({path})，回退用 no_vocals 背景床")
        return None
    return {"path": path, "entry": entry}


def _lcg_stream(seed: int):
    """确定性线性同余生成器(LCG)：由 crc32(slug) 派生的种子起步，逐次产出 [0,1) 的
    伪随机浮点数流。不读 random/时间戳——跟 pick_bgm_track 的 crc32 选曲同一思路，
    保证同一 slug 重跑水印切换区间(watermark_windows)完全一致，换 slug 大概率
    不同(种子不同)。"""
    state = seed & 0x7FFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / 0x7FFFFFFF


def watermark_windows(
    slug: str, video_dur: float, wm_min_sec: float = WM_MIN_SEC, wm_max_sec: float = WM_MAX_SEC,
) -> list[tuple[float, float, str]]:
    """把 [0, video_dur] 切成随机 wm_min_sec~wm_max_sec(默认 5~10min)长的区间，位置
    从右上(TR)开始逐窗交替右下(BR)——电视台角标换位样式。切分点由 _lcg_stream(按
    slug 的 crc32 播种)生成，纯函数确定性：同 slug 两次调用返回完全一致的窗口列表，
    换 slug 大概率不同。末尾窗口裁到 video_dur(不越界，天然让窗口精确铺满整段
    时长)；片长比 wm_min_sec 还短时，第一个采样区间必然 >= wm_min_sec > video_dur，
    自然只切出一个覆盖全片的 TR 窗口。"""
    if video_dur is None or video_dur <= 0:
        return []
    rng = _lcg_stream(zlib.crc32(slug.encode("utf-8")))
    windows: list[tuple[float, float, str]] = []
    positions = (WM_TR, WM_BR)
    t = 0.0
    i = 0
    while t < video_dur:
        interval = wm_min_sec + next(rng) * (wm_max_sec - wm_min_sec)
        end = min(t + interval, video_dur)
        windows.append((round(t, 3), round(end, 3), positions[i % 2]))
        t = end
        i += 1
    return windows


def _watermark_enable_expr(windows: list[tuple[float, float, str]], position: str) -> str:
    """给定位置(TR/BR)把命中窗口拼成 ffmpeg overlay 的 enable 表达式：
    `between(t,a,b)+between(t,c,d)+...`——落在任一窗口区间内就显示，否则隐藏。"""
    return "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b, p in windows if p == position)


def build_watermark_filter(wm_idx: int, windows: list[tuple[float, float, str]]) -> str:
    """拼水印叠加的 filter_complex 片段，接在已烧好字幕的 `[vsub]` 后面产出
    `[vout]`：水印输入缩到 140px 宽、转 rgba、调到 85% 不透明度；TR 叠
    `overlay=W-w-24:24`，BR 叠 `overlay=W-w-24:H-h-130`(抬高 130px 避开烧字幕
    常驻的底部区域，不挡字)。只有一个窗口(短视频不够切换一次)时只叠对应那一路，
    省一次 split/overlay；两路都有窗口时 split 成 wmA(右上)/wmB(右下)分别叠加，
    先叠 TR 再叠 BR，最终汇到 `[vout]`。"""
    tr_expr = _watermark_enable_expr(windows, WM_TR)
    br_expr = _watermark_enable_expr(windows, WM_BR)
    scale = f"[{wm_idx}:v]scale=140:-1,format=rgba,colorchannelmixer=aa=0.85"
    if tr_expr and br_expr:
        return (
            f"{scale},split[wmA][wmB];"
            f"[vsub][wmA]overlay=W-w-24:24:enable='{tr_expr}'[vtr];"
            f"[vtr][wmB]overlay=W-w-24:H-h-130:enable='{br_expr}'[vout]"
        )
    position_expr = tr_expr or br_expr
    y = "24" if tr_expr else "H-h-130"
    return f"{scale}[wm];[vsub][wm]overlay=W-w-24:{y}:enable='{position_expr}'[vout]"


def build_assemble_cmd(
    source: Path, bg: Path, narration: Path | None, subs: Path, out_path: Path, bg_gain: float,
    bg_input_opts: list[str] | None = None, encoder: str = "libx264",
    video_dur: float | None = None, slug: str = "", config: Config | None = None,
    warn: Callable[[str], None] | None = None,
) -> list[str]:
    """第②遍：源片视频烧字幕 + 背景床(narration 非空时 sidechaincompress 被旁白 duck，
    再与旁白 amix；narration=None 时只压低背景床，跳过 duck/amix，因为没有旁白可侧链)。

    bg_input_opts：背景输入的额外 ffmpeg input 选项(如 replace 模式的
    `-stream_loop -1 -t {video_dur}` 循环铺满源片时长)，插在背景的 `-i` 之前；
    original 模式(bg=no_vocals.mp3，天然与源片等长)不需要，传 None/空列表。

    encoder：`libx264`(默认，笔记本兼容) | `nvenc`(zion 原生跑时用 GPU 编码，
    见 `p7.encoder`)——nvenc 跑失败时调用方(run())负责 warn + 回退 libx264 重试，
    这里只管拼命令，不做失败判断。

    P7 T18 水印：video_dur/slug/config 三者齐全 + `p7.video_watermark`(默认开)
    未关闭 + `p7.watermark` 指向的文件存在，才会追加水印输入 + overlay 段(接在
    subtitles 输出 `[vsub]` 后面，最终仍汇到 `[vout]`，两种 narration 变体
    的音频编排/`-map` 完全不受影响)。三者缺一(config=None——多数既有直调用例
    不传、video_dur=None、开关关、文件缺失/不可读)都直接跳过，按老逻辑拼命令；
    文件缺失时若传了 warn 会告警一次，绝不因为水印功能中止渲染。"""
    subs_esc = str(subs).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    cmd = ["ffmpeg", "-y", "-i", str(source)] + list(bg_input_opts or []) + ["-i", str(bg)]

    wm_idx = 2
    if narration is not None:
        cmd += ["-i", str(narration)]
        wm_idx = 3
        audio_fc = (
            f"[1:a]volume={bg_gain}[bg];"
            f"[bg][2:a]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=300[ducked];"
            f"[ducked][2:a]amix=inputs=2:normalize=0,alimiter=limit=0.95[aout]"
        )
    else:
        audio_fc = f"[1:a]volume={bg_gain},alimiter=limit=0.95[aout]"

    wm_path: Path | None = None
    if config is not None and config.get("p7.video_watermark", True):
        candidate = Path(config.get("p7.watermark", DEFAULT_WATERMARK))
        if candidate.exists():
            wm_path = candidate
        elif warn is not None:
            warn(f"d7 台标水印文件缺失({candidate})，跳过视频水印叠加")

    if wm_path is not None and video_dur is not None and video_dur > 0:
        cmd += ["-i", str(wm_path)]
        windows = watermark_windows(
            slug, video_dur,
            config.get("p7.wm_min_sec", WM_MIN_SEC), config.get("p7.wm_max_sec", WM_MAX_SEC),
        )
        video_fc = f"[0:v]subtitles='{subs_esc}'[vsub];" + build_watermark_filter(wm_idx, windows)
    else:
        video_fc = f"[0:v]subtitles='{subs_esc}'[vout]"

    fc = f"{audio_fc};{video_fc}"
    video_codec_args = (
        ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23"] if encoder == "nvenc"
        else ["-c:v", "libx264"]
    )
    cmd += [
        "-filter_complex", fc,
        "-map", "[vout]", "-map", "[aout]",
        *video_codec_args, "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", str(out_path),
    ]
    return cmd


def nvenc_env(config: Config, base_env: dict | None = None) -> dict:
    """nvenc 编码要 `/run/opengl-driver/lib` 里的 NixOS NVIDIA 驱动库(zion 上
    验证过)，PREPEND 进 `LD_LIBRARY_PATH`——合并进 base_env(默认 os.environ)，
    不是替换：保留调用方环境里原有的其它变量/已有的 LD_LIBRARY_PATH 路径，
    传给 subprocess.run(env=...)。base_env 参数主要供单测注入固定环境，不依赖
    真实 os.environ 的当前状态。"""
    lib_path = config.get("p7.encoder_lib_path", DEFAULT_ENCODER_LIB_PATH)
    env = dict(base_env if base_env is not None else os.environ)
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib_path}:{existing}" if existing else lib_path
    return env


def _ensure_fd_limit(need: int) -> None:
    """把本进程(及其子进程 ffmpeg)的打开文件数软上限提到 need。
    旁白第一遍合成给每段一个 `-i` 输入:1261 段(NASA+ Cosmic Dawn)撞上默认
    soft=1024 → ffmpeg "Too many open files"。zion 硬限 524288,余量充足;
    提不动(权限/平台)只告警级放行,让 ffmpeg 自己报错兜底。"""
    import resource
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft >= need:
            return
        target = need if hard == resource.RLIM_INFINITY else min(need, hard)
        if target > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError):
        pass


def run(ctx: Ctx) -> None:
    from src.util.restore import restore_from_archive

    job = ctx.job
    source = job / SOURCE
    # 重制场景:d10 归档后本地 source.mp4 被裁剪 → 自动从 NAS 找回(任务#50)
    restore_from_archive(job, SOURCE, ctx.config, lambda m: ctx.warn(f"{ctx.slug}: {m}"))
    subs = job / SUBS
    bg_gain = ctx.config.get("p7.bg_gain", 0.5)
    bgm_mode = ctx.config.get("p7.bgm", "replace")

    segments = read_json(job, NARRATION).get("segments", []) if exists(job, NARRATION) else []
    segments = enrich_speaker_fields(segments, job)   # 旧数据回填 gender,供角色分块变速
    # 重制场景续(任务#50):配音段文件也会被 d10 裁剪——挨个找回,汇总告警一条(不刷屏)
    _restored: list[str] = []
    for seg in segments:
        if seg.get("file"):
            restore_from_archive(job, seg["file"], ctx.config, _restored.append)
    narration_path: Path | None = None

    bg = job / BACKGROUND
    bg_input_opts: list[str] = []
    picked = None
    if bgm_mode == "replace":
        bgm_dir = Path(ctx.config.get("p7.bgm_dir", DEFAULT_BGM_DIR))
        # 按影片内容选曲(用户定):片名/署名关键词→情绪桶;d9c 英文版用同样输入,中英同曲
        src_meta = read_json(job, "source.json") if exists(job, "source.json") else {}
        mood = infer_bgm_mood(str(src_meta.get("title", "")), str(src_meta.get("attribution", "")))
        picked = pick_bgm_track(bgm_dir, ctx.slug, lambda m: ctx.warn(f"{ctx.slug}: {m}"), mood=mood)
    elif bgm_mode != "original":
        ctx.warn(f"{ctx.slug}: d7 未知 p7.bgm={bgm_mode!r}(应为 replace|original)，回退用 no_vocals 背景床")
    if picked is None:   # original 模式/回退时才需要 no_vocals 背景床,一并找回
        restore_from_archive(job, BACKGROUND, ctx.config, _restored.append)
    if _restored:
        ctx.warn(f"{ctx.slug}: 已从NAS归档恢复 {len(_restored)} 个媒体文件(d10 裁剪后重跑)")

    # P7 T18：水印开关+文件存在性只影响"要不要 ffprobe 源片时长"这个判断本身，
    # 不在这里 warn(避免跟 build_assemble_cmd 里同一份检查重复告警一次)——真正
    # 的开关/文件校验 + warn 落在 build_assemble_cmd 里，此处只是它的一份轻量预判。
    watermark_ready = bool(ctx.config.get("p7.video_watermark", True)) and \
        Path(ctx.config.get("p7.watermark", DEFAULT_WATERMARK)).exists()

    video_dur: float | None = None
    if segments or picked is not None or watermark_ready:
        video_dur = round(duration_sec(source), 3)

    if picked is not None:
        bg = picked["path"]
        bg_input_opts = ["-stream_loop", "-1", "-t", f"{video_dur}"]
        write_json(job, JOB_BGM, picked["entry"])

    if segments:
        cut = [s.get("id") for s, pl in zip(segments, place_segments(segments, video_dur))
               if pl["cut"] is not None]
        if cut:
            ctx.warn(f"{ctx.slug}: d7 有 {len(cut)} 个段落末句封顶仍超窗(截尾+淡出): "
                     f"seg {cut[:8]}——比例高说明 d3 字数预算没压住,查翻译")
        narration_path = job / NARRATION_TRACK
        _ensure_fd_limit(len(segments) + 256)   # 每段一个输入fd;1261段撞默认1024上限(Cosmic Dawn真跑)
        cmd1 = build_narration_cmd(job, segments, video_dur, narration_path)
        proc1 = subprocess.run(cmd1, capture_output=True, text=True)
        if proc1.returncode != 0:
            raise RuntimeError(
                f"ffmpeg 旁白轨合成失败 (exit {proc1.returncode}):\n{(proc1.stderr or '')[-2000:]}"
            )
    else:
        ctx.warn(f"{ctx.slug}: d7_assemble 无旁白段(narration.json 缺失或为空)，仅用背景+字幕合成")

    encoder = ctx.config.get("p7.encoder", "libx264")

    def wm_warn(msg: str) -> None:
        ctx.warn(f"{ctx.slug}: {msg}")

    cmd2 = build_assemble_cmd(source, bg, narration_path, subs, job / OUTPUT, bg_gain,
                               bg_input_opts, encoder=encoder,
                               video_dur=video_dur, slug=ctx.slug, config=ctx.config, warn=wm_warn)
    env2 = nvenc_env(ctx.config) if encoder == "nvenc" else None
    proc2 = subprocess.run(cmd2, capture_output=True, text=True, env=env2)
    if proc2.returncode != 0 and encoder == "nvenc":
        # nvenc 挂了(驱动/显存/环境问题)不该丢了这条片子——warn 后回退 libx264 重试一次，
        # never lose the render。
        ctx.warn(f"{ctx.slug}: d7 nvenc 编码失败(exit {proc2.returncode})，回退 libx264 重试一次: "
                 f"{(proc2.stderr or '')[-500:]}")
        cmd2 = build_assemble_cmd(source, bg, narration_path, subs, job / OUTPUT, bg_gain,
                                   bg_input_opts, encoder="libx264",
                                   video_dur=video_dur, slug=ctx.slug, config=ctx.config, warn=wm_warn)
        proc2 = subprocess.run(cmd2, capture_output=True, text=True)
    if proc2.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 合成失败 (exit {proc2.returncode}):\n{(proc2.stderr or '')[-2000:]}"
        )
