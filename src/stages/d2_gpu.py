"""P7 T3：阶段②——把 source.mp4 的音轨转录成英文字幕(时间轴对齐) +
分离出去人声的背景轨(no_vocals.mp3)和干声轨(vocals.mp3)，为下游翻译/配音/
混音铺路。

本地只做一件事：从 source.mp4 提取一份"普通"音频(不预先降到 16k/单声道——
demucs 分离要吃相对完整的音质，16k 单声道降采样是 zion 侧 whisper 转录
专用的派生步骤，见 scripts/zion_gpu_asr.py._make_16k_mono)。真正的
GPU 转录 + CPU demucs 分离都编排给 src/remote.py 的 RemoteGPU
(fake/ssh)完成。

P7 T17：vocals.mp3(demucs 干声轨)现在也落成规范产物——d4_tts 的克隆配音模式
(tts.provider=="clone")要用它按段切出每个说话人的原声，当 F5-TTS 的参考音色。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from src.context import Ctx
from src.errors import StageBlocked
from src.remote import get_remote
from src.util.jobs import read_json

OUTPUT = "transcript.json"
INPUT = "source.mp4"
NO_VOCALS_OUTPUT = "no_vocals.mp3"
VOCALS_OUTPUT = "vocals.mp3"      # P7 T17：demucs 干声轨，下游 d4_tts clone 模式的克隆参考音色来源
AUDIO_WORK_NAME = "source_audio.wav"


def _extract_audio(source: Path, out: Path) -> None:
    """source.mp4 -> 普通 wav(保留原声道/采样率，只去视频轨)，供 remote 侧
    同时喂给 whisper(会自行再派生 16k 单声道)与 demucs(要相对完整音质)。"""
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(source), "-vn", "-acodec", "pcm_s16le", str(out)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"d2_gpu: ffmpeg 音频提取失败 (exit {proc.returncode}, source={source}):\n"
            f"{(proc.stderr or '')[-2000:]}"
        )


def _land(produced: Path, canonical: Path) -> None:
    """remote 产出可能已经就是 canonical 路径(fake/ssh 都写 out_dir=ctx.job)；
    若不是(未来 provider 落到别处/临时目录)，原子搬到规范文件名，并清理陈旧
    的同名旧产物(上次失败运行残留)。"""
    if produced.resolve() == canonical.resolve():
        return
    produced.replace(canonical)


def run(ctx: Ctx) -> None:
    source = ctx.job / INPUT
    audio_path = ctx.job / AUDIO_WORK_NAME
    _extract_audio(source, audio_path)

    remote = get_remote(ctx.config)
    result = remote.transcribe_separate(audio_path, ctx.job)

    transcript_path = ctx.job / OUTPUT
    no_vocals_path = ctx.job / NO_VOCALS_OUTPUT
    vocals_path = ctx.job / VOCALS_OUTPUT
    _land(Path(result["transcript"]), transcript_path)
    _land(Path(result["no_vocals"]), no_vocals_path)
    _land(Path(result["vocals"]), vocals_path)

    # 平台字幕优先(2026-07-04):youtube_gov 片源若 d1 落了 source_subs.json3
    # (YouTube 字幕,人工或自动),用它替换 whisper 文本/时间轴——对讲机/无线电
    # 对话 whisper 漏抓严重而平台 ASR 是主场(实测 264条 vs 39段)。whisper 转录
    # 仍保留:它带基频性别,按时间重叠映射给字幕段;解析失败回退 whisper,不新增故障。
    subs_path = ctx.job / "source_subs.json3"
    if subs_path.exists():
        try:
            from src.util.jobs import write_json
            from src.util.ytsubs import genders_by_overlap, parse_json3_segments
            subs_segs = parse_json3_segments(subs_path)
            whisper_n = len(read_json(ctx.job, OUTPUT).get("segments", []))
            whisper = read_json(ctx.job, OUTPUT)
            if subs_segs and whisper_n < 3:
                # 无解说+有字幕(2026-07-04 用户定)= 字幕版产品:字幕多为屏幕文字/
                # 说明卡(ISS POV 纯音乐片),不配音、保留原声、只烧中文字幕直接发布。
                # mode 标记贯穿 d3→d4(跳过配音)→d5(原字幕时间轴)→d7(原声完整保留)。
                write_json(ctx.job, OUTPUT, {"duration": whisper.get("duration"),
                                             "segments": subs_segs, "mode": "subs_only"})
                ctx.warn(f"{ctx.slug}: d2 无解说+有字幕({len(subs_segs)}段)→字幕版模式"
                         "(不配音,原声保留,只烧中文字幕)")
            elif subs_segs:
                genders_by_overlap(subs_segs, whisper.get("segments", []))
                write_json(ctx.job, OUTPUT, {"duration": whisper.get("duration"),
                                             "segments": subs_segs})
                ctx.warn(f"{ctx.slug}: d2 使用平台字幕转录({len(subs_segs)}段,"
                         f"whisper {len(whisper.get('segments', []))}段仅供性别映射)")
        except Exception as e:  # noqa: BLE001 — 字幕解析失败回退 whisper
            ctx.warn(f"{ctx.slug}: d2 平台字幕解析失败({type(e).__name__}),沿用 whisper 转录")

    data = read_json(ctx.job, OUTPUT)
    segments = data.get("segments", [])
    if not segments:
        # 0 段 = 源片没有可转录的解说且无字幕(纯素材/音乐/静音)。绝不能继续——那会
        # 产出没有中文旁白/字幕的废片甚至上传。拦截前先把该片记入登记表(status=
        # rejected),否则重跑 d1 会再次选中同一部(白化野牛片真跑踩坑 2026-07-04)。
        try:
            from src.util.jobs import read_json as _rj
            from src.util.registry import append_published
            ident = _rj(ctx.job, "source.json").get("identifier")
            if ident:
                append_published(ctx.root, {"identifier": ident,
                                            "status": "rejected_no_narration"})
                ctx.warn(f"{ctx.slug}: 已登记拒片 {ident}(无解说无字幕),重选不会再命中")
        except Exception:  # noqa: BLE001 — 登记失败不挡拦截主逻辑
            pass
        raise StageBlocked(
            f"{ctx.slug}: d2_gpu 转录得到 0 段——源片无解说旁白(纯画面/音乐)，"
            f"不适合整片翻译。删掉 source.mp4 用 --force 重跑 d1_source 换片。"
        )
