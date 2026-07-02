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

    data = read_json(ctx.job, OUTPUT)
    segments = data.get("segments", [])
    if not segments:
        # 0 段 = 源片没有可转录的解说(纯素材/音乐/静音)。绝不能继续——那会产出没有
        # 中文旁白/字幕的废片甚至上传。拦住，让上层换一部有解说的源片。
        raise StageBlocked(
            f"{ctx.slug}: d2_gpu 转录得到 0 段——源片无解说旁白(纯画面/音乐)，"
            f"不适合整片翻译。删掉 source.mp4 用 --force 重跑 d1_source 换片。"
        )
