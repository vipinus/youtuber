"""zion 端 GPU ASR + 人声分离脚本（P7 T1，T9 追加性别匹配配音的基频检测）。

**必须**经 `~/gpu-py` 调用，不能直接跑 `~/asr-venv/bin/python`——否则
torch/ctranslate2 缺 LD_LIBRARY_PATH 加载不了 libstdc++/libz/libcuda（详见
memory `zion-gpu-asr-setup`）。

用法（在 zion 上）：
    ~/gpu-py zion_gpu_asr.py <audio_in> <out_dir> [whisper_model]

做的事（已在 zion 上端到端实测验通，见 P7 计划）：
  1. faster-whisper（GPU, float16, vad_filter=True）转录 <audio_in> 派生出的
     16k 单声道 wav → <out_dir>/transcript.json
     = {"duration": float, "segments": [{"start","end","text","gender"}, ...]}
  2. demucs（`--two-stems vocals --mp3 -d cpu`；这台机器 GPU cuDNN 与 demucs
     不兼容，禁止 `-d cuda`）直接对 <audio_in> 分离 → 把产出的 no_vocals.mp3
     复制到 <out_dir>/no_vocals.mp3（vocals.mp3 留在 demucs 的临时输出目录，
     只用来算基频，不落 out_dir）
  3. P7 T9：用 demucs 分出的 vocals（干净人声，无背景音乐/音效干扰）算每段
     中位基频（torchaudio.functional.detect_pitch_frequency），按阈值分类
     "male"/"female"/None（不确定就 null，绝不瞎猜——下游 d4_tts 拿它按段
     选中文 TTS 男声/女声，猜错比不猜更糟），写回每段的 "gender" 字段
  4. 打印完成标记 "ZION_DONE" 供编排侧探测成功与否

保持依赖精简：只用 stdlib + faster_whisper / torch / torchaudio（直接
import，但只在用到的函数体内 import——本地(zion 之外)单测/import 这个文件
时不需要装 torch，只有真正调用 transcribe()/detect_genders() 才会触发）+
demucs（经 subprocess 调 `python -m demucs.separate`，不直接 import，避免
拉入 demucs 的 CLI/日志副作用）。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

DEFAULT_MODEL = "medium.en"

# -- P7 T9 性别检测参数（已与用户对齐，见计划文档）--------------------------
MIN_SEGMENT_SEC = 0.6      # 短于此不测（太短基频统计不可靠）
PITCH_FREQ_LOW = 60.0      # detect_pitch_frequency 的搜索下限(Hz)
PITCH_FREQ_HIGH = 400.0    # 搜索上限(Hz)——覆盖成年男女声基频范围
MALE_MAX_HZ = 150.0        # 中位基频 < 此值 -> "male"
FEMALE_MIN_HZ = 165.0      # 中位基频 > 此值 -> "female"
# 阈值用真片校准(ISS Benefits doc):男主旁白中位 115Hz、90分位 139Hz;灰区里
# 163-170Hz 的段比男旁白高 ~50%,几乎必是女受访者,原 175 门槛把她们全漏成
# null→落男声(恰是用户投诉)。165 是文献常用男女分界;150-165 仍留 null 缓冲。
# [145, 175] 之间（含边界）视为模糊地带 -> None，宁可不猜。
MIN_VOICED_FRAMES = 3      # 可信帧太少(基本没测到人声) -> None


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _make_16k_mono(audio_in: Path, work_dir: Path) -> Path:
    """派生 faster-whisper 要的 16k 单声道 wav（ffmpeg 对任意可解码输入通用）。"""
    wav16k = work_dir / "whisper16k.wav"
    _run(["ffmpeg", "-y", "-i", str(audio_in), "-vn", "-ac", "1", "-ar", "16000", str(wav16k)])
    return wav16k


def transcribe(wav16k: Path, model_name: str = DEFAULT_MODEL) -> dict:
    """faster-whisper GPU 转录，返回 {"duration": float, "segments": [...]}。"""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cuda", compute_type="float16")
    segments, info = model.transcribe(str(wav16k), vad_filter=True)
    segs = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments if s.text.strip()
    ]
    return {"duration": round(info.duration, 2), "segments": segs}


def separate(audio_in: Path, work_dir: Path) -> tuple[Path, Path]:
    """demucs CPU 人声分离，返回 (vocals.mp3, no_vocals.mp3) 路径二元组。

    直接对 <audio_in>（原始文件，不重采样/不重声道）分离——demucs 自己按需解码，
    不需要我们先转成"全质量 wav"。GPU（-d cuda）在这台机器上 cuDNN 不兼容，
    必须固定 `-d cpu`；输出必须 `--mp3`（默认 wav 落盘要 torchcodec，这台机器
    上不可用）。vocals.mp3（干净人声）供 P7 T9 算基频分性别；no_vocals.mp3
    是背景床，两个都要产出才算成功。
    """
    demucs_out = work_dir / "demucs_out"
    _run([
        sys.executable, "-m", "demucs.separate",
        "--two-stems", "vocals", "--mp3", "-d", "cpu",
        "-o", str(demucs_out), str(audio_in),
    ])
    stem = audio_in.stem
    stem_dir = demucs_out / "htdemucs" / stem
    vocals = stem_dir / "vocals.mp3"
    no_vocals = stem_dir / "no_vocals.mp3"
    missing = [p for p in (vocals, no_vocals) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"demucs 未产出预期文件: {missing}")
    return vocals, no_vocals


def classify_f0(median_f0: float | None) -> str | None:
    """纯函数、零依赖（不碰 torch）——按一段人声的中位基频分类说话人性别。

    阈值取自成年男女声基频统计的常见分界（已与用户对齐）：
    < 145Hz -> "male"，> 175Hz -> "female"，[145,175] 模糊地带或输入 None
    （静音/时长太短/可信帧不足）一律返回 None——宁可不猜，也不能给下游一个
    错的性别标签（配错声比不配更糟，这正是本任务要修的用户投诉）。
    """
    if median_f0 is None:
        return None
    if median_f0 < MALE_MAX_HZ:
        return "male"
    if median_f0 > FEMALE_MIN_HZ:
        return "female"
    return None


def _segment_median_f0(pitch_frames, rms_frames, frame_time: float,
                        start: float, end: float):
    """按时间切片拿该段的基频帧，过滤掉不可信帧后返回中位数（无 torch 时不可
    单测，只在 zion 上跑）：
      - 段时长 < MIN_SEGMENT_SEC -> None（太短统计不可靠）
      - 帧本身在 [PITCH_FREQ_LOW, PITCH_FREQ_HIGH] 开区间内才算——
        detect_pitch_frequency 对静音/噪声常年卡在搜索边界上，卡边界的是
        检测失败而非真实基频
      - 帧能量(RMS) 明显低于该段峰值(< 10%) 的算非人声/静音帧，一并丢弃
      - 剩余可信帧 < MIN_VOICED_FRAMES -> None（几乎没测到人声，不够下结论）
    """
    if end - start < MIN_SEGMENT_SEC:
        return None
    if frame_time <= 0:
        return None
    start_idx = max(0, int(start / frame_time))
    end_idx = max(start_idx + 1, int(round(end / frame_time)))
    seg_pitch = pitch_frames[start_idx:end_idx]
    seg_rms = rms_frames[start_idx:end_idx]
    if seg_pitch.numel() == 0:
        return None

    rms_gate = seg_rms.max() * 0.1 if seg_rms.numel() else 0.0
    mask = (
        (seg_pitch > PITCH_FREQ_LOW) & (seg_pitch < PITCH_FREQ_HIGH)
        & (seg_rms > rms_gate)
    )
    voiced = seg_pitch[mask]
    if voiced.numel() < MIN_VOICED_FRAMES:
        return None
    return float(voiced.median())


def detect_genders(vocals_path: Path, segments: list[dict]) -> list[str | None]:
    """对 vocals 干声轨整轨算一次基频（比逐段重跑 pitch 快得多，见计划要求
    "vectorize/slice efficiently"），再按每段 start/end 切片分类。返回与
    segments 等长的 "male"/"female"/None 列表。

    只在这里 import torch/torchaudio——本地(zion 之外)静态检查/导入这个脚本
    文件不需要装 torch，只有真在 zion 上跑到这一步才会触发 import。
    """
    import torch
    import torchaudio

    # 不用 torchaudio.load：这台 NixOS 机上它没有任何可用后端(soundfile 未装、
    # ffmpeg 后端库不可见、torchcodec .so 加载失败——实测 RuntimeError
    # "Couldn't find appropriate backend")。改用 ffmpeg 子进程直接解码成
    # 16k 单声道裸 PCM(s16le)，零后端依赖；ffmpeg 二进制在 zion 验证可用。
    sr = 16000
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(vocals_path),
         "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True, check=True,
    )
    waveform = (torch.frombuffer(bytearray(proc.stdout), dtype=torch.int16)
                .float().div_(32768.0).unsqueeze(0))

    pitch = torchaudio.functional.detect_pitch_frequency(
        waveform, sr, freq_low=int(PITCH_FREQ_LOW), freq_high=int(PITCH_FREQ_HIGH),
    )[0]                                            # [time_frames]（单声道，取第 0 个通道）

    n_frames = int(pitch.shape[-1])
    n_samples = int(waveform.shape[-1])
    duration = n_samples / sr if sr else 0.0
    frame_time = duration / n_frames if n_frames else 0.0

    # 逐帧粗略 RMS（窗口与 pitch 帧对齐），用能量过滤掉非人声/静音帧。
    hop = max(1, int(round(frame_time * sr))) if frame_time else n_samples
    rms_vals = []
    mono = waveform[0]
    for i in range(n_frames):
        lo = i * hop
        hi = min(n_samples, lo + hop)
        window = mono[lo:hi]
        rms_vals.append(float(window.pow(2).mean().sqrt()) if window.numel() else 0.0)
    rms = torch.tensor(rms_vals)

    return [
        classify_f0(_segment_median_f0(
            pitch, rms, frame_time, seg.get("start", 0.0), seg.get("end", 0.0)))
        for seg in segments
    ]


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("用法: zion_gpu_asr.py <audio_in> <out_dir> [whisper_model]", file=sys.stderr)
        return 2

    audio_in = Path(argv[1]).resolve()
    out_dir = Path(argv[2]).resolve()
    model_name = argv[3] if len(argv) > 3 else DEFAULT_MODEL

    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    wav16k = _make_16k_mono(audio_in, work_dir)
    transcript = transcribe(wav16k, model_name)

    vocals, no_vocals = separate(audio_in, work_dir)
    shutil.copyfile(no_vocals, out_dir / "no_vocals.mp3")

    # P7 T9：性别检测是锦上添花，不是主产物——万一 torch/torchaudio 或某段
    # 音频触发意外异常，绝不能让整条转录+去人声的核心产物报废（同 d8 SDXL
    # 失败回退纯色底的"挂不中止 run"思路）。失败就全段落 None，照常继续。
    try:
        genders = detect_genders(vocals, transcript["segments"])
    except Exception:
        traceback.print_exc()
        print("性别检测失败，本次全部段落 gender=null（不影响转录/去人声主产物）",
              file=sys.stderr)
        genders = [None] * len(transcript["segments"])
    for seg, gender in zip(transcript["segments"], genders):
        seg["gender"] = gender

    (out_dir / "transcript.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=1), encoding="utf-8")

    n_male = sum(1 for g in genders if g == "male")
    n_female = sum(1 for g in genders if g == "female")
    print(f"transcript.json: {len(transcript['segments'])} 段, "
          f"时长 {transcript['duration']:.0f}s, "
          f"gender: male={n_male} female={n_female} unknown="
          f"{len(genders) - n_male - n_female}")
    print("ZION_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
