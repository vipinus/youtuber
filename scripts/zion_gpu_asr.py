"""zion 端 GPU ASR + 人声分离脚本（P7 T1）。

**必须**经 `~/gpu-py` 调用，不能直接跑 `~/asr-venv/bin/python`——否则
torch/ctranslate2 缺 LD_LIBRARY_PATH 加载不了 libstdc++/libz/libcuda（详见
memory `zion-gpu-asr-setup`）。

用法（在 zion 上）：
    ~/gpu-py zion_gpu_asr.py <audio_in> <out_dir> [whisper_model]

做的事（已在 zion 上端到端实测验通，见 P7 计划）：
  1. faster-whisper（GPU, float16, vad_filter=True）转录 <audio_in> 派生出的
     16k 单声道 wav → <out_dir>/transcript.json
     = {"duration": float, "segments": [{"start","end","text"}, ...]}
  2. demucs（`--two-stems vocals --mp3 -d cpu`；这台机器 GPU cuDNN 与 demucs
     不兼容，禁止 `-d cuda`）直接对 <audio_in> 分离 → 把产出的 no_vocals.mp3
     复制到 <out_dir>/no_vocals.mp3
  3. 打印完成标记 "ZION_DONE" 供编排侧探测成功与否

保持依赖精简：只用 stdlib + faster_whisper（直接 import）+ demucs（经
subprocess 调 `python -m demucs.separate`，不直接 import，避免拉入 demucs 的
CLI/日志副作用）。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_MODEL = "medium.en"


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


def separate(audio_in: Path, work_dir: Path) -> Path:
    """demucs CPU 人声分离，返回产出的 no_vocals.mp3 路径。

    直接对 <audio_in>（原始文件，不重采样/不重声道）分离——demucs 自己按需解码，
    不需要我们先转成"全质量 wav"。GPU（-d cuda）在这台机器上 cuDNN 不兼容，
    必须固定 `-d cpu`；输出必须 `--mp3`（默认 wav 落盘要 torchcodec，这台机器
    上不可用）。
    """
    demucs_out = work_dir / "demucs_out"
    _run([
        sys.executable, "-m", "demucs.separate",
        "--two-stems", "vocals", "--mp3", "-d", "cpu",
        "-o", str(demucs_out), str(audio_in),
    ])
    stem = audio_in.stem
    produced = demucs_out / "htdemucs" / stem / "no_vocals.mp3"
    if not produced.exists():
        raise FileNotFoundError(f"demucs 未产出预期文件: {produced}")
    return produced


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
    (out_dir / "transcript.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=1), encoding="utf-8")

    no_vocals = separate(audio_in, work_dir)
    shutil.copyfile(no_vocals, out_dir / "no_vocals.mp3")

    print(f"transcript.json: {len(transcript['segments'])} 段, "
          f"时长 {transcript['duration']:.0f}s")
    print("ZION_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
