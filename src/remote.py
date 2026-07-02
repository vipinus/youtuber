"""P7 T1：编排远程 GPU 主机 zion 跑 faster-whisper 转录 + demucs 人声分离。

zion 侧配方（已在 zion 上真实端到端验证过，见 memory `zion-gpu-asr-setup` /
计划 docs/superpowers/plans/2026-07-01-explore-ufo-pipeline-phase7-doc-translation.md）：
  - 必须经 `~/gpu-py` 调用 python（该 wrapper 先设好 LD_LIBRARY_PATH 再 exec
    `~/asr-venv/bin/python`），不能直接跑 venv 里的 python，否则
    torch/ctranslate2 加载不了 libstdc++/libz/libcuda。
  - faster-whisper 用 GPU（`device="cuda", compute_type="float16"`）。
  - demucs 用 CPU（`-d cpu`，`--mp3`）——demucs 与这台机器的 GPU cuDNN 不兼容。
  - zion 侧脚本见 scripts/zion_gpu_asr.py（本模块负责部署它 + 编排 scp/ssh）。

P7 T8 追加了 `generate_image`（AI 缩略图背景）：编排手法跟上面完全一样——部署
scripts/zion_sdxl.py 到 zion、经 `~/gpu-py` 跑 Stable Diffusion XL、scp 一张
1280x720 PNG 拉回本地。zion 已装好 diffusers/torch 2.6.0+cu124 且模型已缓存
（见计划文档 SDXL 配方：`enable_model_cpu_offload`+`enable_vae_slicing` 撑住
8GB 显存，28 步 ~12s）。prompt 里必须带 "no text"——中文标题/水印由本地
d8_meta 用 Pillow 后期叠加，不指望 SDXL 出中文字。
"""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Protocol

from src.config import Config

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ZION_SCRIPT_NAME = "zion_gpu_asr.py"
_ZION_SDXL_SCRIPT_NAME = "zion_sdxl.py"


class RemoteGPU(Protocol):
    def transcribe_separate(self, audio: Path, out_dir: Path) -> dict: ...
    def generate_image(self, prompt: str, out_path: Path) -> Path: ...


class FakeRemoteGPU:
    """离线确定性：写一份罐头 transcript.json + 一段真正可解码的静音 no_vocals
    音频到 out_dir，不 ssh/不触网。

    音频必须是真正可解码的文件（而非占位字节）——参照 src/media.FakeMedia 的
    教训：下游若拿到非法数据会在 ffmpeg/ffprobe 处反复重试或崩溃，使
    remote=fake 无法离线跑通整条流水线。这里用 ffmpeg 的 lavfi anullsrc 现场
    生成 1 秒静音 mp3。
    """

    def transcribe_separate(self, audio: Path, out_dir: Path) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)

        transcript = {
            "duration": 5.0,
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "This is a fake transcript segment."},
                {"start": 2.0, "end": 5.0, "text": "It exists purely for offline tests."},
            ],
        }
        transcript_path = out_dir / "transcript.json"
        import json
        transcript_path.write_text(
            json.dumps(transcript, ensure_ascii=False, indent=1), encoding="utf-8")

        no_vocals_path = out_dir / "no_vocals.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
             "-t", "1", str(no_vocals_path)],
            check=True, capture_output=True,
        )
        return {"transcript": transcript_path, "no_vocals": no_vocals_path}

    def generate_image(self, prompt: str, out_path: Path) -> Path:
        """离线确定性：现场用 Pillow 画一张真正可解码的 1280x720 PNG（纵向渐变），
        不 ssh/不触网、不依赖 SDXL。下游 d8_meta 的合成/缩略图逻辑吃真图即可，
        不关心内容是不是 AI 生成的。"""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image, ImageDraw

        w, h = 1280, 720
        img = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(img)
        top, bottom = (12, 16, 40), (60, 30, 90)          # 深蓝 -> 深紫，按行画线(快)，不逐像素(慢)
        for y in range(h):
            t = y / (h - 1)
            row = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
            draw.line([(0, y), (w, y)], fill=row)
        img.save(out_path, "PNG")
        return out_path


class SSHRemoteGPU:
    """SSH/scp 编排 zion：推音频上去、经 `~/gpu-py` 跑 zion_gpu_asr.py、拉
    transcript.json + no_vocals.mp3 回本地 out_dir。

    构造阶段不做任何网络调用——只在 transcribe_separate 真正被调用时才
    ssh/scp，保证「import/构造」在离线单测里绝对安全。
    """

    def __init__(
        self,
        host: str = "fabric@zion",
        workdir: str = "~/p7-work",
        wrapper: str = "~/gpu-py",
        whisper_model: str = "medium.en",
        script_name: str = _ZION_SCRIPT_NAME,
        timeout_sec: int = 1200,
    ):
        self.host = host
        self.workdir = workdir
        self.wrapper = wrapper
        self.whisper_model = whisper_model
        self.script_name = script_name
        self.timeout_sec = timeout_sec         # 卡死的 zion(掉网/驱动挂)不能无限挂住管线

    # -- 底层 ssh/scp 封装：失败时报清晰错误（含命令 + stderr），不吞异常 --

    def _ssh(self, remote_cmd: str) -> str:
        try:
            result = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", self.host, remote_cmd],
                capture_output=True, text=True, timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"ssh 超时 {self.timeout_sec}s (host={self.host}): {remote_cmd!r} "
                f"—— zion 可能掉线/卡死，管线中止而非无限挂起"
            ) from e
        if result.returncode != 0:
            raise RuntimeError(
                f"ssh 失败 (host={self.host}): {remote_cmd!r}\n"
                f"[returncode={result.returncode}] stdout={result.stdout!r} "
                f"stderr={result.stderr!r}"
            )
        return result.stdout

    def _scp(self, src: str, dst: str) -> None:
        try:
            result = subprocess.run(
                ["scp", "-o", "BatchMode=yes", src, dst],
                capture_output=True, text=True, timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"scp 超时 {self.timeout_sec}s: {src} -> {dst}") from e
        if result.returncode != 0:
            raise RuntimeError(
                f"scp 失败: {src} -> {dst}\n"
                f"[returncode={result.returncode}] stdout={result.stdout!r} "
                f"stderr={result.stderr!r}"
            )

    def _deploy_script(self, script_name: str | None = None) -> str:
        """确保脚本存在于 zion 的 workdir（每次都重新推送，保证跑的是本地最新
        版本）；返回 zion 上的脚本绝对路径（写法上用 workdir 内相对路径即可，
        ssh 里 shell 会展开 ~）。不传 script_name 时用 self.script_name（转录
        脚本），generate_image 会显式传 _ZION_SDXL_SCRIPT_NAME。"""
        name = script_name or self.script_name
        remote_script = f"{self.workdir}/{name}"
        self._ssh(f"mkdir -p {self.workdir}")
        local_script = _REPO_ROOT / "scripts" / name
        self._scp(str(local_script), f"{self.host}:{remote_script}")
        return remote_script

    def transcribe_separate(self, audio: Path, out_dir: Path) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)

        remote_script = self._deploy_script()
        remote_audio = f"{self.workdir}/{audio.name}"
        remote_out = f"{self.workdir}/out_{audio.stem}"

        self._scp(str(audio), f"{self.host}:{remote_audio}")
        self._ssh(
            f"{self.wrapper} {remote_script} {remote_audio} {remote_out} "
            f"{self.whisper_model}"
        )

        transcript_path = out_dir / "transcript.json"
        no_vocals_path = out_dir / "no_vocals.mp3"
        self._scp(f"{self.host}:{remote_out}/transcript.json", str(transcript_path))
        self._scp(f"{self.host}:{remote_out}/no_vocals.mp3", str(no_vocals_path))
        return {"transcript": transcript_path, "no_vocals": no_vocals_path}

    def generate_image(self, prompt: str, out_path: Path) -> Path:
        """部署 scripts/zion_sdxl.py、经 wrapper 跑 SDXL、把生成的 PNG scp 回
        out_path。prompt 是普通字符串（非文件），不需要先 scp 上去——直接当
        命令行参数传给远端脚本，用 shlex.quote 防止里面的空格/引号把远端 shell
        命令行拆散。"""
        out_path.parent.mkdir(parents=True, exist_ok=True)

        remote_script = self._deploy_script(_ZION_SDXL_SCRIPT_NAME)
        remote_out = f"{self.workdir}/{out_path.name}"

        self._ssh(f"{self.wrapper} {remote_script} {shlex.quote(prompt)} {remote_out}")
        self._scp(f"{self.host}:{remote_out}", str(out_path))
        return out_path


def get_remote(config: Config) -> RemoteGPU:
    provider = config.get("remote.provider", "ssh")
    if provider == "fake":
        return FakeRemoteGPU()
    if provider == "ssh":
        return SSHRemoteGPU(
            host=config.get("remote.host", "fabric@zion"),
            workdir=config.get("remote.workdir", "~/p7-work"),
            whisper_model=config.get("remote.whisper_model", "medium.en"),
            timeout_sec=config.get("remote.timeout_sec", 1200),
        )
    raise ValueError(f"未知 remote provider: {provider}")
