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

P7 T14（最终迁移）追加了 `LocalGPU`：管线本身也部署到 zion 上原生跑（不再是
笔记本 ssh 过去），这种模式下 zion 既是控制端又是 GPU 端，SSHRemoteGPU 的
scp/ssh 编排变成纯浪费（同机器多此一举地走网络栈）——`LocalGPU` 直接用
subprocess 在本机（此时就是 zion）经 `~/gpu-py` 跑 scripts/zion_gpu_asr.py /
zion_sdxl.py，脚本读写的 out_dir/out_path 就是最终产物落点，不需要再 scp 拉回。
`remote.provider: local` 选它（见 config-zion.yaml）；笔记本模式（`ssh`）完全
不受影响，两套并存。

P7 T17 追加了 `clone_synthesize`：F5-TTS one-shot 跨语种声音克隆配音——每个
说话人的中文旁白用他/她自己的原声（demucs 分离出的 vocals.mp3 切片）当参考
音色。zion 侧配方（已在 zion 上真实验通，用户认可克隆质量）：新 venv
`~/clone-venv` 装了 f5-tts，同 gpu-py 的坑——必须经 `~/clone-py` wrapper 调用
（先设好 LD_LIBRARY_PATH 再 exec clone-venv python），不能直接跑裸 venv
python。zion 侧脚本见 scripts/zion_clone_tts.py（本模块负责部署它 +
编排/subprocess 调用，跟 transcribe_separate/generate_image 手法一致）。
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
_ZION_CLONE_SCRIPT_NAME = "zion_clone_tts.py"


def _run_checked(
    cmd: list[str], timeout_sec: int, timeout_msg: str, fail_msg_prefix: str,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    """跑一条子进程命令：超时/非零 returncode 都抛清晰 RuntimeError（含
    stdout/stderr），不吞异常、不无限挂起。ssh/scp（SSHRemoteGPU）和本机
    gpu-py 编排（LocalGPU）共用同一套超时+报错语义，只是各自的错误文案
    前缀不同（调用方传 timeout_msg/fail_msg_prefix）。"""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_sec, env=env,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(timeout_msg) from e
    if result.returncode != 0:
        raise RuntimeError(
            f"{fail_msg_prefix}\n"
            f"[returncode={result.returncode}] stdout={result.stdout!r} "
            f"stderr={result.stderr!r}"
        )
    return result


class RemoteGPU(Protocol):
    def transcribe_separate(self, audio: Path, out_dir: Path) -> dict: ...
    def generate_image(self, prompt: str, out_path: Path) -> Path: ...
    def clone_synthesize(self, zh_script: Path, vocals: Path, out_dir: Path) -> Path: ...


class FakeRemoteGPU:
    """离线确定性：写一份罐头 transcript.json + 两段真正可解码的静音
    no_vocals/vocals 音频到 out_dir，不 ssh/不触网。

    音频必须是真正可解码的文件（而非占位字节）——参照 src/media.FakeMedia 的
    教训：下游若拿到非法数据会在 ffmpeg/ffprobe 处反复重试或崩溃，使
    remote=fake 无法离线跑通整条流水线。这里用 ffmpeg 的 lavfi anullsrc 现场
    生成 1 秒静音 mp3。P7 T17：vocals.mp3 是下游 clone_synthesize 的参考音色
    来源（真实 zion 侧是 demucs 干声轨），罐头版一样要落地、可解码。
    """

    def transcribe_separate(self, audio: Path, out_dir: Path) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)

        transcript = {
            "duration": 5.0,
            "segments": [
                # gender 覆盖两条路径（P7 T9 性别匹配配音离线可测）：一男一女。
                {"start": 0.0, "end": 2.0, "text": "This is a fake transcript segment.",
                 "gender": "male"},
                {"start": 2.0, "end": 5.0, "text": "It exists purely for offline tests.",
                 "gender": "female"},
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
        vocals_path = out_dir / "vocals.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
             "-t", "1", str(vocals_path)],
            check=True, capture_output=True,
        )
        return {
            "transcript": transcript_path,
            "no_vocals": no_vocals_path,
            "vocals": vocals_path,
        }

    def clone_synthesize(self, zh_script: Path, vocals: Path, out_dir: Path) -> Path:
        """离线确定性：不跑 F5-TTS，只按 zh_script 里每个非空 zh 段写一份真正
        可解码的 1 秒静音 wav `n{id:03d}.wav`——供 d4_tts 的 clone 分支离线测试
        「narration 文件存在 + 可 ffprobe」这条路径，不 ssh/不触网。"""
        out_dir.mkdir(parents=True, exist_ok=True)
        import json
        data = json.loads(Path(zh_script).read_text(encoding="utf-8"))
        for seg in data.get("segments", []):
            zh = (seg.get("zh") or "").strip()
            if not zh:
                continue
            seg_id = int(seg.get("id", 0))
            out_path = out_dir / f"n{seg_id:03d}.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                 "-t", "1", str(out_path)],
                check=True, capture_output=True,
            )
        return out_dir

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
    transcript.json + no_vocals.mp3 + vocals.mp3 回本地 out_dir。

    构造阶段不做任何网络调用——只在 transcribe_separate 真正被调用时才
    ssh/scp，保证「import/构造」在离线单测里绝对安全。
    """

    def __init__(
        self,
        host: str = "fabric@zion",
        workdir: str = "~/p7-work",
        wrapper: str = "~/gpu-py",
        clone_wrapper: str = "~/clone-py",
        whisper_model: str = "medium.en",
        script_name: str = _ZION_SCRIPT_NAME,
        timeout_sec: int = 1200,
        clone_timeout_sec: int = 7200,
    ):
        self.host = host
        self.workdir = workdir
        self.wrapper = wrapper
        self.clone_wrapper = clone_wrapper     # P7 T17：F5-TTS 克隆配音走独立 wrapper(~/clone-py)
        self.whisper_model = whisper_model
        self.script_name = script_name
        self.timeout_sec = timeout_sec         # 卡死的 zion(掉网/驱动挂)不能无限挂住管线
        self.clone_timeout_sec = clone_timeout_sec  # 克隆配音逐段推理耗时更长(200段约15-25min)，独立超时上限

    # -- 底层 ssh/scp 封装：失败时报清晰错误（含命令 + stderr），不吞异常 --

    def _ssh(self, remote_cmd: str, timeout_sec: int | None = None) -> str:
        timeout = self.timeout_sec if timeout_sec is None else timeout_sec
        result = _run_checked(
            ["ssh", "-o", "BatchMode=yes", self.host, remote_cmd], timeout,
            timeout_msg=(
                f"ssh 超时 {timeout}s (host={self.host}): {remote_cmd!r} "
                f"—— zion 可能掉线/卡死，管线中止而非无限挂起"
            ),
            fail_msg_prefix=f"ssh 失败 (host={self.host}): {remote_cmd!r}",
        )
        return result.stdout

    def _scp(self, src: str, dst: str, timeout_sec: int | None = None) -> None:
        timeout = self.timeout_sec if timeout_sec is None else timeout_sec
        _run_checked(
            ["scp", "-o", "BatchMode=yes", src, dst], timeout,
            timeout_msg=f"scp 超时 {timeout}s: {src} -> {dst}",
            fail_msg_prefix=f"scp 失败: {src} -> {dst}",
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
        vocals_path = out_dir / "vocals.mp3"
        self._scp(f"{self.host}:{remote_out}/transcript.json", str(transcript_path))
        self._scp(f"{self.host}:{remote_out}/no_vocals.mp3", str(no_vocals_path))
        self._scp(f"{self.host}:{remote_out}/vocals.mp3", str(vocals_path))
        return {
            "transcript": transcript_path,
            "no_vocals": no_vocals_path,
            "vocals": vocals_path,
        }

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

    def clone_synthesize(self, zh_script: Path, vocals: Path, out_dir: Path) -> Path:
        """部署 scripts/zion_clone_tts.py、scp zh_script.json + vocals.mp3 上去、
        经 `~/clone-py` 跑 F5-TTS 逐段克隆配音、把产出的 nNNN.wav 整目录 tar 打包
        拉回本地 out_dir。

        用 tar 而非逐文件 scp：产物文件名(id)是跑完才知道的动态集合(部分段可能
        因单段失败而缺失)，逐文件 scp 得先 ssh 查一遍远端目录列文件名，tar 打包
        一次拉回更简单可靠——同 P7 计划里"keep simple"的取舍。"""
        out_dir.mkdir(parents=True, exist_ok=True)

        remote_script = self._deploy_script(_ZION_CLONE_SCRIPT_NAME)
        remote_zh_script = f"{self.workdir}/{Path(zh_script).name}"
        remote_vocals = f"{self.workdir}/clone_{Path(vocals).name}"
        remote_out = f"{self.workdir}/clone_out_{Path(zh_script).stem}"
        remote_tar = f"{remote_out}.tar"

        self._scp(str(zh_script), f"{self.host}:{remote_zh_script}")
        self._scp(str(vocals), f"{self.host}:{remote_vocals}")
        self._ssh(f"mkdir -p {remote_out}")
        self._ssh(
            f"{self.clone_wrapper} {remote_script} {remote_zh_script} {remote_vocals} "
            f"{remote_out}",
            timeout_sec=self.clone_timeout_sec,
        )
        self._ssh(f"tar -C {remote_out} -cf {remote_tar} .", timeout_sec=self.clone_timeout_sec)

        local_tar = out_dir / "_clone_out.tar"
        self._scp(f"{self.host}:{remote_tar}", str(local_tar), timeout_sec=self.clone_timeout_sec)
        import tarfile
        with tarfile.open(local_tar) as tf:
            tf.extractall(out_dir)
        local_tar.unlink()
        return out_dir


class LocalGPU:
    """P7 T14：管线本身就跑在 zion 上时用它——同一台机器既是控制端又是 GPU
    端，SSHRemoteGPU 那套 scp 上传/ssh 触发/scp 拉回完全是多余的网络往返。
    LocalGPU 直接用 subprocess 在本机经 `~/gpu-py` 跑
    scripts/zion_gpu_asr.py / zion_sdxl.py：脚本的 out_dir/out_path 参数就是
    最终产物落点（同一份文件系统），跑完直接读，不需要再搬一次。

    协议（transcribe_separate/generate_image/clone_synthesize 的输入输出契约）
    跟 SSHRemoteGPU 完全一致，因此上游 d2_gpu/d4_tts/d8_meta 不用感知走的是
    local 还是 ssh。
    """

    def __init__(
        self,
        wrapper: str = "~/gpu-py",
        clone_wrapper: str = "~/clone-py",
        whisper_model: str = "medium.en",
        timeout_sec: int = 1200,
        clone_timeout_sec: int = 7200,
        repo_root: Path | None = None,
    ):
        self.wrapper = str(Path(wrapper).expanduser())
        self.clone_wrapper = str(Path(clone_wrapper).expanduser())  # P7 T17：clone-venv 的 wrapper
        self.whisper_model = whisper_model
        self.timeout_sec = timeout_sec        # 同 SSHRemoteGPU：GPU 任务卡死不能无限挂住管线
        self.clone_timeout_sec = clone_timeout_sec  # 克隆配音逐段推理耗时更长(200段约15-25min)，独立超时上限
        self.repo_root = Path(repo_root) if repo_root is not None else _REPO_ROOT

    def transcribe_separate(self, audio: Path, out_dir: Path) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        script = self.repo_root / "scripts" / _ZION_SCRIPT_NAME
        cmd = [self.wrapper, str(script), str(audio), str(out_dir), self.whisper_model]
        _run_checked(
            cmd, self.timeout_sec,
            timeout_msg=(
                f"local gpu-py 超时 {self.timeout_sec}s: {cmd!r} "
                f"—— GPU 任务卡死，管线中止而非无限挂起"
            ),
            fail_msg_prefix=f"local gpu-py 失败: {cmd!r}",
        )
        return {
            "transcript": out_dir / "transcript.json",
            "no_vocals": out_dir / "no_vocals.mp3",
            "vocals": out_dir / "vocals.mp3",
        }

    def generate_image(self, prompt: str, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        script = self.repo_root / "scripts" / _ZION_SDXL_SCRIPT_NAME
        cmd = [self.wrapper, str(script), prompt, str(out_path)]
        _run_checked(
            cmd, self.timeout_sec,
            timeout_msg=(
                f"local gpu-py 超时 {self.timeout_sec}s: {cmd!r} "
                f"—— GPU 任务卡死，管线中止而非无限挂起"
            ),
            fail_msg_prefix=f"local gpu-py 失败: {cmd!r}",
        )
        return out_path

    def clone_synthesize(self, zh_script: Path, vocals: Path, out_dir: Path) -> Path:
        """本机经 `~/clone-py` 跑 scripts/zion_clone_tts.py：脚本直接把
        n{id:03d}.wav 写进 out_dir（同一份文件系统，不需要再搬一次）。"""
        out_dir.mkdir(parents=True, exist_ok=True)
        script = self.repo_root / "scripts" / _ZION_CLONE_SCRIPT_NAME
        cmd = [self.clone_wrapper, str(script), str(zh_script), str(vocals), str(out_dir)]
        _run_checked(
            cmd, self.clone_timeout_sec,
            timeout_msg=(
                f"local clone-py 超时 {self.clone_timeout_sec}s: {cmd!r} "
                f"—— GPU 任务卡死，管线中止而非无限挂起"
            ),
            fail_msg_prefix=f"local clone-py 失败: {cmd!r}",
        )
        return out_dir


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
            clone_wrapper=config.get("remote.clone_wrapper", "~/clone-py"),
            clone_timeout_sec=config.get("remote.clone_timeout_sec", 7200),
        )
    if provider == "local":
        return LocalGPU(
            wrapper=config.get("remote.wrapper", "~/gpu-py"),
            whisper_model=config.get("remote.whisper_model", "medium.en"),
            timeout_sec=config.get("remote.timeout_sec", 1200),
            clone_wrapper=config.get("remote.clone_wrapper", "~/clone-py"),
            clone_timeout_sec=config.get("remote.clone_timeout_sec", 7200),
        )
    raise ValueError(f"未知 remote provider: {provider}")
