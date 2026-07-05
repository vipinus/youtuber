import ast
import inspect
import json
import subprocess
from pathlib import Path

import pytest
from src.config import Config
from src.remote import FakeRemoteGPU, LocalGPU, SSHRemoteGPU, get_remote


# ---------------------------------------------------------------------------
# FakeRemoteGPU — 离线确定性，产物必须是真正可解析/可解码的文件
# ---------------------------------------------------------------------------

def test_fake_remote_gpu_writes_valid_transcript_and_audio(tmp_path):
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"\x00")  # FakeRemoteGPU 不读输入内容，只按约定产出罐头结果
    out_dir = tmp_path / "out"

    result = FakeRemoteGPU().transcribe_separate(audio, out_dir)

    assert result["transcript"] == out_dir / "transcript.json"
    assert result["no_vocals"] == out_dir / "no_vocals.mp3"

    data = json.loads(result["transcript"].read_text(encoding="utf-8"))
    assert isinstance(data["duration"], (int, float)) and data["duration"] > 0
    assert len(data["segments"]) >= 2
    for seg in data["segments"]:
        assert {"start", "end", "text"} <= seg.keys()
        assert isinstance(seg["text"], str) and seg["text"]

    assert result["no_vocals"].exists()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(result["no_vocals"])],
        capture_output=True, text=True, check=True,
    )
    assert "audio" in probe.stdout            # 必须是真正可解码的音频，不是占位字节


def test_fake_remote_gpu_transcript_segments_have_gender_covering_both_paths(tmp_path):
    """P7 T9：罐头 transcript 必须带 gender 字段，且一男一女都出现——下游
    d3/d4 的性别匹配配音测试才能离线覆盖两条路径。"""
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"\x00")
    out_dir = tmp_path / "out"

    result = FakeRemoteGPU().transcribe_separate(audio, out_dir)
    data = json.loads(result["transcript"].read_text(encoding="utf-8"))

    genders = [seg["gender"] for seg in data["segments"]]
    assert "male" in genders
    assert "female" in genders


def test_fake_remote_gpu_creates_out_dir_if_missing(tmp_path):
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"\x00")
    out_dir = tmp_path / "nested" / "out"     # 尚不存在的多级目录
    FakeRemoteGPU().transcribe_separate(audio, out_dir)
    assert out_dir.is_dir()


# ---------------------------------------------------------------------------
# P7 T17：FakeRemoteGPU 也要落一份真正可解码的 vocals.mp3(demucs 干声轨罐头版)
# ---------------------------------------------------------------------------

def test_fake_remote_gpu_transcribe_separate_lands_valid_vocals_mp3(tmp_path):
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"\x00")
    out_dir = tmp_path / "out"

    result = FakeRemoteGPU().transcribe_separate(audio, out_dir)

    assert result["vocals"] == out_dir / "vocals.mp3"
    assert result["vocals"].exists()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(result["vocals"])],
        capture_output=True, text=True, check=True,
    )
    assert "audio" in probe.stdout


# ---------------------------------------------------------------------------
# P7 T17：FakeRemoteGPU.clone_synthesize — 离线确定性，产物必须是真正可解码的 wav
# ---------------------------------------------------------------------------

def test_fake_remote_gpu_clone_synthesize_writes_valid_wav_per_nonempty_segment(tmp_path):
    zh_script = tmp_path / "zh_script.json"
    zh_script.write_text(json.dumps({"segments": [
        {"id": 0, "start": 0.0, "end": 1.0, "en": "Hi", "zh": "你好"},
        {"id": 1, "start": 1.0, "end": 2.0, "en": "", "zh": ""},          # 空 zh -> 不产出
        {"id": 5, "start": 2.0, "end": 3.0, "en": "World", "zh": "世界"},
    ]}), encoding="utf-8")
    vocals = tmp_path / "vocals.mp3"
    vocals.write_bytes(b"\x00")
    out_dir = tmp_path / "narration"

    result = FakeRemoteGPU().clone_synthesize(zh_script, vocals, out_dir)

    assert result == out_dir
    assert (out_dir / "n000.wav").exists()
    assert (out_dir / "n005.wav").exists()
    assert not (out_dir / "n001.wav").exists()      # 空 zh 段不产出

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(out_dir / "n000.wav")],
        capture_output=True, text=True, check=True,
    )
    assert "audio" in probe.stdout


def test_fake_remote_gpu_clone_synthesize_creates_out_dir_if_missing(tmp_path):
    zh_script = tmp_path / "zh_script.json"
    zh_script.write_text(json.dumps({"segments": []}), encoding="utf-8")
    vocals = tmp_path / "vocals.mp3"
    vocals.write_bytes(b"\x00")
    out_dir = tmp_path / "nested" / "narration"

    FakeRemoteGPU().clone_synthesize(zh_script, vocals, out_dir)
    assert out_dir.is_dir()


# ---------------------------------------------------------------------------
# FakeRemoteGPU.generate_image — 离线确定性，产物必须是真正可解码的 PNG
# ---------------------------------------------------------------------------

def test_fake_remote_gpu_generate_image_writes_valid_png(tmp_path):
    from PIL import Image

    out_path = tmp_path / "thumb_bg.png"
    result = FakeRemoteGPU().generate_image("a dramatic space scene, no text", out_path)

    assert result == out_path
    assert out_path.exists() and out_path.stat().st_size > 0
    img = Image.open(out_path)
    assert img.size == (1280, 720)
    assert img.format == "PNG"


def test_fake_remote_gpu_generate_image_creates_parent_dir(tmp_path):
    out_path = tmp_path / "nested" / "thumb_bg.png"
    FakeRemoteGPU().generate_image("prompt", out_path)
    assert out_path.exists()


# ---------------------------------------------------------------------------
# get_remote 工厂
# ---------------------------------------------------------------------------

def test_get_remote_fake():
    assert isinstance(get_remote(Config({"remote": {"provider": "fake"}})), FakeRemoteGPU)


def test_get_remote_ssh_default_host():
    remote = get_remote(Config({"remote": {"provider": "ssh"}}))
    assert isinstance(remote, SSHRemoteGPU)
    assert remote.host == "fabric@zion"
    assert remote.whisper_model == "medium.en"


def test_get_remote_default_provider_is_ssh():
    """未配置 remote 块时也不能崩——默认 provider=ssh（构造不触网）。"""
    assert isinstance(get_remote(Config({})), SSHRemoteGPU)


def test_get_remote_ssh_custom_config():
    cfg = Config({"remote": {
        "provider": "ssh", "host": "fabric@otherhost",
        "workdir": "~/custom-work", "whisper_model": "large-v3",
    }})
    remote = get_remote(cfg)
    assert remote.host == "fabric@otherhost"
    assert remote.workdir == "~/custom-work"
    assert remote.whisper_model == "large-v3"


def test_get_remote_unknown_raises():
    with pytest.raises(ValueError):
        get_remote(Config({"remote": {"provider": "nope"}}))


# ---------------------------------------------------------------------------
# P7 T17：get_remote 工厂 — clone_wrapper/clone_timeout_sec 默认值 + 可覆盖
# ---------------------------------------------------------------------------

def test_get_remote_ssh_default_clone_wrapper_and_timeout():
    remote = get_remote(Config({"remote": {"provider": "ssh"}}))
    assert remote.clone_wrapper == "~/clone-py"
    assert remote.clone_timeout_sec == 7200


def test_get_remote_local_default_clone_wrapper_and_timeout():
    remote = get_remote(Config({"remote": {"provider": "local"}}))
    assert remote.clone_wrapper == str(Path("~/clone-py").expanduser())
    assert remote.clone_timeout_sec == 7200


def test_get_remote_ssh_custom_clone_wrapper_and_timeout():
    cfg = Config({"remote": {
        "provider": "ssh", "clone_wrapper": "~/other-clone-py", "clone_timeout_sec": 99,
    }})
    remote = get_remote(cfg)
    assert remote.clone_wrapper == "~/other-clone-py"
    assert remote.clone_timeout_sec == 99


def test_get_remote_local_custom_clone_wrapper_and_timeout():
    cfg = Config({"remote": {
        "provider": "local", "clone_wrapper": "~/other-clone-py", "clone_timeout_sec": 99,
    }})
    remote = get_remote(cfg)
    assert remote.clone_wrapper == str(Path("~/other-clone-py").expanduser())
    assert remote.clone_timeout_sec == 99


# ---------------------------------------------------------------------------
# 离线保证：import / 构造阶段绝不触网，不依赖额外的 ssh 库
# ---------------------------------------------------------------------------

def test_remote_module_has_no_top_level_network_imports():
    """离线单测不能触网：模块顶层禁止 import paramiko/fabric/requests 之类。"""
    import src.remote as remote_mod

    banned = {"paramiko", "fabric", "requests"}
    tree = ast.parse(inspect.getsource(remote_mod))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] not in banned for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in banned


def test_constructing_ssh_remote_gpu_does_not_touch_network(monkeypatch):
    """仅构造 SSHRemoteGPU（不调用 transcribe_separate）不应调用 subprocess。"""
    def _boom(*a, **k):
        raise AssertionError("不应在构造阶段调用 subprocess（ssh/scp）")

    monkeypatch.setattr(subprocess, "run", _boom)
    remote = SSHRemoteGPU(host="fabric@zion", workdir="~/p7-work")
    assert remote.host == "fabric@zion"


# ---------------------------------------------------------------------------
# SSHRemoteGPU — 用 monkeypatch 假的 subprocess.run 验证命令编排/错误处理，
# 全程不发起真实 ssh/scp。
# ---------------------------------------------------------------------------

class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_ssh_remote_gpu_happy_path_issues_expected_commands(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    audio = tmp_path / "narration.wav"
    audio.write_bytes(b"\x00")
    out_dir = tmp_path / "out"

    remote = SSHRemoteGPU(host="fabric@zion", workdir="~/p7-work",
                          wrapper="~/gpu-py", whisper_model="medium.en")
    result = remote.transcribe_separate(audio, out_dir)

    assert result == {
        "transcript": out_dir / "transcript.json",
        "no_vocals": out_dir / "no_vocals.mp3",
        "vocals": out_dir / "vocals.mp3",
    }
    assert out_dir.is_dir()

    # 命令序列: mkdir workdir, scp 脚本上去, scp 音频上去, ssh 跑脚本, scp 拉回 x3(P7 T17 追加 vocals.mp3)
    assert calls[0][:2] == ["ssh", "-o"]
    assert "mkdir -p ~/p7-work" in calls[0][-1]
    assert calls[1][0] == "scp" and calls[1][-1] == "fabric@zion:~/p7-work/zion_gpu_asr.py"
    assert calls[2][0] == "scp" and calls[2][-1] == "fabric@zion:~/p7-work/narration.wav"
    run_cmd = calls[3][-1]
    assert "~/gpu-py" in run_cmd and "zion_gpu_asr.py" in run_cmd
    assert "~/p7-work/narration.wav" in run_cmd and "medium.en" in run_cmd
    assert calls[4][0] == "scp" and calls[4][1:3] == ["-o", "BatchMode=yes"]
    assert calls[4][3] == "fabric@zion:~/p7-work/out_narration/transcript.json"
    assert calls[5][3] == "fabric@zion:~/p7-work/out_narration/no_vocals.mp3"
    assert calls[6][0] == "scp" and calls[6][3] == "fabric@zion:~/p7-work/out_narration/vocals.mp3"


def test_ssh_remote_gpu_generate_image_happy_path_issues_expected_commands(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    out_path = tmp_path / "thumb_bg.png"
    remote = SSHRemoteGPU(host="fabric@zion", workdir="~/p7-work", wrapper="~/gpu-py")
    result = remote.generate_image("a dramatic scene, no text", out_path)

    assert result == out_path
    assert out_path.parent.is_dir()

    # 命令序列: mkdir workdir, scp zion_sdxl.py 上去, ssh 跑脚本(prompt 被 shlex.quote), scp 拉回
    assert calls[0][:2] == ["ssh", "-o"]
    assert "mkdir -p ~/p7-work" in calls[0][-1]
    assert calls[1][0] == "scp" and calls[1][-1] == "fabric@zion:~/p7-work/zion_sdxl.py"
    run_cmd = calls[2][-1]
    assert "~/gpu-py" in run_cmd and "zion_sdxl.py" in run_cmd
    assert "a dramatic scene, no text" in run_cmd     # 简单 prompt 不含特殊字符，quote 后原样可见
    assert "~/p7-work/thumb_bg.png" in run_cmd
    assert calls[3][0] == "scp" and calls[3][1:3] == ["-o", "BatchMode=yes"]
    assert calls[3][3] == "fabric@zion:~/p7-work/thumb_bg.png"
    assert calls[3][4] == str(out_path)


def test_ssh_remote_gpu_generate_image_quotes_prompt_with_special_chars(tmp_path, monkeypatch):
    """prompt 里带空格/引号时必须被 shlex.quote 安全转义，否则远端 shell 会把
    它拆成多个参数或注入命令。"""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    remote = SSHRemoteGPU()
    remote.generate_image("a scene with \"quotes\" and $(danger); rm -rf", tmp_path / "out.png")

    run_cmd = calls[2][-1]
    import shlex
    assert shlex.quote("a scene with \"quotes\" and $(danger); rm -rf") in run_cmd


def test_ssh_remote_gpu_generate_image_raises_clear_error_on_ssh_failure(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[0] == "ssh" and "gpu-py" in cmd[-1]:
            return _FakeCompleted(returncode=1, stdout="", stderr="CUDA OOM")
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    remote = SSHRemoteGPU()
    with pytest.raises(RuntimeError, match="ssh 失败"):
        remote.generate_image("prompt", tmp_path / "out.png")


def test_ssh_remote_gpu_raises_clear_error_on_ssh_failure(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[0] == "ssh":
            return _FakeCompleted(returncode=255, stdout="", stderr="Connection refused")
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    audio = tmp_path / "narration.wav"
    audio.write_bytes(b"\x00")
    remote = SSHRemoteGPU()
    with pytest.raises(RuntimeError, match="ssh 失败"):
        remote.transcribe_separate(audio, tmp_path / "out")


def test_ssh_remote_gpu_raises_clear_error_on_scp_failure(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[0] == "scp":
            return _FakeCompleted(returncode=1, stdout="", stderr="No such file or directory")
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    audio = tmp_path / "narration.wav"
    audio.write_bytes(b"\x00")
    remote = SSHRemoteGPU()
    with pytest.raises(RuntimeError, match="scp 失败"):
        remote.transcribe_separate(audio, tmp_path / "out")


def test_ssh_timeout_raises_clear_error(monkeypatch):
    """zion 卡死时 ssh 超时应中止并报清晰错误，而不是无限挂起。"""
    import subprocess as sp
    from src.remote import SSHRemoteGPU

    def _boom(*a, **k):
        raise sp.TimeoutExpired(cmd="ssh", timeout=k.get("timeout", 1))
    monkeypatch.setattr(sp, "run", _boom)
    r = SSHRemoteGPU(timeout_sec=5)
    try:
        r._ssh("echo hi")
        assert False, "应抛超时"
    except RuntimeError as e:
        assert "超时" in str(e)


def test_get_remote_passes_timeout(monkeypatch):
    from src.config import Config
    from src.remote import get_remote, SSHRemoteGPU
    r = get_remote(Config({"remote": {"provider": "ssh", "timeout_sec": 77}}))
    assert isinstance(r, SSHRemoteGPU) and r.timeout_sec == 77


# ---------------------------------------------------------------------------
# P7 T17 — SSHRemoteGPU.clone_synthesize：部署 zion_clone_tts.py、scp
# zh_script.json + vocals.mp3 上去、经 clone_wrapper 跑克隆、tar 打包拉回
# out_dir。全程 monkeypatch 假 subprocess.run，绝不真实 ssh/scp。
# ---------------------------------------------------------------------------

def test_ssh_remote_gpu_clone_synthesize_command_shape(tmp_path, monkeypatch):
    import io
    import tarfile

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "scp" and str(cmd[-2]).endswith(".tar"):
            # 模拟 zion 侧打包好的克隆产物 tar 被 scp 拉回本地：现场写一个真实 tar，
            # 内含一份 n000.wav，供 clone_synthesize 的 tarfile.extractall 消费。
            local_dst = Path(cmd[-1])
            local_dst.parent.mkdir(parents=True, exist_ok=True)
            payload = b"fake wav bytes"
            with tarfile.open(local_dst, "w") as tf:
                info = tarfile.TarInfo(name="n000.wav")
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    zh_script = tmp_path / "zh_script.json"
    zh_script.write_text("{}", encoding="utf-8")
    vocals = tmp_path / "vocals.mp3"
    vocals.write_bytes(b"\x00")
    out_dir = tmp_path / "narration"

    remote = SSHRemoteGPU(host="fabric@zion", workdir="~/p7-work", clone_wrapper="~/clone-py")
    result = remote.clone_synthesize(zh_script, vocals, out_dir)

    assert result == out_dir
    assert (out_dir / "n000.wav").exists()
    assert (out_dir / "n000.wav").read_bytes() == b"fake wav bytes"
    assert not (out_dir / "_clone_out.tar").exists()          # 临时 tar 用完即删

    # 命令序列: mkdir workdir(部署脚本), scp 脚本, scp zh_script, scp vocals,
    # mkdir remote_out, ssh 跑克隆, ssh tar 打包, scp tar 拉回
    assert calls[0][:2] == ["ssh", "-o"] and "mkdir -p ~/p7-work" in calls[0][-1]
    assert calls[1][0] == "scp" and calls[1][-1] == "fabric@zion:~/p7-work/zion_clone_tts.py"
    assert calls[2][0] == "scp" and calls[2][-1].endswith("zh_script.json")
    assert calls[3][0] == "scp" and "vocals.mp3" in calls[3][-1]
    assert calls[4][:2] == ["ssh", "-o"] and "mkdir -p" in calls[4][-1]
    run_cmd = calls[5][-1]
    assert "~/clone-py" in run_cmd and "zion_clone_tts.py" in run_cmd
    tar_cmd = calls[6][-1]
    assert tar_cmd.startswith("tar -C")
    assert calls[7][0] == "scp" and str(calls[7][-2]).endswith(".tar")


def test_ssh_remote_gpu_clone_synthesize_uses_clone_timeout(tmp_path, monkeypatch):
    """克隆推理跑一整部片子耗时更长——ssh/tar/scp 调用要用 clone_timeout_sec，
    不能沿用转录用的短 timeout_sec。"""
    import io
    import tarfile

    seen_timeouts = []

    def fake_run(cmd, timeout=None, **kwargs):
        seen_timeouts.append(timeout)
        if cmd[0] == "scp" and str(cmd[-2]).endswith(".tar"):
            local_dst = Path(cmd[-1])
            local_dst.parent.mkdir(parents=True, exist_ok=True)
            payload = b"x"
            with tarfile.open(local_dst, "w") as tf:
                info = tarfile.TarInfo(name="n000.wav")
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    remote = SSHRemoteGPU(timeout_sec=5, clone_timeout_sec=999)
    remote.clone_synthesize(tmp_path / "zh.json", tmp_path / "vocals.mp3", tmp_path / "out")

    # 部署脚本(mkdir + scp)/zh_script/vocals 上传/远端 mkdir 仍走普通 timeout_sec
    # (都是轻量操作)；跑克隆/tar 打包/tar 拉回三条重活走 clone_timeout_sec。
    assert seen_timeouts[:5] == [5, 5, 5, 5, 5]
    assert seen_timeouts[5:] == [999, 999, 999]


def test_ssh_remote_gpu_clone_synthesize_raises_clear_error_on_ssh_failure(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[0] == "ssh" and "clone-py" in cmd[-1]:
            return _FakeCompleted(returncode=1, stdout="", stderr="CUDA OOM")
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    remote = SSHRemoteGPU()
    with pytest.raises(RuntimeError, match="ssh 失败"):
        remote.clone_synthesize(tmp_path / "zh.json", tmp_path / "vocals.mp3", tmp_path / "out")


# ---------------------------------------------------------------------------
# LocalGPU (P7 T14) — 管线原生跑在 zion 上：直接 subprocess 经 ~/gpu-py 跑
# scripts/zion_gpu_asr.py / zion_sdxl.py，不 scp/不 ssh。全程用 monkeypatch
# 假的 subprocess.run 验证命令编排，绝不触发真实 gpu-py。
# ---------------------------------------------------------------------------

class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_local_gpu_transcribe_separate_command_shape(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    audio = tmp_path / "narration.wav"
    audio.write_bytes(b"\x00")
    out_dir = tmp_path / "out"
    repo = tmp_path / "repo"

    remote = LocalGPU(wrapper="~/gpu-py", whisper_model="medium.en",
                      timeout_sec=99, repo_root=repo)
    result = remote.transcribe_separate(audio, out_dir)

    assert result == {
        "transcript": out_dir / "transcript.json",
        "no_vocals": out_dir / "no_vocals.mp3",
        "vocals": out_dir / "vocals.mp3",
    }
    assert out_dir.is_dir()
    assert len(calls) == 1                                # 一条命令，不 scp/不 ssh
    cmd, kwargs = calls[0]
    assert cmd[0] == str(Path("~/gpu-py").expanduser())
    assert cmd[1] == str(repo / "scripts" / "zion_gpu_asr.py")
    assert cmd[2] == str(audio)
    assert cmd[3] == str(out_dir)
    assert cmd[4] == "medium.en"
    assert kwargs["timeout"] == 99


def test_local_gpu_transcribe_separate_creates_out_dir_if_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: _FakeCompleted(returncode=0))
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"\x00")
    out_dir = tmp_path / "nested" / "out"
    LocalGPU(repo_root=tmp_path / "repo").transcribe_separate(audio, out_dir)
    assert out_dir.is_dir()


def test_local_gpu_generate_image_command_shape(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    out_path = tmp_path / "thumb_bg.png"
    repo = tmp_path / "repo"
    remote = LocalGPU(wrapper="~/gpu-py", timeout_sec=88, repo_root=repo)
    result = remote.generate_image("a dramatic scene, no text", out_path)

    assert result == out_path
    assert out_path.parent.is_dir()
    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert cmd[0] == str(Path("~/gpu-py").expanduser())
    assert cmd[1] == str(repo / "scripts" / "zion_sdxl.py")
    assert cmd[2] == "a dramatic scene, no text"           # 无需 shlex.quote（不经 shell）
    assert cmd[3] == str(out_path)
    assert kwargs["timeout"] == 88


def test_local_gpu_wrapper_expanduser_applied(tmp_path):
    remote = LocalGPU(wrapper="~/custom-gpu-py", repo_root=tmp_path)
    assert remote.wrapper == str(Path("~/custom-gpu-py").expanduser())
    assert "~" not in remote.wrapper


def test_local_gpu_default_repo_root_is_running_repo():
    from src.remote import _REPO_ROOT
    remote = LocalGPU()
    assert remote.repo_root == _REPO_ROOT


def test_local_gpu_raises_clear_error_on_nonzero_returncode(tmp_path, monkeypatch):
    def fake_run(cmd, **kw):
        return _FakeCompleted(returncode=1, stdout="", stderr="CUDA OOM")

    monkeypatch.setattr(subprocess, "run", fake_run)
    remote = LocalGPU(repo_root=tmp_path)
    with pytest.raises(RuntimeError, match="local gpu-py 失败"):
        remote.transcribe_separate(tmp_path / "in.wav", tmp_path / "out")


def test_local_gpu_timeout_raises_clear_error(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="gpu-py", timeout=k.get("timeout", 1))

    monkeypatch.setattr(subprocess, "run", _boom)
    remote = LocalGPU(repo_root=tmp_path, timeout_sec=5)
    with pytest.raises(RuntimeError, match="超时"):
        remote.transcribe_separate(tmp_path / "in.wav", tmp_path / "out")


def test_local_gpu_generate_image_timeout_raises_clear_error(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="gpu-py", timeout=k.get("timeout", 1))

    monkeypatch.setattr(subprocess, "run", _boom)
    remote = LocalGPU(repo_root=tmp_path, timeout_sec=5)
    with pytest.raises(RuntimeError, match="超时"):
        remote.generate_image("prompt", tmp_path / "out.png")


def test_local_gpu_does_not_scp_or_ssh(tmp_path, monkeypatch):
    """头号回归风险：LocalGPU 绝不能发起 ssh/scp（那是 SSHRemoteGPU 的事）。"""
    def fake_run(cmd, **kw):
        assert cmd[0] not in ("ssh", "scp"), f"LocalGPU 不该发起 {cmd[0]}"
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    remote = LocalGPU(repo_root=tmp_path)
    remote.transcribe_separate(tmp_path / "in.wav", tmp_path / "out")
    remote.generate_image("prompt", tmp_path / "bg.png")


# ---------------------------------------------------------------------------
# P7 T17 — LocalGPU.clone_synthesize：本机经 ~/clone-py 跑 zion_clone_tts.py，
# 不 scp/不 ssh，脚本直接把 wav 写进 out_dir。
# ---------------------------------------------------------------------------

def test_local_gpu_clone_synthesize_command_shape(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    zh_script = tmp_path / "zh_script.json"
    zh_script.write_text("{}", encoding="utf-8")
    vocals = tmp_path / "vocals.mp3"
    vocals.write_bytes(b"\x00")
    out_dir = tmp_path / "narration"
    repo = tmp_path / "repo"

    remote = LocalGPU(clone_wrapper="~/clone-py", clone_timeout_sec=321, repo_root=repo)
    result = remote.clone_synthesize(zh_script, vocals, out_dir)

    assert result == out_dir
    assert out_dir.is_dir()
    assert len(calls) == 1                                # 一条命令，不 scp/不 ssh
    cmd, kwargs = calls[0]
    assert cmd[0] == str(Path("~/clone-py").expanduser())
    assert cmd[1] == str(repo / "scripts" / "zion_clone_tts.py")
    assert cmd[2] == str(zh_script)
    assert cmd[3] == str(vocals)
    assert cmd[4] == str(out_dir)
    assert kwargs["timeout"] == 321


def test_local_gpu_clone_synthesize_default_wrapper_and_timeout(tmp_path):
    remote = LocalGPU(repo_root=tmp_path)
    assert remote.clone_wrapper == str(Path("~/clone-py").expanduser())
    assert remote.clone_timeout_sec == 7200


def test_local_gpu_clone_synthesize_raises_clear_error_on_nonzero_returncode(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **kw: _FakeCompleted(returncode=1, stdout="", stderr="OOM"),
    )
    remote = LocalGPU(repo_root=tmp_path)
    with pytest.raises(RuntimeError, match="local clone-py 失败"):
        remote.clone_synthesize(tmp_path / "zh.json", tmp_path / "vocals.mp3", tmp_path / "out")


def test_local_gpu_clone_synthesize_does_not_scp_or_ssh(tmp_path, monkeypatch):
    def fake_run(cmd, **kw):
        assert cmd[0] not in ("ssh", "scp"), f"LocalGPU 不该发起 {cmd[0]}"
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    remote = LocalGPU(repo_root=tmp_path)
    remote.clone_synthesize(tmp_path / "zh.json", tmp_path / "vocals.mp3", tmp_path / "out")


# ---------------------------------------------------------------------------
# get_remote 工厂 —— local -> LocalGPU
# ---------------------------------------------------------------------------

def test_get_remote_local():
    remote = get_remote(Config({"remote": {"provider": "local"}}))
    assert isinstance(remote, LocalGPU)
    assert remote.wrapper == str(Path("~/gpu-py").expanduser())
    assert remote.whisper_model == "medium.en"


def test_get_remote_local_custom_wrapper_and_model():
    cfg = Config({"remote": {
        "provider": "local", "wrapper": "~/other-gpu-py",
        "whisper_model": "large-v3", "timeout_sec": 42,
    }})
    remote = get_remote(cfg)
    assert isinstance(remote, LocalGPU)
    assert remote.wrapper == str(Path("~/other-gpu-py").expanduser())
    assert remote.whisper_model == "large-v3"
    assert remote.timeout_sec == 42


def test_get_remote_local_dispatches_cosyvoice_script(tmp_path):
    """tts.provider=cosyvoice(2026-07-05 A/B胜出)→ LocalGPU 换 cosy wrapper+脚本。"""
    from src.config import Config
    from src.remote import LocalGPU, get_remote

    r = get_remote(Config({"remote": {"provider": "local"},
                           "tts": {"provider": "cosyvoice"}}))
    assert isinstance(r, LocalGPU)
    assert r.clone_script == "zion_cosy_tts.py"
    assert r.clone_wrapper.endswith("cosy-py")

    r2 = get_remote(Config({"remote": {"provider": "local"},
                            "tts": {"provider": "clone"}}))
    assert r2.clone_script == "zion_clone_tts.py"
    assert r2.clone_wrapper.endswith("clone-py")
