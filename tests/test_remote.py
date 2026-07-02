import ast
import inspect
import json
import subprocess

import pytest
from src.config import Config
from src.remote import FakeRemoteGPU, SSHRemoteGPU, get_remote


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
    }
    assert out_dir.is_dir()

    # 命令序列: mkdir workdir, scp 脚本上去, scp 音频上去, ssh 跑脚本, scp 拉回 x2
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
