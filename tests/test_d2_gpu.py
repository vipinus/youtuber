from __future__ import annotations

import ast
import inspect
import json
import subprocess
from pathlib import Path

import pytest

from src.config import Config
from src.context import Ctx
from src.llm import FakeLLM
from src.util.jobs import read_json, write_json


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ctx(tmp_path, cfg_data=None) -> Ctx:
    cfg = Config(cfg_data or {"remote": {"provider": "fake"}})
    return Ctx(slug="doc1", config=cfg, root=tmp_path, llm=FakeLLM())


def _write_fake_source_mp4(path: Path) -> None:
    """离线生成一份真正可解码的 mp4（video+audio 双流，1 秒），不触网——
    只用 ffmpeg 的 lavfi 合成源，供 d2_gpu 的 ffmpeg 音频提取步骤消费。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
         "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
         "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(path)],
        check=True, capture_output=True,
    )


# ---------------------------------------------------------------------------
# happy path — provider=fake，全程不触网/不 ssh
# ---------------------------------------------------------------------------

def test_happy_path_lands_canonical_outputs(tmp_path):
    import src.stages.d2_gpu as d2

    ctx = _ctx(tmp_path)
    _write_fake_source_mp4(ctx.job / d2.INPUT)

    d2.run(ctx)

    transcript_path = ctx.job / d2.OUTPUT
    no_vocals_path = ctx.job / d2.NO_VOCALS_OUTPUT
    vocals_path = ctx.job / d2.VOCALS_OUTPUT
    assert transcript_path.exists()
    assert no_vocals_path.exists()
    assert vocals_path.exists()          # P7 T17：克隆配音参考音色来源

    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert isinstance(data.get("segments"), list)
    assert len(data["segments"]) >= 1
    for seg in data["segments"]:
        assert {"start", "end", "text"} <= seg.keys()

    for audio_path in (no_vocals_path, vocals_path):
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", str(audio_path)],
            capture_output=True, text=True, check=True,
        )
        assert "audio" in probe.stdout

    assert ctx.warnings == []          # FakeRemoteGPU 罐头有 2 段，不应告警


def test_audio_extraction_feeds_a_real_decodable_wav_to_remote(tmp_path, monkeypatch):
    """验证音频提取 -> remote 调用的接线：remote 收到的 audio 路径必须是从
    source.mp4 派生出的、真正可解码的音频文件（而非原始 mp4/占位字节）。"""
    import src.stages.d2_gpu as d2

    ctx = _ctx(tmp_path)
    _write_fake_source_mp4(ctx.job / d2.INPUT)

    seen = {}
    real_get_remote = d2.get_remote

    class _Spy:
        def transcribe_separate(self, audio: Path, out_dir: Path) -> dict:
            seen["audio"] = audio
            seen["out_dir"] = out_dir
            return real_get_remote(ctx.config).transcribe_separate(audio, out_dir)

    monkeypatch.setattr(d2, "get_remote", lambda config: _Spy())
    d2.run(ctx)

    audio_path = seen["audio"]
    assert audio_path.exists()
    assert audio_path.suffix == ".wav"
    assert audio_path != ctx.job / d2.INPUT
    assert seen["out_dir"] == ctx.job

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(audio_path)],
        capture_output=True, text=True, check=True,
    )
    assert "audio" in probe.stdout


# ---------------------------------------------------------------------------
# 0 段 transcript -> 告警但不崩
# ---------------------------------------------------------------------------

class _ZeroSegRemote:
    """模拟源片无可检测语音：transcript.json 里 segments 为空数组，
    no_vocals.mp3/vocals.mp3 依然是真正可解码音频。不 ssh/不触网。"""

    def transcribe_separate(self, audio: Path, out_dir: Path) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = out_dir / "transcript.json"
        transcript_path.write_text(
            json.dumps({"duration": 3.0, "segments": []}), encoding="utf-8")
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
        return {"transcript": transcript_path, "no_vocals": no_vocals_path, "vocals": vocals_path}


def test_zero_segments_raises_stageblocked(tmp_path, monkeypatch):
    """0 段 = 源片无解说(纯素材)，必须 StageBlocked 拦住，绝不静默出没中文的废片。"""
    import src.stages.d2_gpu as d2
    from src.errors import StageBlocked
    import pytest

    ctx = _ctx(tmp_path)
    _write_fake_source_mp4(ctx.job / d2.INPUT)
    monkeypatch.setattr(d2, "get_remote", lambda config: _ZeroSegRemote())

    with pytest.raises(StageBlocked):
        d2.run(ctx)


class _DifferentPathRemote:
    """模拟 remote 产出落在非规范文件名/子目录的情况（比如未来 provider 把
    产物写到别处）——d2_gpu 必须把它们搬到规范的
    transcript.json/no_vocals.mp3/vocals.mp3。"""

    def transcribe_separate(self, audio: Path, out_dir: Path) -> dict:
        work = out_dir / "_remote_work"
        work.mkdir(parents=True, exist_ok=True)
        transcript_path = work / "raw_transcript.json"
        transcript_path.write_text(
            json.dumps({"duration": 2.0, "segments": [
                {"start": 0.0, "end": 1.0, "text": "hi"}]}), encoding="utf-8")
        no_vocals_path = work / "raw_no_vocals.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
             "-t", "1", str(no_vocals_path)],
            check=True, capture_output=True,
        )
        vocals_path = work / "raw_vocals.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
             "-t", "1", str(vocals_path)],
            check=True, capture_output=True,
        )
        return {"transcript": transcript_path, "no_vocals": no_vocals_path, "vocals": vocals_path}


def test_lands_outputs_at_canonical_names_when_remote_uses_different_paths(tmp_path, monkeypatch):
    import src.stages.d2_gpu as d2

    ctx = _ctx(tmp_path)
    _write_fake_source_mp4(ctx.job / d2.INPUT)
    monkeypatch.setattr(d2, "get_remote", lambda config: _DifferentPathRemote())

    d2.run(ctx)

    assert (ctx.job / d2.OUTPUT).exists()
    assert (ctx.job / d2.NO_VOCALS_OUTPUT).exists()
    assert (ctx.job / d2.VOCALS_OUTPUT).exists()
    data = read_json(ctx.job, d2.OUTPUT)
    assert data["segments"][0]["text"] == "hi"


# ---------------------------------------------------------------------------
# 陈旧产物：上次失败运行残留的 transcript.json/no_vocals.mp3 要被干净覆盖
# ---------------------------------------------------------------------------

def test_overwrites_stale_outputs_from_prior_failed_run(tmp_path):
    import src.stages.d2_gpu as d2

    ctx = _ctx(tmp_path)
    _write_fake_source_mp4(ctx.job / d2.INPUT)

    write_json(ctx.job, d2.OUTPUT, {"duration": 0, "segments": [], "stale": True})
    (ctx.job / d2.NO_VOCALS_OUTPUT).write_bytes(b"not a real mp3, leftover garbage")

    d2.run(ctx)

    data = read_json(ctx.job, d2.OUTPUT)
    assert "stale" not in data
    assert len(data["segments"]) >= 1

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(ctx.job / d2.NO_VOCALS_OUTPUT)],
        capture_output=True, text=True, check=True,
    )
    assert "audio" in probe.stdout          # 陈旧垃圾字节被换成真正可解码音频


# ---------------------------------------------------------------------------
# ffmpeg 音频提取失败 -> 清晰 RuntimeError(含 stderr)
# ---------------------------------------------------------------------------

def test_ffmpeg_extraction_failure_raises_clear_runtime_error(tmp_path, monkeypatch):
    import src.stages.d2_gpu as d2

    ctx = _ctx(tmp_path)
    (ctx.job / d2.INPUT).parent.mkdir(parents=True, exist_ok=True)
    (ctx.job / d2.INPUT).write_bytes(b"not a real mp4")   # ffmpeg 会失败退出

    with pytest.raises(RuntimeError, match="ffmpeg 音频提取失败"):
        d2.run(ctx)

    assert not (ctx.job / d2.OUTPUT).exists()


def test_ffmpeg_extraction_failure_message_includes_stderr(tmp_path, monkeypatch):
    import src.stages.d2_gpu as d2

    class _FakeCompleted:
        returncode = 1
        stdout = ""
        stderr = "Invalid data found when processing input"

    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _FakeCompleted(),
    )

    ctx = _ctx(tmp_path)
    with pytest.raises(RuntimeError, match="Invalid data found when processing input"):
        d2.run(ctx)


# ---------------------------------------------------------------------------
# 离线安全：模块顶层禁止触网 import；本文件所有测试均不发起真实 ssh
# ---------------------------------------------------------------------------

def test_module_import_does_no_network_at_top_level():
    import src.stages.d2_gpu as d2_mod

    banned = {"requests", "urllib3", "httpx", "paramiko", "fabric"}
    tree = ast.parse(inspect.getsource(d2_mod))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] not in banned for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in banned



def test_subs_only_mode_when_whisper_hears_nothing(tmp_path, monkeypatch):
    """字幕版(2026-07-04 用户定):whisper<3段而字幕有货=无解说的字幕片(ISS POV
    纯音乐片)——不再拦截,标记 mode=subs_only,下游跳过配音、保留原声、只烧字幕。"""
    import json as _json
    import pytest
    from src.config import Config
    from src.context import Ctx
    from src.errors import StageBlocked
    from src.llm import FakeLLM
    from src.stages import d2_gpu
    from src.util.jobs import write_json

    ctx = Ctx(slug="screentext", config=Config({"remote": {"provider": "fake"}}),
              root=tmp_path / "jobs", llm=FakeLLM())
    ctx.job.mkdir(parents=True, exist_ok=True)
    # 屏幕文字型字幕(有货)
    (ctx.job / "source_subs.json3").write_text(_json.dumps({"events": [
        {"tStartMs": 1000, "dDurationMs": 3000, "segs": [{"utf8": "Columbus Laboratory Module."}]},
    ]}), encoding="utf-8")

    class SilentRemote:
        def transcribe_separate(self, audio, out_dir):
            write_json(out_dir, "transcript.json", {"duration": 60.0, "segments": []})  # whisper 0段
            (out_dir / "no_vocals.mp3").write_bytes(b"N")
            (out_dir / "vocals.mp3").write_bytes(b"V")
            return {"transcript": out_dir / "transcript.json",
                    "no_vocals": out_dir / "no_vocals.mp3", "vocals": out_dir / "vocals.mp3"}
    monkeypatch.setattr(d2_gpu, "get_remote", lambda cfg: SilentRemote())
    monkeypatch.setattr(d2_gpu, "_extract_audio", lambda s, o: o.write_bytes(b"A"))
    (ctx.job / "source.mp4").write_bytes(b"MP4")

    d2_gpu.run(ctx)   # 不拦截:进入字幕版模式

    from src.util.jobs import read_json
    t = read_json(ctx.job, "transcript.json")
    assert t["mode"] == "subs_only"
    assert len(t["segments"]) == 1
    assert "Columbus" in t["segments"][0]["text"]


def test_zero_segments_registers_rejection_before_blocking(tmp_path, monkeypatch):
    """无解说且无字幕→拦截前登记拒片,重跑 d1 不再选中同一部(白化野牛片踩坑)。"""
    import pytest
    from src.config import Config
    from src.context import Ctx
    from src.errors import StageBlocked
    from src.llm import FakeLLM
    from src.stages import d2_gpu
    from src.util.jobs import write_json
    from src.util.registry import published_identifiers

    ctx = Ctx(slug="silent", config=Config({"remote": {"provider": "fake"}}),
              root=tmp_path / "jobs", llm=FakeLLM())
    ctx.job.mkdir(parents=True, exist_ok=True)
    write_json(ctx.job, "source.json", {"identifier": "yt:silentclip"})
    (ctx.job / "source.mp4").write_bytes(b"MP4")

    class SilentRemote:
        def transcribe_separate(self, audio, out_dir):
            write_json(out_dir, "transcript.json", {"duration": 60.0, "segments": []})
            (out_dir / "no_vocals.mp3").write_bytes(b"N")
            (out_dir / "vocals.mp3").write_bytes(b"V")
            return {"transcript": out_dir / "transcript.json",
                    "no_vocals": out_dir / "no_vocals.mp3", "vocals": out_dir / "vocals.mp3"}
    monkeypatch.setattr(d2_gpu, "get_remote", lambda cfg: SilentRemote())
    monkeypatch.setattr(d2_gpu, "_extract_audio", lambda s, o: o.write_bytes(b"A"))

    with pytest.raises(StageBlocked):
        d2_gpu.run(ctx)
    assert "yt:silentclip" in published_identifiers(ctx.root)   # 已登记,d1 不再选
