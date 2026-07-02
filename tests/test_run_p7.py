"""P7 T7：编排层测试——管线选择器（original vs doc）+ P7 d1-d7 全链路 e2e。

⚠️ 头号风险（本项目每个阶段都复发过）：e2e 一旦漏 fake 掉某个 provider，就会真触网/
真 ssh。这里对 d1_source 的处理与既有 tests/test_d1_source.py 完全一致的思路——
`requests.get` 全量 monkeypatch（search/metadata 假响应；download 分支直接断言不该被
走到，因为真正的"下载"改由 monkeypatch `d1_source._download` 现场用 ffmpeg lavfi 合成
一份可解码的小 mp4，不发一次真实 HTTP 请求）。d2_gpu 用 `remote.provider=fake`
（FakeRemoteGPU，见 src/remote.py），d3_translate 用显式传入的 FakeLLM(canned=...)，
d4_tts 用 `tts.provider=fake`（FakeTTS）。d5_subs/d7_assemble 全程本地 ffmpeg/文本处理，
不触网。全程无一次真实 HTTP/SSH。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src.config import Config
from src.llm import FakeLLM
from src.util.jobs import read_json, write_json

import src.run as run_mod
import src.stages.d1_source as d1_source


# ---------------------------------------------------------------------------
# 管线选择器 —— 纯编排层，不跑任何真实阶段
# ---------------------------------------------------------------------------

def test_stages_for_resolves_original_and_doc():
    assert run_mod._stages_for("original") is run_mod.STAGES
    assert run_mod._stages_for("doc") is run_mod.STAGES_P7


def test_stages_p7_includes_d10_archive_right_after_d9_publish():
    """P7 T15：归档阶段接在发布之后，且是链路的最后一站。"""
    names = [s[0] for s in run_mod.STAGES_P7]
    assert names[-2:] == ["d9_publish", "d10_archive"]


def test_stages_for_rejects_unknown_pipeline_name():
    with pytest.raises(ValueError, match="未知 pipeline"):
        run_mod._stages_for("nope")


def test_run_pipeline_default_config_uses_original_stages(tmp_path, monkeypatch):
    calls = []

    def fake_stage(ctx):
        calls.append("orig")
        write_json(ctx.job, "fake.json", {})

    monkeypatch.setattr(run_mod, "STAGES", [("fake", fake_stage, "fake.json")])
    run_mod.run_pipeline("s", Config({}), tmp_path, FakeLLM())          # 无 pipeline 键
    assert calls == ["orig"]


def test_run_pipeline_config_pipeline_key_selects_doc_stages(tmp_path, monkeypatch):
    calls = []

    def fake_stage(ctx):
        calls.append("doc")
        write_json(ctx.job, "fake.json", {})

    monkeypatch.setattr(run_mod, "STAGES_P7", [("fake", fake_stage, "fake.json")])
    run_mod.run_pipeline("s", Config({"pipeline": "doc"}), tmp_path, FakeLLM())
    assert calls == ["doc"]


def test_run_pipeline_explicit_pipeline_arg_overrides_config_key(tmp_path, monkeypatch):
    calls = []

    def fake_stage(ctx):
        calls.append("doc")
        write_json(ctx.job, "fake.json", {})

    monkeypatch.setattr(run_mod, "STAGES_P7", [("fake", fake_stage, "fake.json")])
    # config 说 original，显式 pipeline= 入参必须赢
    run_mod.run_pipeline("s", Config({"pipeline": "original"}), tmp_path, FakeLLM(),
                          pipeline="doc")
    assert calls == ["doc"]


def test_main_cli_pipeline_flag_wires_through_to_run_pipeline(tmp_path, monkeypatch):
    """--pipeline doc 从 argparse 一路接到 run_pipeline 的 pipeline= 入参；
    不跑真实阶段（run_pipeline/get_llm 都被替身接管），只校验接线正确。"""
    captured = {}

    def fake_run_pipeline(slug, config, root, llm, frm=None, to=None, force=False,
                           pipeline=None):
        captured["slug"] = slug
        captured["pipeline"] = pipeline

    monkeypatch.setattr(run_mod, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(run_mod, "get_llm", lambda config: FakeLLM())
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("cost_tier: free\n", encoding="utf-8")

    run_mod.main(["--case", "x", "--pipeline", "doc", "--config", str(cfg_path)])
    assert captured == {"slug": "x", "pipeline": "doc"}


def test_main_cli_pipeline_flag_defaults_to_none_deferring_to_config(tmp_path, monkeypatch):
    captured = {}

    def fake_run_pipeline(slug, config, root, llm, frm=None, to=None, force=False,
                           pipeline=None):
        captured["pipeline"] = pipeline

    monkeypatch.setattr(run_mod, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(run_mod, "get_llm", lambda config: FakeLLM())
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("cost_tier: free\n", encoding="utf-8")

    run_mod.main(["--case", "x", "--config", str(cfg_path)])   # 没传 --pipeline
    assert captured["pipeline"] is None


# ---------------------------------------------------------------------------
# P7 d1-d7 全链路 e2e —— 全 fake/离线，产出 video.mp4
# ---------------------------------------------------------------------------

class FakeArchiveResp:
    def __init__(self, json_data=None):
        self._json = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


def _install_fake_archive_get(monkeypatch, identifier: str, title: str):
    """只假 search + metadata 两个分支；download 分支若被真的走到就直接
    AssertionError——真正的"下载"改由 monkeypatch `_download` 本身现场用 ffmpeg
    lavfi 合成小 mp4，requests.get 永远不该收到 /download/ 请求。"""

    search_json = {"response": {"docs": [{"identifier": identifier, "title": title}]}}
    metadata_json = {
        "nasadoc1": {
            "metadata": {
                "licenseurl": "http://creativecommons.org/licenses/publicdomain/",
                "collection": ["nasa"],
                "title": title,
            },
            "files": [
                {"name": f"{identifier}.mp4", "size": "300000", "length": "600",
                 "width": "1280", "height": "720"},
            ],
        },
    }

    def fake_get(url, params=None, headers=None, timeout=None, stream=False):
        assert headers and "User-Agent" in headers
        if "advancedsearch.php" in url:
            return FakeArchiveResp(json_data=search_json)
        if "/metadata/" in url:
            ident = url.rsplit("/", 1)[-1]
            return FakeArchiveResp(json_data=metadata_json.get(ident, {}))
        raise AssertionError(
            f"e2e 离线契约破了：requests.get 不该走到 {url!r}"
            "（下载应由 monkeypatch 的 _download 现场用 ffmpeg lavfi 合成，不发真实 HTTP）"
        )

    monkeypatch.setattr("requests.get", fake_get)


def _fake_download_writes_lavfi_mp4(url: str, dest: Path) -> None:
    """替身版 d1_source._download：不发任何 HTTP 请求，现场用 ffmpeg 的
    lavfi 生成一份真正可解码的 mp4（video+audio 双流，2 秒）——与
    tests/test_d2_gpu.py._write_fake_source_mp4 同一手法，供下游 d2_gpu 的
    ffmpeg 音频提取 / d7_assemble 的合成消费。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "color=c=black:s=64x64:d=2",
         "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
         "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(dest)],
        check=True, capture_output=True,
    )


def _ffprobe_kinds(path: Path) -> set[str]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return set(r.stdout.split())


def _require_ffmpeg():
    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0:
        pytest.skip("ffmpeg not available")


class _ScriptedLLM:
    """d1-d9 全链路里有两个阶段会调 LLM（d3_translate、d8_meta），各自要的
    JSON 形状完全不同（items 数组包 vs title/description/tags/image_prompt
    对象），不能像单阶段测试那样用一个 FakeLLM(canned=...) 打天下。按
    system prompt 里的关键词分发到各自的罐头响应，不改动 src/llm.FakeLLM
    （其它测试仍用它）。"""

    def __init__(self, responses: dict[str, str]):
        self._responses = responses
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        for key, resp in self._responses.items():
            if key in system:
                return resp
        raise AssertionError(f"e2e 离线契约破了：未预期的 system prompt {system[:80]!r}")


def test_p7_doc_pipeline_e2e_all_offline_produces_video_and_publishes(tmp_path, monkeypatch):
    _require_ffmpeg()

    _install_fake_archive_get(monkeypatch, "nasadoc1", "NASA Space Documentary")
    monkeypatch.setattr(d1_source, "_download", _fake_download_writes_lavfi_mp4)

    # FakeRemoteGPU（src/remote.py）罐头 transcript 固定 2 段：
    #   id 0 "This is a fake transcript segment."
    #   id 1 "It exists purely for offline tests."
    canned_translation = json.dumps({"items": [
        {"id": 0, "zh": "这是一个假的转录片段。"},
        {"id": 1, "zh": "它纯粹是为了离线测试而存在。"},
    ]})
    canned_meta = json.dumps({
        "title": "NASA太空纪录片：人类最伟大的冒险",
        "description": "一段震撼的太空探索旅程，带你重返历史现场。",
        "tags": ["NASA", "太空", "纪录片"],
        "image_prompt": "a lone astronaut facing a giant ringed planet, dramatic lighting, no text",
    })
    llm = _ScriptedLLM({
        "字幕翻译": canned_translation,        # d3_translate.SYSTEM 的关键词
        "YouTube 运营": canned_meta,           # d8_meta.SYSTEM 的关键词
    })

    cfg = Config({
        "pipeline": "doc",              # 走 config 的选择器路径（不是显式 pipeline= 入参）
        "remote": {"provider": "fake"},
        "tts": {"provider": "fake"},
        "llm": {"provider": "fake"},
        "youtube": {"provider": "fake"},           # 阶段⑨：假上传器，绝不真触网 YouTube
        "publish": {"auto_upload": True, "visibility": "private"},  # 护栏#3 仍是私享
        "p7": {"collections": ["nasa"]},
        # 阶段⑩：archive.dir 指向 tmp_path 下的假 NAS，绝不能吃 config.yaml 里
        # 真实的 `~/nas/Share/youtuber` 默认值（那样会在跑测试的机器家目录下
        # mkdir 出一棵真目录树）。trim_local=False，好让下面沿用的既有断言
        # （job/video.mp4 等还在本地、ffprobe 能读到真实音视频流）继续成立——
        # 裁剪行为本身已经在 tests/test_d10_archive.py 里独立覆盖。
        "archive": {"dir": str(tmp_path / "nas_archive"), "trim_local": False},
    })
    (tmp_path / "nas_archive").mkdir()   # d10_archive 要求归档根目录预先存在，才不算 unreachable

    run_mod.run_pipeline("nasa-e2e", cfg, tmp_path, llm)

    job = tmp_path / "nasa-e2e"

    # 全链路每一站的产物都落地了（含新接的 阶段⑧⑨）
    assert (job / "source.mp4").exists()
    assert (job / "transcript.json").exists()
    assert (job / "zh_script.json").exists()
    assert (job / "narration.json").exists()
    assert (job / "subs.ass").exists()
    video = job / "video.mp4"
    assert video.exists() and video.stat().st_size > 0
    assert (job / "meta.json").exists()
    assert (job / "thumb_bg.png").exists()
    assert (job / "thumb.jpg").exists()
    assert (job / "publish.json").exists()
    assert (job / "archive.json").exists()

    # 翻译真的走了 LLM（不是英文回退）
    zh_script = read_json(job, "zh_script.json")
    assert zh_script["segments"][0]["zh"] == "这是一个假的转录片段。"
    assert zh_script["segments"][1]["zh"] == "它纯粹是为了离线测试而存在。"
    assert len(llm.calls) == 2                    # d3_translate 一批(2段<80) + d8_meta 一次

    # 配音真的合成了音频文件
    narration = read_json(job, "narration.json")
    assert len(narration["segments"]) == 2
    for seg in narration["segments"]:
        assert (job / seg["file"]).exists()

    # 成片是真正可解码、有音有画的视频（不是占位字节）
    kinds = _ffprobe_kinds(video)
    assert "video" in kinds and "audio" in kinds

    # 阶段⑧：AI 缩略图 + 简介，署名真的追加了
    from PIL import Image
    meta = read_json(job, "meta.json")
    assert meta["title"] == "NASA太空纪录片：人类最伟大的冒险"
    assert "NASA / NASA Space Documentary" in meta["description"]
    assert "公有领域" in meta["description"]
    thumb = job / "thumb.jpg"
    assert Image.open(thumb).size == (1280, 720)

    # 阶段⑨：s9_publish 原样复用，护栏#3 依旧生效（private，不是默认公开）
    publish = read_json(job, "publish.json")
    assert publish["videoId"] == "FAKE12345"
    assert publish["status"] == "private"

    # 阶段⑩：全链路最后一站——发布完的 job 真的搬去了（假）NAS，且校验通过
    archive = read_json(job, "archive.json")
    assert archive["status"] == "archived"
    dest = Path(archive["dest"])
    assert dest == tmp_path / "nas_archive" / "nasa-e2e"
    assert (dest / "video.mp4").exists()
    assert (dest / "video.mp4").stat().st_size == video.stat().st_size
    assert (dest / "meta.json").exists()
    assert archive["trimmed"] == []          # trim_local=False，本地一律没删

    # P7 T16：登记表——发布成功后 jobs_root/published.json 记下了这一集，
    # 供下次 run 的 d1_source 去重，防止同一条 query 反复重挑同一部片。
    published = read_json(tmp_path, "published.json")
    assert len(published) == 1
    entry = published[0]
    assert entry["identifier"] == "nasadoc1"
    assert entry["videoId"] == "FAKE12345"
    assert entry["slug"] == "nasa-e2e"
    assert entry["published_at"]
    # 归档也把登记表备份到了（假）NAS 根目录
    assert (tmp_path / "nas_archive" / "published.json").exists()
