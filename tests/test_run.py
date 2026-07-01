from pathlib import Path
import json
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.util.jobs import write_json, read_json
from src.errors import StageBlocked
import src.run as run_mod


def test_run_pipeline_skips_when_output_exists(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_stage(ctx: Ctx) -> None:
        calls["n"] += 1
        write_json(ctx.job, "fake.json", {"ok": True})

    monkeypatch.setattr(run_mod, "STAGES", [("fake", fake_stage, "fake.json")])

    args = ("slugX", Config({}), tmp_path, FakeLLM())
    run_mod.run_pipeline(*args)
    assert calls["n"] == 1                       # 第一次跑
    run_mod.run_pipeline(*args)
    assert calls["n"] == 1                       # 产物已存在 → 跳过
    run_mod.run_pipeline(*args, force=True)
    assert calls["n"] == 2                       # --force → 重跑


def test_run_pipeline_from_to_subrange(tmp_path, monkeypatch):
    log = []

    def mk(name):
        def fn(ctx: Ctx) -> None:
            log.append(name)
            write_json(ctx.job, f"{name}.json", {})
        return fn

    monkeypatch.setattr(run_mod, "STAGES", [
        ("a", mk("a"), "a.json"),
        ("b", mk("b"), "b.json"),
        ("c", mk("c"), "c.json"),
    ])
    run_mod.run_pipeline("s", Config({}), tmp_path, FakeLLM(), frm="b", to="b")
    assert log == ["b"]


def test_run_pipeline_stops_on_blocked(tmp_path, monkeypatch):
    ran = []

    def blocker(ctx):
        ran.append("a")
        raise StageBlocked("nope")

    def after(ctx):
        ran.append("b")
        write_json(ctx.job, "b.json", {})

    monkeypatch.setattr(run_mod, "STAGES", [
        ("a", blocker, "a.json"),
        ("b", after, "b.json"),
    ])
    run_mod.run_pipeline("s", Config({}), tmp_path, FakeLLM())
    assert ran == ["a"]                       # b 未运行（被 blocked 停下）


class SeqLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.i = 0

    def generate(self, system, prompt):
        r = self.responses[self.i]
        self.i += 1
        return r


def test_end_to_end_topic_research_script(tmp_path):
    topic = json.dumps({"title_working": "罗斯威尔", "year": 1947})
    research = json.dumps({"facts": [
        {"claim": "军方新闻稿", "evidence": "documented", "source": "存档"},
        {"claim": "Mogul", "evidence": "documented", "source": "报告"},
        {"claim": "Mogul2", "evidence": "documented", "source": "报告"},
    ]})
    script = json.dumps({"title_options": ["a", "b", "c"], "segments": [
        {"id": 1, "text": "一二三四五六七八九十", "visual_query": "q", "est_sec": 8},
    ]})
    run_mod.run_pipeline("roswell-1947", Config({}), tmp_path,
                         SeqLLM([topic, research, script]), to="s3_script")
    job = tmp_path / "roswell-1947"
    assert read_json(job, "topic.json")["year"] == 1947
    assert read_json(job, "research.json")["blocked"] is False
    assert read_json(job, "script.json")["total_words"] == 10


def test_e2e_through_subtitles(tmp_path):
    topic = json.dumps({"title_working": "罗斯威尔", "year": 1947})
    research = json.dumps({"facts": [
        {"claim": "a", "evidence": "documented", "source": "s"},
        {"claim": "b", "evidence": "documented", "source": "s"},
        {"claim": "c", "evidence": "documented", "source": "s"},
    ]})
    script = json.dumps({"title_options": ["t"], "segments": [
        {"id": 1, "text": "飞碟来了。", "visual_query": "q", "est_sec": 3},
    ]})
    cfg = Config({"tts": {"provider": "fake"}, "media": {"provider": "fake"}})
    run_mod.run_pipeline("roswell-1947", cfg, tmp_path,
                         SeqLLM([topic, research, script]), to="s6_media")
    job = tmp_path / "roswell-1947"
    assert read_json(job, "timing.json")["segments"][0]["words"][0]["text"] == "飞"
    assert "Dialogue:" in (job / "subs.ass").read_text(encoding="utf-8")
    assert (job / "audio" / "seg_001.mp3").exists()


def test_e2e_full_pipeline(tmp_path, monkeypatch):
    """九阶段端到端：s1~s9 全链路、全 fake provider，验证 STAGES 注册与编排，不触网。

    两处 monkeypatch 不是给 s8/s9 打补丁，而是绕开 FakeMedia 占位实现的既有缺口：
    FakeMedia.fetch 写的是占位字节 b"FAKEIMG"，不是合法 JPEG——此前没有测试真正把它
    交给 ffmpeg/PIL 解码过。真实 ffmpeg 对 -loop 1 的图片输入遇到非法数据会反复重试
    解码而不是快速报错（实测挂起 > 2 分钟未完成），s8 的封面背景也会真的用 PIL 打开
    这张图而抛 UnidentifiedImageError。这两处都在 run.py/媒体供给的既有实现里，改
    src/media.py 会破坏 test_media.py / test_s6_media.py 对 b"FAKEIMG" 的断言，超出本
    任务范围；因此只在本测试内用 monkeypatch 让 FakeMedia 写一张合法小图，并按
    test_s7_assemble.py 的既有写法 mock 掉 ffmpeg 子进程，使 e2e 保持离线、确定性、
    亚秒级。
    """
    import json
    from pathlib import Path

    from PIL import Image

    import src.media as media_mod
    import src.stages.s7_assemble as s7

    def fake_fetch(self, candidate, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16)).save(dest)

    monkeypatch.setattr(media_mod.FakeMedia, "fetch", fake_fetch)

    # s7.subprocess 是标准库 subprocess 模块本体（同一对象），s4_tts 的 ffprobe 时长探测
    # 也走它；只假掉 ffmpeg 调用本身，其余命令（如 ffprobe）透传给真实 subprocess.run。
    real_subprocess_run = s7.subprocess.run

    def fake_ffmpeg_run(cmd, **kw):
        if cmd and cmd[0] == "ffmpeg":
            Path(cmd[-1]).write_bytes(b"FAKEMP4")

            class R:
                returncode = 0

            return R()
        return real_subprocess_run(cmd, **kw)

    monkeypatch.setattr(s7.subprocess, "run", fake_ffmpeg_run)

    topic = json.dumps({"title_working": "罗斯威尔", "year": 1947})
    research = json.dumps({"facts": [
        {"claim": "a", "evidence": "documented", "source": "s"},
        {"claim": "b", "evidence": "documented", "source": "s"},
        {"claim": "c", "evidence": "documented", "source": "s"}]})
    script = json.dumps({"title_options": ["罗斯威尔之谜"],
                         "segments": [{"id": 1, "text": "飞碟坠落。", "visual_query": "roswell", "est_sec": 3}]})
    meta = json.dumps({"title": "罗斯威尔之谜", "description": "真相。", "tags": ["UFO"]})
    cfg = Config({"tts": {"provider": "fake"}, "media": {"provider": "fake"},
                  "youtube": {"provider": "fake"},
                  "publish": {"auto_upload": True, "visibility": "private"}})
    run_mod.run_pipeline("roswell-1947", cfg, tmp_path,
                         SeqLLM([topic, research, script, meta]))
    job = tmp_path / "roswell-1947"
    assert read_json(job, "meta.json")["title"] == "罗斯威尔之谜"
    assert read_json(job, "publish.json")["status"] == "private"      # 全链通到发布
    assert (job / "video.mp4").exists() and (job / "thumb.jpg").exists()


def test_e2e_through_media(tmp_path):
    topic = json.dumps({"title_working": "罗斯威尔", "year": 1947})
    research = json.dumps({"facts": [
        {"claim": "a", "evidence": "documented", "source": "s"},
        {"claim": "b", "evidence": "documented", "source": "s"},
        {"claim": "c", "evidence": "documented", "source": "s"},
    ]})
    script = json.dumps({"title_options": ["t"], "segments": [
        {"id": 1, "text": "飞碟来了。", "visual_query": "roswell", "est_sec": 3},
    ]})
    cfg = Config({"tts": {"provider": "fake"}, "media": {"provider": "fake"}})
    run_mod.run_pipeline("roswell-1947", cfg, tmp_path,
                         SeqLLM([topic, research, script]), to="s6_media")
    job = tmp_path / "roswell-1947"
    assert read_json(job, "credits.json")["items"][0]["license"] == "public-domain"
    assert (job / "media" / "seg_001.jpg").exists()
    assert (job / "subs.ass").read_text(encoding="utf-8").count("Dialogue:") >= 1
