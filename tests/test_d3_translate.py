from __future__ import annotations

import ast
import inspect
import json

from src.config import Config
from src.context import Ctx
from src.llm import FakeLLM
from src.util.jobs import read_json, write_json

import src.stages.d3_translate as d3


def _ctx(tmp_path, llm=None) -> Ctx:
    return Ctx(slug="doc1", config=Config({}), root=tmp_path, llm=llm or FakeLLM())


def _write_transcript(ctx: Ctx, segments: list[dict], duration: float = 10.0) -> None:
    write_json(ctx.job, d3.INPUT, {"duration": duration, "segments": segments})


# ---------------------------------------------------------------------------
# happy path — ids aligned, timing preserved
# ---------------------------------------------------------------------------

def test_happy_path_maps_zh_by_id_and_preserves_timing(tmp_path):
    canned = json.dumps({
        "items": [
            {"id": 0, "zh": "你好世界"},
            {"id": 1, "zh": "这是第二句"},
        ]
    })
    ctx = _ctx(tmp_path, FakeLLM(canned=canned))
    _write_transcript(ctx, [
        {"start": 0.0, "end": 2.0, "text": "Hello world", "gender": "male"},
        {"start": 2.0, "end": 4.5, "text": "This is the second sentence", "gender": "female"},
    ])

    d3.run(ctx)

    out = read_json(ctx.job, d3.OUTPUT)
    assert out["duration"] == 10.0
    assert len(out["segments"]) == 2

    seg0, seg1 = out["segments"]
    assert seg0 == {
        "id": 0, "start": 0.0, "end": 2.0,
        "en": "Hello world", "zh": "你好世界", "gender": "male",
    }
    assert seg1 == {
        "id": 1, "start": 2.0, "end": 4.5,
        "en": "This is the second sentence", "zh": "这是第二句", "gender": "female",
    }
    assert ctx.warnings == []


# ---------------------------------------------------------------------------
# 0 segments -> empty output + warn, no crash
# ---------------------------------------------------------------------------

def test_zero_segments_writes_empty_output_and_warns(tmp_path):
    ctx = _ctx(tmp_path)
    _write_transcript(ctx, [], duration=0.0)

    d3.run(ctx)          # must not raise

    out = read_json(ctx.job, d3.OUTPUT)
    assert out["segments"] == []
    assert any("0" in w for w in ctx.warnings)
    assert ctx.llm.calls == []          # no LLM call for empty input


# ---------------------------------------------------------------------------
# P7 T9 — gender passthrough (transcript.json -> zh_script.json)
# ---------------------------------------------------------------------------

def test_gender_field_passes_through_from_transcript_to_zh_script(tmp_path):
    """d2_gpu(zion 侧基频估计) 写进 transcript.json 的 gender 必须原样透传到
    zh_script.json，供 d4_tts 按段挑男/女声。没有 gender 键的段(旧数据/检测
    失败) -> 输出 gender=None，不瞎猜。"""
    canned = json.dumps({
        "items": [
            {"id": 0, "zh": "男声"},
            {"id": 1, "zh": "女声"},
            {"id": 2, "zh": "未知"},
        ]
    })
    ctx = _ctx(tmp_path, FakeLLM(canned=canned))
    _write_transcript(ctx, [
        {"start": 0.0, "end": 1.0, "text": "one", "gender": "male"},
        {"start": 1.0, "end": 2.0, "text": "two", "gender": "female"},
        {"start": 2.0, "end": 3.0, "text": "three"},        # no gender key at all
    ])

    d3.run(ctx)

    out = read_json(ctx.job, d3.OUTPUT)
    genders = [seg["gender"] for seg in out["segments"]]
    assert genders == ["male", "female", None]


# ---------------------------------------------------------------------------
# id-mismatch resilience — LLM returns fewer items than segments
# ---------------------------------------------------------------------------

def test_missing_ids_fall_back_without_misalignment(tmp_path):
    # 3 segments in, LLM only returns translations for id 0 and id 2 (id 1 missing)
    canned = json.dumps({
        "items": [
            {"id": 0, "zh": "第一句"},
            {"id": 2, "zh": "第三句"},
        ]
    })
    ctx = _ctx(tmp_path, FakeLLM(canned=canned))
    _write_transcript(ctx, [
        {"start": 0.0, "end": 1.0, "text": "One"},
        {"start": 1.0, "end": 2.0, "text": "Two"},
        {"start": 2.0, "end": 3.0, "text": "Three"},
    ])

    d3.run(ctx)           # must not raise

    out = read_json(ctx.job, d3.OUTPUT)
    segs = out["segments"]
    assert len(segs) == 3          # all segments present, none dropped

    assert segs[0] == {
        "id": 0, "start": 0.0, "end": 1.0, "en": "One", "zh": "第一句", "gender": None,
    }
    # id 1 missing from LLM response -> fallback (English), timing/id untouched
    assert segs[1]["id"] == 1
    assert segs[1]["start"] == 1.0
    assert segs[1]["end"] == 2.0
    assert segs[1]["en"] == "Two"
    assert segs[1]["zh"] == "Two"          # fallback = English original
    assert segs[1]["gender"] is None       # input transcript never had gender -> passthrough None
    assert segs[2] == {
        "id": 2, "start": 2.0, "end": 3.0, "en": "Three", "zh": "第三句", "gender": None,
    }

    assert any("1/3" in w for w in ctx.warnings)


# ---------------------------------------------------------------------------
# regression guard: extract_json parses an OBJECT, never a bare top-level array
# ---------------------------------------------------------------------------

def test_prompt_requests_object_wrapped_items_not_bare_array():
    prompt = d3.build_prompt([{"id": 0, "en": "hi"}])
    assert '"items"' in prompt
    assert prompt.strip()[0] != "["            # never instructs a bare array reply


def test_object_wrapped_canned_response_parses_correctly(tmp_path):
    """Guards the exact pitfall: a canned {"items": [...]} response (object) must
    parse fine via extract_json — a bare `[...]` array response would raise."""
    canned = json.dumps({"items": [{"id": 0, "zh": "你好"}]})
    ctx = _ctx(tmp_path, FakeLLM(canned=canned))
    _write_transcript(ctx, [{"start": 0.0, "end": 1.0, "text": "hi"}])

    d3.run(ctx)          # must not raise (would raise ValueError if array-shaped)

    out = read_json(ctx.job, d3.OUTPUT)
    assert out["segments"][0]["zh"] == "你好"


# ---------------------------------------------------------------------------
# chunking — large transcripts are split into batches of CHUNK_SIZE and merged
# ---------------------------------------------------------------------------

class _EchoLLM:
    """Fake LLM that echoes back translations for whatever ids it was actually
    asked about (parses the object-wrapped prompt payload) — lets us verify
    chunk batches are each answered correctly and merged back without
    id collisions across batches."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        payload = json.loads(prompt.split("输入：", 1)[1])
        items = [{"id": it["id"], "zh": f"[ZH]{it['en']}"} for it in payload["items"]]
        return json.dumps({"items": items})


def test_large_transcript_is_chunked_and_merged_without_id_collisions(tmp_path):
    n = d3.CHUNK_SIZE * 2 + 5          # forces 3 batches
    segments = [
        {"start": float(i), "end": float(i) + 1, "text": f"sentence {i}"}
        for i in range(n)
    ]
    echo = _EchoLLM()
    ctx = _ctx(tmp_path, echo)
    _write_transcript(ctx, segments)

    d3.run(ctx)

    assert len(echo.calls) == 6          # 80+80+5 初译 + 全部超预算(echo恒回原文)定向返工再3批
    out = read_json(ctx.job, d3.OUTPUT)
    out_segs = out["segments"]
    assert len(out_segs) == n
    for i, seg in enumerate(out_segs):
        assert seg["id"] == i
        assert seg["zh"] == f"[ZH]sentence {i}"
    # echo 假译文恒超小预算 → 允许唯一的"定向压缩返工"提示,不允许其它告警
    assert all("定向压缩返工" in w for w in ctx.warnings)


# ---------------------------------------------------------------------------
# offline safety: module import does no network at top level; FakeLLM only
# ---------------------------------------------------------------------------

def test_module_import_does_no_network_at_top_level():
    banned = {"requests", "urllib3", "httpx", "paramiko", "fabric", "google"}
    tree = ast.parse(inspect.getsource(d3))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] not in banned for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in banned


class _FlakyLLM:
    """第一次吐残缺 JSON(真跑遇到的 Gemini 抽风),第二次正常。"""
    def __init__(self, good: str):
        self.good = good; self.calls = 0
    def generate(self, system: str, prompt: str) -> str:
        self.calls += 1
        return '{"items": [{"id": 0, "zh": "残缺' if self.calls == 1 else self.good


def test_batch_retries_on_malformed_json(tmp_path):
    """LLM 偶发残缺 JSON 不该杀死整条 run——重试后成功。"""
    import json as _json
    from src.config import Config
    from src.context import Ctx
    from src.util.jobs import write_json, read_json
    import src.stages.d3_translate as d3
    good = _json.dumps({"items": [{"id": 0, "zh": "你好"}]})
    ctx = Ctx(slug="c", config=Config({}), root=tmp_path, llm=_FlakyLLM(good))
    write_json(ctx.job, d3.INPUT, {"duration": 2, "segments": [
        {"start": 0, "end": 2, "text": "hi", "gender": "female"}]})
    d3.run(ctx)
    out = read_json(ctx.job, d3.OUTPUT)["segments"]
    assert out[0]["zh"] == "你好"
    assert any("解析失败" in w for w in ctx.warnings)


def test_batch_falls_back_after_3_bad_json(tmp_path):
    """三次全残缺→本批次回退 zh=en,不抛异常。"""
    from src.config import Config
    from src.context import Ctx
    from src.util.jobs import write_json, read_json
    from src.llm import FakeLLM
    import src.stages.d3_translate as d3
    ctx = Ctx(slug="c", config=Config({}), root=tmp_path,
              llm=FakeLLM(canned='{"items": [{"id": 0, "zh": "残缺'))
    write_json(ctx.job, d3.INPUT, {"duration": 2, "segments": [
        {"start": 0, "end": 2, "text": "hello world"}]})
    d3.run(ctx)
    out = read_json(ctx.job, d3.OUTPUT)["segments"]
    assert out[0]["zh"] == "hello world"          # 回退英文原文
