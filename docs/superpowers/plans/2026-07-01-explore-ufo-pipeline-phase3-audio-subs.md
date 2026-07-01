# 探索飞碟与外星人 · P3 音字 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `script.json` 变成中文配音 + 精确字幕：阶段④ 用 edge-tts（免费，无配额）合成逐段音频并采集字级时间戳，阶段⑤ 用文稿+时间戳直接生成 `.ass` 字幕（无需 Whisper）。全程免费、不碰 LLM 配额。

**Architecture:** 沿用 P1/P2 的模块化 CLI + job 文件夹交接。TTS 引擎抽象成 provider（`edge`/`fake`），与 LLM provider 同构，使单测离线。阶段④是首个多产物阶段（`audio/` + `timing.json`）——用「最后写入的 `timing.json` 作为 checkpoint」满足现有单产物 skip 机制，**不改编排器**。阶段⑤纯逻辑：读 `timing.json` 的字级全局时间戳，按字数/句读分行，渲染 ASS。

**Tech Stack:** Python 3.14；`edge-tts`（免费微软中文神经音 + WordBoundary 时间戳）；ASS 字幕（纯文本）；pytest（TDD，单测用 FakeTTS 离线）。

## Global Constraints

- Python ≥ 3.14；新代码 `from __future__ import annotations` + 现代类型。
- **TTS 走 provider 抽象**（`src/tts.py`）：`TTSEngine.synth(text) -> tuple[bytes, list[dict]]`，words 为**段内相对**时间 `{text, offset_sec, dur_sec}`。`edge`→EdgeTTS（惰性 import edge_tts，async 包一层），`fake`→FakeTTS（确定性、离线）。`get_tts(config)` 按 `config.tts.provider` 选（默认 `edge`）。单测**绝不联网**（用 fake）。
- 阶段接口 `run(ctx: Ctx) -> None` + 模块 `OUTPUT`/`INPUT`。**阶段④ `OUTPUT="timing.json"`（最后写，作 checkpoint）；音频落在 `audio/`。**
- `timing.json` 的 word `start_sec` 为**全局**时间（= 段起点累积 + 段内 offset），供字幕直接用。
- UTF-8，JSON `ensure_ascii=False`，字幕文件 UTF-8。
- 声线/语速读 `config.tts.voice`（zh-CN-YunyangNeural）/`config.tts.rate`（-8%）。

## File Structure

```
新增：
  src/tts.py                    # TTSEngine 协议 / FakeTTS / EdgeTTS / get_tts
  src/stages/s4_tts.py          # 阶段④ 配音 + 字级时间戳
  src/stages/s5_subs.py         # 阶段⑤ ASS 字幕
  tests/test_tts.py
  tests/test_s4_tts.py
  tests/test_s5_subs.py
修改：
  config.yaml                   # tts.provider: edge
  src/run.py                    # 注册 s4/s5
  tests/test_run.py             # 五阶段 e2e（FakeLLM+FakeTTS）
```

---

## Task 1: TTS 引擎抽象 `src/tts.py`

**Files:**
- Create: `src/tts.py`, `tests/test_tts.py`
- Modify: `config.yaml`

**Interfaces:**
- Produces:
  - `TTSEngine(Protocol)`: `synth(self, text: str) -> tuple[bytes, list[dict]]`（words 段内相对：`{"text","offset_sec","dur_sec"}`）。
  - `FakeTTS`: 确定性——每字一个 word，每字 0.3s，音频固定字节；离线。
  - `EdgeTTS(voice, rate)`: 惰性 import `edge_tts`，`asyncio.run` 跑 `Communicate.stream()`，收集 audio + WordBoundary（offset/duration 是 100ns 单位 → /1e7 秒）。
  - `get_tts(config) -> TTSEngine`: `fake`→FakeTTS；`edge`→EdgeTTS(voice,rate)；else `ValueError`。

- [ ] **Step 1: 写失败测试 `tests/test_tts.py`**
```python
import pytest
from src.config import Config
from src.tts import FakeTTS, get_tts


def test_fake_tts_deterministic():
    tts = FakeTTS()
    data, words = tts.synth("飞碟")
    assert data  # 有音频字节
    assert [w["text"] for w in words] == ["飞", "碟"]
    assert words[0]["offset_sec"] == 0.0
    assert words[1]["offset_sec"] == 0.3
    assert all(w["dur_sec"] == 0.3 for w in words)


def test_get_tts_fake():
    assert isinstance(get_tts(Config({"tts": {"provider": "fake"}})), FakeTTS)


def test_get_tts_unknown_raises():
    with pytest.raises(ValueError):
        get_tts(Config({"tts": {"provider": "nope"}}))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_tts.py -q`
Expected: FAIL（`No module named 'src.tts'`）

- [ ] **Step 3: 写 `src/tts.py`**
```python
from __future__ import annotations

from typing import Protocol

from src.config import Config


class TTSEngine(Protocol):
    def synth(self, text: str) -> tuple[bytes, list[dict]]: ...


class FakeTTS:
    """确定性离线 TTS：每字一个 word，每字 0.3s。"""

    def synth(self, text: str) -> tuple[bytes, list[dict]]:
        words = []
        t = 0.0
        for ch in text:
            words.append({"text": ch, "offset_sec": round(t, 3), "dur_sec": 0.3})
            t += 0.3
        return b"FAKEAUDIO", words


class EdgeTTS:
    """edge-tts 免费微软中文神经音 + 字级时间戳。惰性 import。"""

    def __init__(self, voice: str = "zh-CN-YunyangNeural", rate: str = "-8%"):
        self.voice = voice
        self.rate = rate

    def synth(self, text: str) -> tuple[bytes, list[dict]]:
        import asyncio

        import edge_tts

        async def _go() -> tuple[bytes, list[dict]]:
            comm = edge_tts.Communicate(text, voice=self.voice, rate=self.rate)
            audio = bytearray()
            words: list[dict] = []
            async for ch in comm.stream():
                if ch["type"] == "audio":
                    audio.extend(ch["data"])
                elif ch["type"] == "WordBoundary":
                    words.append({
                        "text": ch["text"],
                        "offset_sec": round(ch["offset"] / 1e7, 3),
                        "dur_sec": round(ch["duration"] / 1e7, 3),
                    })
            return bytes(audio), words

        return asyncio.run(_go())


def get_tts(config: Config) -> TTSEngine:
    provider = config.get("tts.provider", "edge")
    if provider == "fake":
        return FakeTTS()
    if provider == "edge":
        return EdgeTTS(
            config.get("tts.voice", "zh-CN-YunyangNeural"),
            config.get("tts.rate", "-8%"),
        )
    raise ValueError(f"未知 tts provider: {provider}")
```

- [ ] **Step 4: config.yaml 加 provider**

在 `config.yaml` 的 `tts:` 块里，`voice:` 之上加一行：
```yaml
  provider: edge            # edge | fake
```

- [ ] **Step 5: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_tts.py -q`
Expected: `3 passed`

- [ ] **Step 6: Commit**
```bash
git add src/tts.py config.yaml tests/test_tts.py
git commit -m "feat(tts): TTS provider 抽象（edge-tts / FakeTTS）"
```

---

## Task 2: 阶段④ 配音 `src/stages/s4_tts.py`

**Files:**
- Create: `src/stages/s4_tts.py`, `tests/test_s4_tts.py`

**Interfaces:**
- Consumes: `Ctx`、`read_json`/`write_json`、`get_tts`（Task 1）。
- Produces: `OUTPUT="timing.json"`, `INPUT="script.json"`, `run(ctx)`。副产物 `audio/seg_NNN.mp3`。`timing.json` word `start_sec` 为全局。

- [ ] **Step 1: 写失败测试 `tests/test_s4_tts.py`**
```python
import json
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.util.jobs import write_json, read_json
from src.stages.s4_tts import run, OUTPUT, INPUT


def _ctx(tmp_path, script):
    cfg = Config({"tts": {"provider": "fake"}})
    ctx = Ctx(slug="c", config=cfg, root=tmp_path, llm=FakeLLM())
    write_json(ctx.job, INPUT, script)
    return ctx


def test_run_writes_timing_and_audio(tmp_path):
    script = {"segments": [
        {"id": 1, "text": "飞碟"},     # 2 字 → 0.6s
        {"id": 2, "text": "外星人"},   # 3 字 → 0.9s
    ]}
    ctx = _ctx(tmp_path, script)
    run(ctx)
    timing = read_json(ctx.job, OUTPUT)
    assert len(timing["segments"]) == 2
    s1, s2 = timing["segments"]
    assert s1["start_sec"] == 0.0
    assert round(s2["start_sec"], 1) == 0.6           # 累积：段2 起点 = 段1 时长
    assert timing["total_sec"] == round(0.6 + 0.9, 3)
    # 全局字时间：段2 首字从 0.6 开始
    assert s2["words"][0]["start_sec"] == 0.6
    assert (ctx.job / "audio" / "seg_001.mp3").exists()
    assert (ctx.job / "audio" / "seg_002.mp3").exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_s4_tts.py -q`
Expected: FAIL（`No module named 'src.stages.s4_tts'`）

- [ ] **Step 3: 写 `src/stages/s4_tts.py`**
```python
from __future__ import annotations

from src.context import Ctx
from src.tts import get_tts
from src.util.jobs import read_json, write_json

OUTPUT = "timing.json"
INPUT = "script.json"


def run(ctx: Ctx) -> None:
    script = read_json(ctx.job, INPUT)
    engine = get_tts(ctx.config)
    audio_dir = ctx.job / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    seg_out: list[dict] = []
    cursor = 0.0
    for seg in script.get("segments", []):
        text = seg.get("text", "")
        data, words = engine.synth(text)
        name = f"seg_{int(seg.get('id', 0)):03d}.mp3"
        (audio_dir / name).write_bytes(data)
        dur = (words[-1]["offset_sec"] + words[-1]["dur_sec"]) if words else 0.0
        gwords = [
            {
                "text": w["text"],
                "start_sec": round(cursor + w["offset_sec"], 3),
                "dur_sec": w["dur_sec"],
            }
            for w in words
        ]
        seg_out.append({
            "id": seg.get("id"),
            "audio": name,
            "start_sec": round(cursor, 3),
            "dur_sec": round(dur, 3),
            "words": gwords,
        })
        cursor += dur

    write_json(ctx.job, OUTPUT, {"segments": seg_out, "total_sec": round(cursor, 3)})
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_s4_tts.py -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**
```bash
git add src/stages/s4_tts.py tests/test_s4_tts.py
git commit -m "feat(s4): 阶段④ edge-tts 配音 + 字级时间戳（timing.json 作 checkpoint）"
```

---

## Task 3: 阶段⑤ 字幕 `src/stages/s5_subs.py`

**Files:**
- Create: `src/stages/s5_subs.py`, `tests/test_s5_subs.py`

**Interfaces:**
- Consumes: `Ctx`、`read_json`。
- Produces: `OUTPUT="subs.ass"`, `INPUT="timing.json"`, `run(ctx)`, 辅助 `build_lines(timing, max_chars)`（→ `list[tuple[start,end,text]]`）、`ass_time(sec)`、`render_ass(lines, config)`。
- 分行规则：逐段累积字，达 `max_chars`（默认 16）或遇句读（`。！？，、；`）即断行；行首时间=首字 start，行尾时间=末字 start+dur。

- [ ] **Step 1: 写失败测试 `tests/test_s5_subs.py`**
```python
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.util.jobs import write_json
from src.stages.s5_subs import run, build_lines, ass_time, OUTPUT, INPUT


def test_ass_time():
    assert ass_time(0) == "0:00:00.00"
    assert ass_time(3.25) == "0:00:03.25"
    assert ass_time(3661.5) == "1:01:01.50"


def test_build_lines_breaks_on_punct_and_length():
    timing = {"segments": [{"words": [
        {"text": "今", "start_sec": 0.0, "dur_sec": 0.3},
        {"text": "天", "start_sec": 0.3, "dur_sec": 0.3},
        {"text": "。", "start_sec": 0.6, "dur_sec": 0.3},   # 句读 → 断行
        {"text": "很", "start_sec": 0.9, "dur_sec": 0.3},
        {"text": "好", "start_sec": 1.2, "dur_sec": 0.3},
    ]}]}
    lines = build_lines(timing, max_chars=16)
    assert len(lines) == 2
    assert lines[0][2] == "今天。"
    assert lines[0][0] == 0.0
    assert round(lines[0][1], 1) == 0.9        # 末字(。) start0.6+dur0.3
    assert lines[1][2] == "很好"


def test_run_writes_ass(tmp_path):
    ctx = Ctx(slug="c", config=Config({}), root=tmp_path, llm=FakeLLM())
    write_json(ctx.job, INPUT, {"segments": [{"words": [
        {"text": "飞", "start_sec": 0.0, "dur_sec": 0.3},
        {"text": "碟", "start_sec": 0.3, "dur_sec": 0.3},
    ]}]})
    run(ctx)
    txt = (ctx.job / OUTPUT).read_text(encoding="utf-8")
    assert "[Script Info]" in txt
    assert "[Events]" in txt
    assert txt.count("Dialogue:") == 1
    assert "飞碟" in txt
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_s5_subs.py -q`
Expected: FAIL（`No module named 'src.stages.s5_subs'`）

- [ ] **Step 3: 写 `src/stages/s5_subs.py`**
```python
from __future__ import annotations

from src.context import Ctx
from src.util.jobs import read_json

OUTPUT = "subs.ass"
INPUT = "timing.json"
SENT_PUNCT = "。！？，、；"

_ASS_HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,3,1,2,80,80,70,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_time(sec: float) -> str:
    cs = int(round(sec * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def build_lines(timing: dict, max_chars: int = 16) -> list[tuple[float, float, str]]:
    lines: list[tuple[float, float, str]] = []
    for seg in timing.get("segments", []):
        cur: list[dict] = []
        for w in seg.get("words", []):
            cur.append(w)
            text = "".join(x["text"] for x in cur)
            last = w["text"][-1] if w.get("text") else ""
            if len(text) >= max_chars or last in SENT_PUNCT:
                start = cur[0]["start_sec"]
                end = w["start_sec"] + w["dur_sec"]
                lines.append((start, end, text))
                cur = []
        if cur:
            start = cur[0]["start_sec"]
            end = cur[-1]["start_sec"] + cur[-1]["dur_sec"]
            lines.append((start, end, "".join(x["text"] for x in cur)))
    return lines


def render_ass(lines: list[tuple[float, float, str]], config: Ctx | None = None) -> str:
    font = "Noto Sans CJK SC"
    size = 54
    body = "".join(
        f"Dialogue: 0,{ass_time(s)},{ass_time(e)},Default,,0,0,0,,{text}\n"
        for s, e, text in lines
    )
    return _ASS_HEAD.format(font=font, size=size) + body


def run(ctx: Ctx) -> None:
    timing = read_json(ctx.job, INPUT)
    max_chars = ctx.config.get("subs.max_chars", 16)
    lines = build_lines(timing, max_chars)
    (ctx.job / OUTPUT).write_text(render_ass(lines), encoding="utf-8")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_s5_subs.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**
```bash
git add src/stages/s5_subs.py tests/test_s5_subs.py
git commit -m "feat(s5): 阶段⑤ 文稿+时间戳生成 ASS 字幕（无需 Whisper）"
```

---

## Task 4: 注册 + 五阶段端到端 + 真实 edge-tts 冒烟

**Files:**
- Modify: `src/run.py`, `tests/test_run.py`

**Interfaces:**
- Consumes: `s4_tts`, `s5_subs`（Task 2/3）。
- Produces: `STAGES` 增加 s4/s5；五阶段离线 e2e（SeqLLM 出脚本 + `tts.provider=fake` 出音字）。

- [ ] **Step 1: 加 e2e 测试到 `tests/test_run.py`**

在文件末尾追加（复用已有的 `SeqLLM`）：
```python
def test_e2e_through_subtitles(tmp_path):
    import json
    from src.util.jobs import read_json
    topic = json.dumps({"title_working": "罗斯威尔", "year": 1947})
    research = json.dumps({"facts": [
        {"claim": "a", "evidence": "documented", "source": "s"},
        {"claim": "b", "evidence": "documented", "source": "s"},
        {"claim": "c", "evidence": "documented", "source": "s"},
    ]})
    script = json.dumps({"title_options": ["t"], "segments": [
        {"id": 1, "text": "飞碟来了。", "visual_query": "q", "est_sec": 3},
    ]})
    cfg = Config({"tts": {"provider": "fake"}})
    run_mod.run_pipeline("roswell-1947", cfg, tmp_path, SeqLLM([topic, research, script]))
    job = tmp_path / "roswell-1947"
    assert read_json(job, "timing.json")["segments"][0]["words"][0]["text"] == "飞"
    assert "Dialogue:" in (job / "subs.ass").read_text(encoding="utf-8")
    assert (job / "audio" / "seg_001.mp3").exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_run.py::test_e2e_through_subtitles -q`
Expected: FAIL（STAGES 无 s4/s5，timing.json/subs.ass 未生成）

- [ ] **Step 3: 注册 s4/s5 到 `src/run.py`**

import 区改为：
```python
from src.stages import s1_topic, s2_research, s3_script, s4_tts, s5_subs
```
`STAGES` 末尾加两行：
```python
    ("s4_tts", s4_tts.run, s4_tts.OUTPUT),
    ("s5_subs", s5_subs.run, s5_subs.OUTPUT),
```

- [ ] **Step 4: 全量测试**

Run: `.venv/bin/pytest -q`
Expected: 全绿（含五阶段 e2e）。

- [ ] **Step 5: 真实 edge-tts 冒烟（免费、无配额；不需要 Gemini）**

用手写的 script.json 直接跑 s4+s5（真实 edge-tts 联网合成，免费）：
```bash
mkdir -p jobs/_smoke
cat > jobs/_smoke/script.json <<'JSON'
{"title_options":["罗斯威尔之谜"],"segments":[
 {"id":1,"text":"1947年的夏天，新墨西哥州的沙漠里，发生了一件至今没有定论的事。","visual_query":"desert","est_sec":6},
 {"id":2,"text":"军方先说回收了一只飞碟，几小时后又改口说，那只是一只气象气球。","visual_query":"newspaper","est_sec":6}
]}
JSON
.venv/bin/python -m src.run --case _smoke --from s4_tts
echo "=== 音频 ==="; ls -la jobs/_smoke/audio/
echo "=== timing 摘要 ==="; .venv/bin/python -c "import json;d=json.load(open('jobs/_smoke/timing.json'));print('total_sec=',d['total_sec'],'段数=',len(d['segments']),'首段字数=',len(d['segments'][0]['words']))"
echo "=== 字幕前 12 行 ==="; head -12 jobs/_smoke/subs.ass
```
Expected: `[run ] s4_tts … [run ] s5_subs … [done]`；`audio/seg_001.mp3`、`seg_002.mp3` 是**非空真实音频**（几十~上百 KB）；`timing.json.total_sec` > 0；`subs.ass` 有若干 `Dialogue:` 行、中文字幕。可用任意播放器放 mp3 + 挂 ass 验证对轴。

- [ ] **Step 6: Commit**
```bash
git add src/run.py tests/test_run.py
git commit -m "feat(run): 注册阶段④⑤ + 五阶段端到端"
```

---

## Self-Review

**1. Spec 覆盖（P3 范围）：**
- 阶段④ I/O 契约 audio/*.mp3 + timing.json（spec §6④）→ Task 2 ✅
- 阶段⑤ I/O 契约 subs.ass（spec §6⑤）→ Task 3 ✅
- 「文稿+TTS 时间戳直接生成字幕，无需 Whisper」（spec §5 技术红利）→ Task 3（读 timing.json 全局字时间）✅
- 多产物 checkpoint（P2 终审推迟项）→ Task 2 用「timing.json 作 checkpoint」约定解决，**不改编排器** ✅
- 免费档（edge-tts + 无 GPU）→ 全程 ✅

**2. 占位符扫描：** 无 TBD/TODO；每步含完整代码+命令+预期。

**3. 类型一致性：** `TTSEngine.synth -> (bytes, list[dict])`（段内相对）/ `get_tts` / s4 转全局 `start_sec` / `timing.json`{segments:[{start_sec,dur_sec,words:[{text,start_sec,dur_sec}]}],total_sec} / s5 `build_lines`/`ass_time`/`render_ass` 读全局 word start_sec / `OUTPUT`/`INPUT` / `STAGES` 三元组 一致。s4 的 `OUTPUT=timing.json`（最后写）与编排器单产物 skip 对齐。

---

## 后续（P4–P5）

- **P4 成片**：`s6_media`（护栏#2 白名单源 NASA/archive.org/Wikimedia + license 校验 + Ken Burns）→ `s7_assemble`（ffmpeg：图片推拉 + 配音 + 字幕 + BGM 闪避 + 片头尾 → video.mp4）。s6 也是多产物（media/ + credits.json），沿用「credits.json 作 checkpoint」。
- **P5 发布**：`s8_meta`（标题/简介/封面）→ `s9_publish`（护栏#3 YouTube Data API，`auto_upload:true`/`visibility:private`，一次性 OAuth）。
- **LLM 升级（随时可插）**：`get_llm` 的 `claude` 分支，绕开 Gemini 每日配额 + 提升脚本质量。

### P4 必处理输入（来自 P3 终审，写 P4 计划时必须纳入）

1. **【Important】字幕丢标点**：真机 edge-tts 的中文 `WordBoundary` 是词/短语粒度且**丢标点**，s5 目前只读 `timing.json` 拼 `word["text"]` → 真实字幕无逗号/句号，且句读断行分支在真实数据上恒不触发。**修法**：s5 兼读 `script.json`，用 segment 原文 `text` 作字幕串（按词时间戳对齐断行点），保留标点。（spec §6⑤ 本就写的是「script.json + timing.json」，P3 计划当初简化成只读 timing.json，被真机发现推翻。）
2. **【Important】音频↔时间轴漂移**：s4 段 `dur_sec` 只到最后一个词尾，漏了尾静音（真机实测末段欠 ~0.6s）→ `cursor += dur` 使 `timing.json` 全局时间比真实拼接音频**偏早**，40 段会累积到数秒，s7 拼真实 mp3 后字幕越走越超前。**修法**（P4 定夺）：(a) s4 用 `ffprobe`（已装 ffmpeg8）记真实 mp3 时长推进 cursor，让 timing.json 权威；或 (b) s7 给每段音频补尾静音到 `dur_sec`。
3. **【建议】`write_json` 原子化**：`src/util/jobs.py` 现为非原子 `write_text`，若在写「作 checkpoint 的最后产物」（P3 `timing.json` / P4 `credits.json`）时被杀，会留半截文件而 `exists()` 跳过误信。改成临时文件 + `os.replace`，让整个「末产物=checkpoint」策略防崩溃。低成本，P4 加 s6 前做。
4. **【建议】多产物阶段内的 per-artifact skip**：P4 s6 素材下载昂贵，重跑不应全下；考虑段级/文件级跳过（s4 现在重跑会全量重合成，TTS 免费快无所谓，但对 s6 是坏先例）。
5. **【Minor】s4 `id` 处理**：文件名用 `seg.get("id",0)`（缺省 0）而记录用 `seg.get("id")`（→None）不一致，两个无 id 段会都写 `seg_000.mp3` 覆盖；`{"id":null}` 会 `int(None)` 崩。真实 s3 输出恒有整数 id，低风险，可 `enumerate` 统一。
6. **【Minor】ASS 文本转义**：旁白若含 `{ } \` 会被 ASS 当 override 标签；防御性替换 `{}` 更稳。
7. **【Minor】`subs.max_chars` / 字体字号入 config**：s5 读 `subs.max_chars`（默认16）但 config 无该键；字体字号硬编码。加个 `subs:` 块补齐可发现性。
