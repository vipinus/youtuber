# 探索飞碟与外星人 · P4a 精度修补 + 素材 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 P4b 合成做好铺垫：(1) 修掉 P3 终审发现的两个真机问题（字幕丢标点、音频时间轴漂移）；(2) 硬化 checkpoint 写入；(3) 加合法素材抽象与阶段⑥（护栏#2），产出 `media/` + `credits.json`。

**Architecture:** 沿用模块化 CLI。素材引擎抽象成 provider（`nasa`/`fake`），与 LLM/TTS 同构，单测离线。护栏#2 = **白名单源 + license 门**（只收 PD/CC）。阶段⑥ `OUTPUT="credits.json"`（最后写，作 checkpoint）；素材落在 `media/`。时间轴以**真实音频时长（ffprobe）**为准，消除漂移。

**Tech Stack:** Python 3.14；`requests`（NASA Images API）；`ffprobe`（真实音频时长，ffmpeg8 已装）；pytest（TDD，单测离线：FakeTTS 产合法 WAV、FakeMedia 本地写字节）。

## Global Constraints

- Python ≥ 3.14；`from __future__ import annotations`；UTF-8，JSON `ensure_ascii=False`。
- **单测绝不联网**：素材用 `FakeMedia`（provider=fake），TTS 用 `FakeTTS`（provider=fake）。
- 阶段接口 `run(ctx)->None` + 模块 `OUTPUT`/`INPUT`；阶段⑥多产物用「`credits.json` 最后写 = checkpoint」。
- **护栏#2**：素材来源**白名单制**（provider 只允许 `nasa`/`fake`），且**逐个校验 license** 仅收 `config.guardrails.allowed_licenses`（默认 `[public-domain, cc0, cc-by, cc-by-sa]`）；无合法素材时用**我方控制的兜底 PD 图** + `ctx.warn`。
- **时间轴单一真源**：段时长 = 真实音频文件时长（ffprobe），不再用「末词尾」估算。为使单测离线可探测，`FakeTTS` 产出**合法 WAV**（时长 = 字数×0.3s）。
- **checkpoint 防崩溃**：`write_json` 原子化（临时文件 + `os.replace`）。

## File Structure

```
新增：
  src/util/ffprobe.py           # duration_sec(path) via ffprobe
  src/media.py                  # MediaSource 协议 / FakeMedia / NASAMedia / get_media_source
  src/stages/s6_media.py        # 阶段⑥ 取合法素材
  tests/test_ffprobe.py
  tests/test_media.py
  tests/test_s6_media.py
修改：
  src/util/jobs.py              # write_json 原子化
  src/tts.py                    # FakeTTS 产合法 WAV
  src/stages/s4_tts.py          # 段时长改用 ffprobe（修漂移）
  src/stages/s5_subs.py         # 兼读 script.json，用原文保标点
  src/run.py                    # 注册 s6
  config.yaml                   # media.provider + paths.fallback_image
  tests/test_jobs.py            # + 原子写（无残留 temp）
  tests/test_tts.py             # FakeTTS WAV 断言
  tests/test_s5_subs.py         # 保标点用例
  tests/test_run.py             # 六阶段 e2e（+ FakeMedia）
```

---

## Task 1: 原子 write_json + ffprobe util + FakeTTS 合法 WAV

**Files:**
- Create: `src/util/ffprobe.py`, `tests/test_ffprobe.py`
- Modify: `src/util/jobs.py`, `src/tts.py`, `tests/test_jobs.py`, `tests/test_tts.py`

**Interfaces:**
- Produces: `duration_sec(path) -> float`（ffprobe）；`write_json` 原子化（同签名）；`FakeTTS.synth` 返回合法 WAV 字节（`b"RIFF"...`），words 不变。

- [ ] **Step 1: 写失败测试 `tests/test_ffprobe.py`**（用 ffmpeg 生成 1s 音再探测，纯本地无网）
```python
import subprocess
from src.util.ffprobe import duration_sec


def test_duration_sec(tmp_path):
    wav = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         str(wav)], check=True,
    )
    assert abs(duration_sec(wav) - 1.0) < 0.1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_ffprobe.py -q`
Expected: FAIL（`No module named 'src.util.ffprobe'`）

- [ ] **Step 3: 写 `src/util/ffprobe.py`**
```python
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def duration_sec(path: str | Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_ffprobe.py -q`
Expected: `1 passed`

- [ ] **Step 5: write_json 原子化 —— 先加测试到 `tests/test_jobs.py`**

在 `tests/test_jobs.py` 末尾追加：
```python
def test_write_json_atomic_no_temp_left(tmp_path):
    from src.util.jobs import write_json
    job = tmp_path / "x"
    write_json(job, "a.json", {"k": 1})
    names = sorted(p.name for p in job.iterdir())
    assert names == ["a.json"]        # 无残留临时文件
```

- [ ] **Step 6: 跑测试确认失败/通过基线**

Run: `.venv/bin/pytest tests/test_jobs.py -q`
Expected: 若当前实现已恰好只留目标文件则可能已过；无论如何 Step 7 后必须过。

- [ ] **Step 7: 改 `src/util/jobs.py` 的 `write_json` 为原子写**
```python
def write_json(job: Path, name: str, data: dict) -> Path:
    import os
    import tempfile

    job.mkdir(parents=True, exist_ok=True)
    p = job / name
    fd, tmp = tempfile.mkstemp(dir=job, prefix=f".{name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))
        os.replace(tmp, p)          # 原子替换
    except BaseException:
        os.unlink(tmp)
        raise
    return p
```
（保留文件顶部已有的 `import json`；其余函数不动。）

- [ ] **Step 8: FakeTTS 产合法 WAV —— 先改 `tests/test_tts.py`**

把 `test_fake_tts_deterministic` 里对 `data` 的断言从「非空」加强为「合法 WAV 头」：在该测试中，把 `assert data` 那行改为：
```python
    assert data[:4] == b"RIFF"        # 合法 WAV，ffprobe 可读
```
（words 相关断言保持不变。）

- [ ] **Step 9: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_tts.py -q`
Expected: FAIL（当前 FakeTTS 返回 `b"FAKEAUDIO"`，非 RIFF）

- [ ] **Step 10: 改 `src/tts.py` 的 `FakeTTS.synth` 产 WAV**

把 `FakeTTS` 类替换为：
```python
class FakeTTS:
    """确定性离线 TTS：每字 0.3s；音频为合法 WAV（8kHz 8-bit 静音），使 ffprobe 可读。"""

    def synth(self, text: str) -> tuple[bytes, list[dict]]:
        import struct

        words = []
        t = 0.0
        for ch in text:
            words.append({"text": ch, "offset_sec": round(t, 3), "dur_sec": 0.3})
            t += 0.3
        rate = 8000
        n = int(len(text) * 0.3 * rate)         # 时长 = 字数×0.3s
        pcm = b"\x00" * n
        header = (
            b"RIFF" + struct.pack("<I", 36 + n) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate, 1, 8)
            + b"data" + struct.pack("<I", n)
        )
        return header + pcm, words
```

- [ ] **Step 11: 跑测试确认通过 + 全量**

Run: `.venv/bin/pytest -q`
Expected: 全绿。

- [ ] **Step 12: Commit**
```bash
git add src/util/ffprobe.py src/util/jobs.py src/tts.py tests/test_ffprobe.py tests/test_jobs.py tests/test_tts.py
git commit -m "feat(util): ffprobe 时长 + write_json 原子化 + FakeTTS 合法 WAV"
```

---

## Task 2: s4 用真实时长修漂移 + s5 保标点

**Files:**
- Modify: `src/stages/s4_tts.py`, `src/stages/s5_subs.py`, `tests/test_s5_subs.py`

**Interfaces:**
- Consumes: `duration_sec`（Task 1）、`read_json`。
- Produces: s4 段 `dur_sec` = `duration_sec(该 mp3)`（真源，修漂移）；s5 用 `script.json` 原文按词时间戳对齐断行、**保标点**。

- [ ] **Step 1: 改 `src/stages/s4_tts.py` —— dur 用 ffprobe**

在 import 区加 `from src.util.ffprobe import duration_sec`。把段时长那行
```python
        dur = (words[-1]["offset_sec"] + words[-1]["dur_sec"]) if words else 0.0
```
改为
```python
        dur = round(duration_sec(audio_dir / name), 3)   # 真实音频时长（含尾静音），修时间轴漂移
```
（其余不动。注：FakeTTS 的 WAV 时长 = 字数×0.3，与旧值一致，故 s4 现有测试数值不变。）

- [ ] **Step 2: s4 回归**

Run: `.venv/bin/pytest tests/test_s4_tts.py -q`
Expected: `1 passed`（数值不变：0.6 / 0.6 / 1.5）

- [ ] **Step 3: 写 s5 保标点失败测试到 `tests/test_s5_subs.py`**

追加（timing 的 words 丢了标点，但 script 原文有标点；断言字幕含标点）：
```python
def test_run_uses_script_text_with_punctuation(tmp_path):
    from src.config import Config
    from src.llm import FakeLLM
    from src.context import Ctx
    from src.util.jobs import write_json
    from src.stages.s5_subs import run, OUTPUT
    ctx = Ctx(slug="c", config=Config({}), root=tmp_path, llm=FakeLLM())
    # script 原文有标点；timing 的 words 无标点（模拟真机 edge-tts）
    write_json(ctx.job, "script.json", {"segments": [
        {"id": 1, "text": "今天，很好。"},
    ]})
    write_json(ctx.job, "timing.json", {"segments": [
        {"id": 1, "words": [
            {"text": "今天", "start_sec": 0.0, "dur_sec": 0.6},
            {"text": "很好", "start_sec": 0.6, "dur_sec": 0.6},
        ]},
    ]})
    run(ctx)
    txt = (ctx.job / OUTPUT).read_text(encoding="utf-8")
    assert "，" in txt and "。" in txt        # 标点从 script 原文保留
```

- [ ] **Step 4: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_s5_subs.py::test_run_uses_script_text_with_punctuation -q`
Expected: FAIL（当前 s5 只读 timing，字幕无标点）

- [ ] **Step 5: 改 `src/stages/s5_subs.py` —— 用 script 原文的每段文本**

把 `run` 改为读 script.json + timing.json，用每段的原文 `text` 作字幕串、用该段词时间戳定行边界；`build_lines` 改成接收「原文 + 该段词列表」。具体：新增 `seg_lines(text, words, max_chars)` 生成 `(start,end,line)`（按 `max_chars`/句读切 `text`，时间用词时间戳线性映射到字符），`run` 遍历 script 段落配对 timing 段落调用它：
```python
def seg_lines(text, words, max_chars):
    lines = []
    if not words:
        return lines
    start = words[0]["start_sec"]
    end = words[-1]["start_sec"] + words[-1]["dur_sec"]
    span = max(end - start, 0.001)
    n = len(text)
    buf = ""
    line_start_i = 0
    for i, ch in enumerate(text):
        buf += ch
        if len(buf) >= max_chars or ch in SENT_PUNCT or i == n - 1:
            s = start + span * (line_start_i / n)
            e = start + span * ((i + 1) / n)
            lines.append((round(s, 3), round(e, 3), buf))
            buf = ""
            line_start_i = i + 1
    return lines


def run(ctx: Ctx) -> None:
    script = read_json(ctx.job, "script.json")
    timing = read_json(ctx.job, INPUT)
    max_chars = ctx.config.get("subs.max_chars", 16)
    tsegs = {s.get("id"): s for s in timing.get("segments", [])}
    all_lines = []
    for seg in script.get("segments", []):
        words = tsegs.get(seg.get("id"), {}).get("words", [])
        all_lines += seg_lines(seg.get("text", ""), words, max_chars)
    (ctx.job / OUTPUT).write_text(render_ass(all_lines), encoding="utf-8")
```
（保留 `ass_time`/`render_ass`/`build_lines`/`SENT_PUNCT`；`INPUT="timing.json"` 不变；`build_lines` 保留供旧测试用。）

- [ ] **Step 6: 跑测试确认通过 + 全量**

Run: `.venv/bin/pytest -q`
Expected: 全绿（含新保标点用例、旧 build_lines/run 用例）。

- [ ] **Step 7: Commit**
```bash
git add src/stages/s4_tts.py src/stages/s5_subs.py tests/test_s5_subs.py
git commit -m "fix(s4,s5): 时长用ffprobe修漂移 + 字幕用script原文保标点（P3终审）"
```

---

## Task 3: 素材抽象 `src/media.py` + 白名单

**Files:**
- Create: `src/media.py`, `tests/test_media.py`
- Modify: `config.yaml`

**Interfaces:**
- Produces:
  - `MediaSource(Protocol)`: `search(query, limit=5) -> list[dict]`（候选 `{url|manifest, ext, license, attribution, source}`）；`fetch(candidate, dest: Path) -> None`。
  - `FakeMedia`（离线：search 返回一个 public-domain 候选；fetch 写 `b"FAKEIMG"`）。
  - `NASAMedia`（真 NASA Images API；结果 license=public-domain）。
  - `get_media_source(config)`: `fake`→FakeMedia，`nasa`→NASAMedia，else `ValueError`。默认 `nasa`。

- [ ] **Step 1: 写失败测试 `tests/test_media.py`**
```python
import pytest
from pathlib import Path
from src.config import Config
from src.media import FakeMedia, get_media_source


def test_fake_media_search_and_fetch(tmp_path):
    m = FakeMedia()
    cands = m.search("roswell desert")
    assert cands and cands[0]["license"] == "public-domain"
    dest = tmp_path / "x.jpg"
    m.fetch(cands[0], dest)
    assert dest.read_bytes() == b"FAKEIMG"


def test_get_media_source_fake():
    assert isinstance(get_media_source(Config({"media": {"provider": "fake"}})), FakeMedia)


def test_get_media_source_unknown_raises():
    with pytest.raises(ValueError):
        get_media_source(Config({"media": {"provider": "nope"}}))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_media.py -q`
Expected: FAIL（`No module named 'src.media'`）

- [ ] **Step 3: 写 `src/media.py`**
```python
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.config import Config


class MediaSource(Protocol):
    def search(self, query: str, limit: int = 5) -> list[dict]: ...
    def fetch(self, candidate: dict, dest: Path) -> None: ...


class FakeMedia:
    """离线：确定性返回一个 PD 候选，fetch 写占位字节。"""

    def search(self, query: str, limit: int = 5) -> list[dict]:
        return [{
            "manifest": f"fake://{query}", "ext": "jpg",
            "license": "public-domain", "attribution": f"Fake/{query}", "source": "fake",
        }]

    def fetch(self, candidate: dict, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"FAKEIMG")


class NASAMedia:
    """NASA Images API（公有领域）。惰性 import requests。"""

    SEARCH = "https://images-api.nasa.gov/search"

    def search(self, query: str, limit: int = 5) -> list[dict]:
        import requests

        r = requests.get(self.SEARCH, params={"q": query, "media_type": "image"}, timeout=20)
        r.raise_for_status()
        items = r.json().get("collection", {}).get("items", [])[:limit]
        out = []
        for it in items:
            data = (it.get("data") or [{}])[0]
            out.append({
                "manifest": it.get("href", ""), "ext": "jpg",
                "license": "public-domain",
                "attribution": f"NASA / {data.get('title', 'image')}",
                "source": "nasa",
            })
        return [c for c in out if c["manifest"]]

    def fetch(self, candidate: dict, dest: Path) -> None:
        import requests

        man = requests.get(candidate["manifest"], timeout=20)
        man.raise_for_status()
        urls = [u for u in man.json() if isinstance(u, str) and u.lower().endswith(".jpg")]
        url = next((u for u in urls if "~orig" in u), urls[0])
        img = requests.get(url, timeout=30)
        img.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(img.content)


def get_media_source(config: Config) -> MediaSource:
    provider = config.get("media.provider", "nasa")
    if provider == "fake":
        return FakeMedia()
    if provider == "nasa":
        return NASAMedia()
    raise ValueError(f"未知 media provider: {provider}")
```

- [ ] **Step 4: config.yaml 加 media + 兜底图路径**

在 `config.yaml` 顶层加：
```yaml
media:
  provider: nasa            # nasa | fake
```
并在 `paths:` 块加：
```yaml
  fallback_image: assets/fallback.jpg
```

- [ ] **Step 5: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_media.py -q`
Expected: `3 passed`

- [ ] **Step 6: Commit**
```bash
git add src/media.py config.yaml tests/test_media.py
git commit -m "feat(media): 素材 provider 抽象（NASA/Fake）+ 白名单配置"
```

---

## Task 4: 阶段⑥ 取合法素材 `src/stages/s6_media.py`

**Files:**
- Create: `src/stages/s6_media.py`, `tests/test_s6_media.py`

**Interfaces:**
- Consumes: `Ctx`、`read_json`/`write_json`、`get_media_source`（Task 3）、`ctx.warn`、`config.guardrails.allowed_licenses`、`config.paths.fallback_image`。
- Produces: `OUTPUT="credits.json"`, `INPUT="script.json"`, `run(ctx)`。副产物 `media/seg_NNN.jpg`。护栏#2：只用白名单源里 license 合法的候选，否则兜底。

- [ ] **Step 1: 写失败测试 `tests/test_s6_media.py`**
```python
import json
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.util.jobs import write_json, read_json
from src.stages.s6_media import run, OUTPUT, INPUT


def _ctx(tmp_path):
    cfg = Config({"media": {"provider": "fake"}})
    ctx = Ctx(slug="c", config=cfg, root=tmp_path, llm=FakeLLM())
    write_json(ctx.job, INPUT, {"segments": [
        {"id": 1, "visual_query": "roswell desert"},
        {"id": 2, "visual_query": "phoenix lights"},
    ]})
    return ctx


def test_run_fetches_whitelisted_media(tmp_path):
    ctx = _ctx(tmp_path)
    run(ctx)
    credits = read_json(ctx.job, OUTPUT)
    assert len(credits["items"]) == 2
    assert all(c["license"] == "public-domain" for c in credits["items"])
    assert (ctx.job / "media" / "seg_001.jpg").read_bytes() == b"FAKEIMG"
    assert (ctx.job / "media" / "seg_002.jpg").exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_s6_media.py -q`
Expected: FAIL（`No module named 'src.stages.s6_media'`）

- [ ] **Step 3: 写 `src/stages/s6_media.py`**
```python
from __future__ import annotations

from pathlib import Path

from src.context import Ctx
from src.media import get_media_source
from src.util.jobs import read_json, write_json

OUTPUT = "credits.json"
INPUT = "script.json"
DEFAULT_ALLOWED = ["public-domain", "cc0", "cc-by", "cc-by-sa"]


def run(ctx: Ctx) -> None:
    script = read_json(ctx.job, INPUT)
    source = get_media_source(ctx.config)
    allowed = set(ctx.config.get("guardrails.allowed_licenses", DEFAULT_ALLOWED))
    media_dir = ctx.job / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    for seg in script.get("segments", []):
        sid = int(seg.get("id", 0))
        name = f"seg_{sid:03d}.jpg"
        query = seg.get("visual_query", "")
        cands = [c for c in source.search(query) if c.get("license") in allowed]  # 护栏#2
        if cands:
            c = cands[0]
            source.fetch(c, media_dir / name)
            items.append({"id": seg.get("id"), "file": name, "source": c["source"],
                          "license": c["license"], "attribution": c["attribution"]})
        else:
            ctx.warn(f"{ctx.slug} seg {sid}: 无合法素材，用兜底 PD 图")
            fb = Path(ctx.config.get("paths.fallback_image", "assets/fallback.jpg"))
            (media_dir / name).write_bytes(fb.read_bytes() if fb.exists() else b"")
            items.append({"id": seg.get("id"), "file": name, "source": "fallback",
                          "license": "public-domain", "attribution": "fallback"})

    write_json(ctx.job, OUTPUT, {"items": items})
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_s6_media.py -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**
```bash
git add src/stages/s6_media.py tests/test_s6_media.py
git commit -m "feat(s6): 阶段⑥ 取合法素材（白名单+license门+兜底，护栏#2）"
```

---

## Task 5: 注册 s6 + 六阶段 e2e + 真实 NASA 冒烟

**Files:**
- Modify: `src/run.py`, `tests/test_run.py`

**Interfaces:**
- Consumes: `s6_media`。
- Produces: `STAGES` 增加 s6；六阶段离线 e2e（SeqLLM + `tts.provider=fake` + `media.provider=fake`）。

- [ ] **Step 1: 加 e2e 测试到 `tests/test_run.py`**（复用 `SeqLLM`）
```python
def test_e2e_through_media(tmp_path):
    import json
    from src.util.jobs import read_json
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
    run_mod.run_pipeline("roswell-1947", cfg, tmp_path, SeqLLM([topic, research, script]))
    job = tmp_path / "roswell-1947"
    assert read_json(job, "credits.json")["items"][0]["license"] == "public-domain"
    assert (job / "media" / "seg_001.jpg").exists()
    assert (job / "subs.ass").read_text(encoding="utf-8").count("Dialogue:") >= 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_run.py::test_e2e_through_media -q`
Expected: FAIL（STAGES 无 s6，credits.json 未生成）

- [ ] **Step 3: 注册 s6 到 `src/run.py`**

import 区加入 `s6_media`；`STAGES` 在 `s5_subs` 后加：
```python
    ("s6_media", s6_media.run, s6_media.OUTPUT),
```
（注意顺序：s6 读 script.json，与 s4/s5 无依赖，放 s5 之后即可；credits.json 供 P4b 的 s7 用。）

- [ ] **Step 4: 全量测试**

Run: `.venv/bin/pytest -q`
Expected: 全绿（含六阶段 e2e）。

> ⚠️ 检查：像 P3-T4 那样，确认没有用默认 provider 的真实管线测试触网 —— 所有跑真实 STAGES 的测试都要带 `tts.provider=fake` + `media.provider=fake`（或 `to=` 提前收尾）。若发现 `test_e2e_through_subtitles` 之类未带 `media.provider=fake` 而现在会跑到 s6→真实 NASA，给它补上 `media.provider=fake`。

- [ ] **Step 5: 真实 NASA 冒烟（免费，无 key）**

用手写 script.json 只跑 s6（真实 NASA）：
```bash
mkdir -p jobs/_smoke6
cat > jobs/_smoke6/script.json <<'JSON'
{"segments":[{"id":1,"visual_query":"roswell new mexico 1947","est_sec":6},
             {"id":2,"visual_query":"unidentified flying object sky","est_sec":6}]}
JSON
.venv/bin/python -c "
from pathlib import Path
from src.config import load_config
from src.llm import FakeLLM
from src.context import Ctx
from src.stages.s6_media import run
cfg = load_config('config.yaml')
run(Ctx('_smoke6', cfg, Path('jobs'), FakeLLM()))
"
echo "=== 素材 ==="; ls -la jobs/_smoke6/media/
echo "=== credits ==="; cat jobs/_smoke6/credits.json
echo "=== 校验是图片 ==="; file jobs/_smoke6/media/seg_001.jpg
```
Expected: `media/seg_001.jpg`、`seg_002.jpg` 是真实 NASA 图片（`file` 报 JPEG，非空）；`credits.json` 每项 `license=public-domain`、`source=nasa`、带 attribution。若某段无匹配则该项 `source=fallback` 并有 warn（此时需先放一张 `assets/fallback.jpg`）。

- [ ] **Step 6: Commit**
```bash
git add src/run.py tests/test_run.py
git commit -m "feat(run): 注册阶段⑥ + 六阶段端到端"
```

---

## Self-Review

**1. Spec 覆盖（P4a 范围）：**
- 阶段⑥ credits.json + media/（spec §6⑥）→ Task 4 ✅
- 护栏#2 白名单 + license 门（spec §7#2）→ Task 3/4（provider 白名单 + allowed_licenses 过滤 + 兜底 PD）✅
- P3 终审 Important #1 字幕保标点 → Task 2（s5 读 script 原文）✅
- P3 终审 Important #2 时间轴漂移 → Task 2（s4 ffprobe 真时长）✅
- P3 终审 建议 原子写 → Task 1 ✅
- 离线测试纪律（provider=fake）+ 防 P3-T4 那类触网回归 → Task 5 Step 4 ⚠️ 检查 ✅

**2. 占位符扫描：** 无 TBD；每步含完整代码+命令+预期。archive.org/Wikimedia 源为**有意的后续升级**（MVP 仅 NASA，架构已支持扩展），非漏洞。

**3. 类型一致性：** `duration_sec` / 原子 `write_json` / `FakeTTS`→WAV / s4 `dur` 用 ffprobe / s5 `seg_lines`(读 script+timing) / `MediaSource.search/fetch` / `FakeMedia`/`NASAMedia` / `get_media_source` / s6 `OUTPUT=credits.json`/`INPUT=script.json` / `STAGES` 三元组 一致。

---

## 后续

- **P4b 合成**：`s7_assemble`（ffmpeg：每段图 Ken Burns `zoompan` 按段真实音频时长展示 → 拼接；叠 `subs.ass`；混全音轨；片头尾）→ `video.mp4`。用 credits.json 的 attribution 上字幕/简介署名。命令构造单测 + 真实 ffmpeg 冒烟出真 mp4。
- **兜底图**：P4b 前放一张我方生成的 `assets/fallback.jpg`（Pillow 星空图，PD 自制），供无匹配段使用。
- **素材源扩展**：archive.org / Wikimedia Commons（含 license 解析）作 `MediaSource` 新 provider。
- **P5 发布**：s8 元数据/封面 → s9 YouTube 上传（护栏#3）。
- **LLM 升级**：`claude` 分支绕开 Gemini 配额。
