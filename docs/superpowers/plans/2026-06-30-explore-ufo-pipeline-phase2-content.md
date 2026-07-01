# 探索飞碟与外星人 · P2 内容 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 P1 地基上加阶段②（调研 + 事实核查🛡️，给每条说法标证据等级）和阶段③（写原创中文悬疑脚本），并把编排器契约补上 `blocked` 控制流与 warnings 通道；LLM 层迁到现行 `google-genai`。跑通后一条命令产出可配音的中文脚本。

**Architecture:** 沿用 P1 的模块化 CLI + job 文件夹交接。阶段②读 `topic.json` → LLM 产 `research.json`（facts + 证据等级 + evidence_summary + blocked）；阶段③读 `research.json` → LLM 产 `script.json`（title_options + segments + total_words/est_minutes）。护栏#1 用代码从证据统计**确定性地**判定 `blocked`（无 documented 事实即 block），并抛 `StageBlocked` 让编排器停在该案例。

**Tech Stack:** Python 3.14；`google-genai`（替换已废弃的 `google-generativeai`）；Gemini 2.5-flash；pytest（TDD，单测用 FakeLLM 不联网）。

## Global Constraints

- Python ≥ 3.14；所有新代码 `from __future__ import annotations` + 现代类型。
- **LLM 走 `google-genai`**（`google-genai>=1.0`），模型 `gemini-2.5-flash`（config `llm.gemini_model`，已设）。provider 抽象不变：`LLMProvider.generate(system, prompt) -> str`；单测用 `FakeLLM`，**绝不联网**（`tmp_path` + FakeLLM/SeqLLM）。
- 阶段统一接口 `run(ctx: Ctx) -> None`；每个阶段模块导出 `OUTPUT`（主产物文件名）和 `INPUT`（读上一步产物名）。
- **护栏#1（事实核查）**：证据等级 `documented` > `claim` > `speculation`。`evidence_summary` 由**代码**从 facts 统计（不信任 LLM 自报）。判定：`blocked = (documented == 0)`；若 `0 < documented < guardrails.min_documented_facts`（config，默认 3）则 `ctx.warn(...)` 并在产物里记 `warnings`，但不 block。
- **`StageBlocked`**（`src/errors.py`）：阶段抛出后编排器打印 `[blocked]` 并停止该案例（不再跑后续阶段）。
- **warnings 通道**：`ctx.warn(msg)` 记录非致命警告并即时打印 `[warn]`；编排器结束时汇总条数。
- **DRY**：`extract_json` 提到 `src/util/llmjson.py`，s1/s2/s3 共用（消除 P1 的重复），并补上 P1 缺的 `ValueError` 分支测试。
- UTF-8，JSON `ensure_ascii=False`。

## File Structure

```
新增：
  src/errors.py                 # StageBlocked
  src/util/llmjson.py           # extract_json（共享）
  src/stages/s2_research.py     # 阶段②
  src/stages/s3_script.py       # 阶段③
  tests/test_llmjson.py
  tests/test_s2_research.py
  tests/test_s3_script.py
修改：
  requirements.txt              # google-generativeai → google-genai
  src/llm.py                    # GeminiLLM 迁 google-genai
  src/context.py                # + warnings 字段 + warn()
  src/run.py                    # 捕获 StageBlocked、注册 s2/s3、warnings 汇总
  src/stages/s1_topic.py        # extract_json 改从 util.llmjson 导入
  tests/test_context.py         # + warn 测试
  tests/test_run.py             # + StageBlocked 停止 + 三阶段 e2e
```

---

## Task 1: LLM 层迁移到 `google-genai`

**Files:**
- Modify: `requirements.txt`, `src/llm.py`

**Interfaces:**
- Produces: `GeminiLLM(api_key, model)` 内部改用 `google-genai`；`generate(system, prompt) -> str` 签名与行为不变（`get_llm` / `FakeLLM` / `LLMProvider` 完全不动）。

- [ ] **Step 1: 换依赖 `requirements.txt`**

把 `google-generativeai>=0.8` 这一行改为：
```
google-genai>=1.0
```

- [ ] **Step 2: 装新包**

Run: `.venv/bin/pip install -q "google-genai>=1.0"`
Expected: 安装成功。

- [ ] **Step 3: 改 `src/llm.py` 的 `GeminiLLM`（其余不动）**

把 `GeminiLLM` 整个类替换为：
```python
class GeminiLLM:
    """Gemini（google-genai）。惰性 import，使单测无需 SDK / 网络。"""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, system: str, prompt: str) -> str:
        from google.genai import types

        resp = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system),
        )
        return resp.text or ""
```

- [ ] **Step 4: 单测不回归**

Run: `.venv/bin/pytest -q`
Expected: 全绿（FakeLLM/get_llm 分支不受影响）。

- [ ] **Step 5: 真实联网冒烟（.env 已有可用 key）——确认 deprecation warning 消失**

Run:
```bash
.venv/bin/python -m src.run --case roswell-1947 --force 2>&1 | tee /tmp/p2t1.log
grep -i "FutureWarning\|google.generativeai" /tmp/p2t1.log && echo "!! 仍有废弃警告" || echo "OK: 无废弃警告"
```
Expected: 打印 `[done] s1_topic`，且最后一行 `OK: 无废弃警告`。若 `google.genai` 的 `generate_content`/`GenerateContentConfig` 签名与上面不符，按实际报错调整（这是集成步骤，以真实 SDK 为准），改完重跑本步。

- [ ] **Step 6: Commit**
```bash
git add requirements.txt src/llm.py
git commit -m "refactor(llm): 迁移到 google-genai，消除废弃警告"
```

---

## Task 2: 编排器契约扩展 —— StageBlocked + warnings + 共享 extract_json

**Files:**
- Create: `src/errors.py`, `src/util/llmjson.py`, `tests/test_llmjson.py`
- Modify: `src/context.py`, `src/run.py`, `src/stages/s1_topic.py`, `tests/test_context.py`, `tests/test_run.py`

**Interfaces:**
- Produces:
  - `class StageBlocked(Exception)` in `src/errors.py`。
  - `extract_json(text: str) -> dict` in `src/util/llmjson.py`（首个`{`到末个`}`切片 + `json.loads`；无则 `ValueError`）。
  - `Ctx.warnings: list[str]` + `Ctx.warn(msg: str) -> None`（append + 即时 `print("[warn] ...")`）。
  - `run_pipeline` 捕获 `StageBlocked` → 打印 `[blocked] {name}: {e}` 并 `break`；循环结束若有 warnings 打印汇总。
- Consumes: `Ctx`（P1）。

- [ ] **Step 1: 写 `src/errors.py`**
```python
from __future__ import annotations


class StageBlocked(Exception):
    """阶段判定该案例不应继续（如证据不足）。编排器捕获后停止本案例。"""
```

- [ ] **Step 2: 写失败测试 `tests/test_llmjson.py`（含 P1 缺的 ValueError 分支）**
```python
import pytest
from src.util.llmjson import extract_json


def test_extract_json_bare():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('好的：{"a": 2} 完') == {"a": 2}


def test_extract_json_no_object_raises():
    with pytest.raises(ValueError):
        extract_json("这里没有 JSON 对象")
```

- [ ] **Step 3: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_llmjson.py -q`
Expected: FAIL（`No module named 'src.util.llmjson'`）

- [ ] **Step 4: 写 `src/util/llmjson.py`**
```python
from __future__ import annotations

import json


def extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"LLM 输出里没有 JSON 对象: {text[:200]!r}")
    return json.loads(text[start : end + 1])
```

- [ ] **Step 5: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_llmjson.py -q`
Expected: `3 passed`

- [ ] **Step 6: 让 `src/stages/s1_topic.py` 复用共享 extract_json（删本地副本）**

在 `s1_topic.py` 中删除本地的 `def extract_json(...)` 整个函数，并在文件顶部 import 区加入：
```python
from src.util.llmjson import extract_json
```
（保持 `extract_json` 仍可从 `s1_topic` 导入，`tests/test_s1_topic.py` 的 `from src.stages.s1_topic import ... extract_json ...` 不受影响。）

- [ ] **Step 7: s1 回归**

Run: `.venv/bin/pytest tests/test_s1_topic.py -q`
Expected: `4 passed`

- [ ] **Step 8: Ctx 加 warnings —— 先改 `tests/test_context.py` 加测试**

在 `tests/test_context.py` 末尾追加：
```python
def test_ctx_warn_collects(tmp_path):
    from src.config import Config
    from src.llm import FakeLLM
    ctx = Ctx(slug="x", config=Config({}), root=tmp_path, llm=FakeLLM())
    assert ctx.warnings == []
    ctx.warn("something")
    assert ctx.warnings == ["something"]
```

- [ ] **Step 9: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_context.py -q`
Expected: FAIL（`AttributeError: ... 'warnings'` / `'warn'`）

- [ ] **Step 10: 改 `src/context.py`**

替换整个文件为：
```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config import Config
from src.llm import LLMProvider


@dataclass
class Ctx:
    slug: str
    config: Config
    root: Path
    llm: LLMProvider
    warnings: list[str] = field(default_factory=list)

    @property
    def job(self) -> Path:
        return self.root / self.slug

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"[warn] {msg}")
```

- [ ] **Step 11: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_context.py -q`
Expected: `2 passed`

- [ ] **Step 12: run_pipeline 捕获 StageBlocked + warnings 汇总 —— 先加测试 `tests/test_run.py`**

在 `tests/test_run.py` 顶部 import 区加 `from src.errors import StageBlocked`，并追加：
```python
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
```

- [ ] **Step 13: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_run.py -q`
Expected: FAIL（StageBlocked 未被捕获，异常抛出）

- [ ] **Step 14: 改 `src/run.py` 的 `run_pipeline`（其余不动）**

在 import 区加 `from src.errors import StageBlocked`，把 `run_pipeline` 的循环体替换为：
```python
    for name, fn, output in STAGES[i0 : i1 + 1]:
        if not force and exists(ctx.job, output):
            print(f"[skip] {name} ({output} 已存在)")
            continue
        print(f"[run ] {name}")
        try:
            fn(ctx)
        except StageBlocked as e:
            print(f"[blocked] {name}: {e}")
            break
        print(f"[done] {name} -> {output}")
    if ctx.warnings:
        print(f"[warnings] 本次共 {len(ctx.warnings)} 条")
```

- [ ] **Step 15: 全量测试**

Run: `.venv/bin/pytest -q`
Expected: 全绿。

- [ ] **Step 16: Commit**
```bash
git add src/errors.py src/util/llmjson.py src/context.py src/run.py src/stages/s1_topic.py tests/test_llmjson.py tests/test_context.py tests/test_run.py
git commit -m "feat(pipeline): StageBlocked + warnings 通道 + 共享 extract_json"
```

---

## Task 3: 阶段② 调研 + 事实核查 🛡️ 护栏#1

**Files:**
- Create: `src/stages/s2_research.py`, `tests/test_s2_research.py`

**Interfaces:**
- Consumes: `Ctx`、`read_json`/`write_json`（P1）、`extract_json`（Task 2）、`StageBlocked`（Task 2）、`ctx.warn`（Task 2）、`ctx.config.get("guardrails.min_documented_facts", 3)`。
- Produces: `OUTPUT="research.json"`, `INPUT="topic.json"`, `run(ctx)`, 辅助 `summarize(facts)`, `build_prompt(topic)`。

- [ ] **Step 1: 写失败测试 `tests/test_s2_research.py`**
```python
import json
import pytest
from pathlib import Path
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.errors import StageBlocked
from src.util.jobs import write_json, read_json
from src.stages.s2_research import run, summarize, OUTPUT, INPUT


def _ctx(tmp_path, canned, cfg=None):
    ctx = Ctx(slug="c", config=cfg or Config({}), root=tmp_path,
              llm=FakeLLM(canned=canned))
    write_json(ctx.job, INPUT, {"slug": "c", "title_working": "案例"})
    return ctx


def test_summarize_counts():
    facts = [{"evidence": "documented"}, {"evidence": "documented"},
             {"evidence": "claim"}, {"evidence": "speculation"}, {"evidence": "x"}]
    assert summarize(facts) == {"documented": 2, "claim": 1, "speculation": 1}


def test_run_normal_writes_research(tmp_path):
    canned = json.dumps({"facts": [
        {"claim": "军方发过飞碟新闻稿", "evidence": "documented", "source": "存档"},
        {"claim": "Mogul 气球计划", "evidence": "documented", "source": "报告"},
        {"claim": "有非人类残骸", "evidence": "claim", "source": "证词"},
    ]})
    ctx = _ctx(tmp_path, canned)
    run(ctx)
    out = read_json(ctx.job, OUTPUT)
    assert out["blocked"] is False
    assert out["evidence_summary"] == {"documented": 2, "claim": 1, "speculation": 0}


def test_run_blocked_when_no_documented(tmp_path):
    canned = json.dumps({"facts": [
        {"claim": "据说有外星人", "evidence": "claim", "source": "论坛"},
    ]})
    ctx = _ctx(tmp_path, canned)
    with pytest.raises(StageBlocked):
        run(ctx)
    out = read_json(ctx.job, OUTPUT)   # 产物仍写下（记录为何 blocked）
    assert out["blocked"] is True


def test_run_warns_below_threshold(tmp_path):
    canned = json.dumps({"facts": [
        {"claim": "一条有记录的事实", "evidence": "documented", "source": "存档"},
    ]})
    ctx = _ctx(tmp_path, canned, cfg=Config({"guardrails": {"min_documented_facts": 3}}))
    run(ctx)
    out = read_json(ctx.job, OUTPUT)
    assert out["blocked"] is False
    assert ctx.warnings                 # 有 warning
    assert out.get("warnings")          # 产物里也记了
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_s2_research.py -q`
Expected: FAIL（`No module named 'src.stages.s2_research'`）

- [ ] **Step 3: 写 `src/stages/s2_research.py`**
```python
from __future__ import annotations

import json

from src.context import Ctx
from src.errors import StageBlocked
from src.util.jobs import read_json, write_json
from src.util.llmjson import extract_json

OUTPUT = "research.json"
INPUT = "topic.json"
LEVELS = ("documented", "claim", "speculation")

SYSTEM = (
    "你是「探索飞碟与外星人」频道的事实核查研究员。针对给定 UFO 案例列出关键事实，"
    "并为每条标注证据等级：documented(有档案/记录) / claim(目击或证词) / speculation(推测)。"
    "只输出一个 JSON 对象，不要额外文字。"
)


def build_prompt(topic: dict) -> str:
    return (
        "为下面的案例做事实核查，输出 JSON，字段 facts：数组，每个元素 "
        "{claim(中文,该说法), evidence(documented|claim|speculation), source(来源,中文)}。"
        "尽量给出可查证的 documented 事实。\n"
        f"案例：{json.dumps(topic, ensure_ascii=False)}"
    )


def summarize(facts: list[dict]) -> dict:
    out = {lv: 0 for lv in LEVELS}
    for f in facts:
        lv = f.get("evidence")
        if lv in out:
            out[lv] += 1
    return out


def run(ctx: Ctx) -> None:
    topic = read_json(ctx.job, INPUT)
    research = extract_json(ctx.llm.generate(SYSTEM, build_prompt(topic)))
    facts = research.get("facts", [])
    summary = summarize(facts)
    research["evidence_summary"] = summary
    documented = summary["documented"]
    min_doc = ctx.config.get("guardrails.min_documented_facts", 3)
    blocked = documented == 0
    research["blocked"] = blocked
    if not blocked and documented < min_doc:
        ctx.warn(f"{ctx.slug}: documented 事实 {documented} 条 < 阈值 {min_doc}")
        research.setdefault("warnings", []).append(
            f"documented={documented} < {min_doc}"
        )
    write_json(ctx.job, OUTPUT, research)
    if blocked:
        raise StageBlocked(f"{ctx.slug}: 无 documented 事实，证据不足，停止本案例")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_s2_research.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**
```bash
git add src/stages/s2_research.py tests/test_s2_research.py
git commit -m "feat(s2): 阶段② 调研+事实核查（证据等级/blocked/warn）"
```

---

## Task 4: 阶段③ 写脚本

**Files:**
- Create: `src/stages/s3_script.py`, `tests/test_s3_script.py`

**Interfaces:**
- Consumes: `Ctx`、`read_json`/`write_json`、`extract_json`、`StageBlocked`。
- Produces: `OUTPUT="script.json"`, `INPUT="research.json"`, `run(ctx)`, `build_prompt(research)`。`total_words`/`est_minutes` 由代码计算。

- [ ] **Step 1: 写失败测试 `tests/test_s3_script.py`**
```python
import json
import pytest
from pathlib import Path
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.errors import StageBlocked
from src.util.jobs import write_json, read_json
from src.stages.s3_script import run, OUTPUT, INPUT


def _ctx(tmp_path, canned, research):
    ctx = Ctx(slug="c", config=Config({}), root=tmp_path, llm=FakeLLM(canned=canned))
    write_json(ctx.job, INPUT, research)
    return ctx


def test_run_writes_script_and_computes_stats(tmp_path):
    canned = json.dumps({
        "title_options": ["标题A", "标题B", "标题C"],
        "segments": [
            {"id": 1, "text": "一二三四五", "visual_query": "desert night", "est_sec": 5},
            {"id": 2, "text": "六七八九十", "visual_query": "old photo", "est_sec": 5},
        ],
    })
    ctx = _ctx(tmp_path, canned, {"blocked": False, "facts": []})
    run(ctx)
    out = read_json(ctx.job, OUTPUT)
    assert len(out["segments"]) == 2
    assert out["total_words"] == 10           # 5 + 5 个字
    assert out["est_minutes"] == round(10 / 300, 1)


def test_run_refuses_blocked_research(tmp_path):
    ctx = _ctx(tmp_path, "{}", {"blocked": True, "facts": []})
    with pytest.raises(StageBlocked):
        run(ctx)
    assert not (ctx.job / OUTPUT).exists()     # 不生成脚本
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_s3_script.py -q`
Expected: FAIL（`No module named 'src.stages.s3_script'`）

- [ ] **Step 3: 写 `src/stages/s3_script.py`**
```python
from __future__ import annotations

import json

from src.context import Ctx
from src.errors import StageBlocked
from src.util.jobs import read_json, write_json
from src.util.llmjson import extract_json

OUTPUT = "script.json"
INPUT = "research.json"
CPM = 300  # 中文旁白约 300 字/分钟

SYSTEM = (
    "你是「探索飞碟与外星人」频道的中文脚本作者，写 8–20 分钟的悬疑解说。"
    "严格区分证据等级：documented 当事实讲；claim 用「目击者称/据称」；"
    "speculation 用「有推测认为」。只输出一个 JSON 对象，不要额外文字。"
)


def build_prompt(research: dict) -> str:
    return (
        "根据以下核查过的事实写一篇原创中文悬疑解说脚本，输出 JSON，字段："
        "title_options(3个中文标题字符串数组)，"
        "segments(数组，每段 {id(整数), text(中文旁白), visual_query(英文画面搜索词), est_sec(整数秒)})。"
        "结构：钩子→背景→事件经过→疑点→各方解释→开放式收尾+订阅引导。\n"
        f"事实：{json.dumps(research, ensure_ascii=False)}"
    )


def run(ctx: Ctx) -> None:
    research = read_json(ctx.job, INPUT)
    if research.get("blocked"):
        raise StageBlocked(f"{ctx.slug}: research.blocked=true，不生成脚本")
    script = extract_json(ctx.llm.generate(SYSTEM, build_prompt(research)))
    segments = script.get("segments", [])
    total_words = sum(len(s.get("text", "")) for s in segments)
    script["total_words"] = total_words
    script["est_minutes"] = round(total_words / CPM, 1)
    write_json(ctx.job, OUTPUT, script)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_s3_script.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**
```bash
git add src/stages/s3_script.py tests/test_s3_script.py
git commit -m "feat(s3): 阶段③ 原创悬疑脚本（分段+画面提示+字数/时长）"
```

---

## Task 5: 注册阶段 + 端到端 + 真实冒烟

**Files:**
- Modify: `src/run.py`, `tests/test_run.py`

**Interfaces:**
- Consumes: `s2_research`, `s3_script`（Task 3/4）。
- Produces: `STAGES` 增加 s2/s3 两行；三阶段 e2e 测试（SeqLLM 顺序返回）。

- [ ] **Step 1: 加 e2e 测试到 `tests/test_run.py`**

在文件末尾追加（`SeqLLM` 按调用顺序返回不同 canned，模拟 s1→s2→s3 各一次 LLM 调用）：
```python
class SeqLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.i = 0

    def generate(self, system, prompt):
        r = self.responses[self.i]
        self.i += 1
        return r


def test_end_to_end_topic_research_script(tmp_path):
    import json
    from src.util.jobs import read_json
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
                         SeqLLM([topic, research, script]))
    job = tmp_path / "roswell-1947"
    assert read_json(job, "topic.json")["year"] == 1947
    assert read_json(job, "research.json")["blocked"] is False
    assert read_json(job, "script.json")["total_words"] == 10
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_run.py::test_end_to_end_topic_research_script -q`
Expected: FAIL（STAGES 只有 s1_topic，research.json/script.json 未生成）

- [ ] **Step 3: 注册 s2/s3 到 `src/run.py` 的 STAGES**

import 区加：
```python
from src.stages import s1_topic, s2_research, s3_script
```
把 `STAGES` 改为：
```python
STAGES: list[tuple[str, Callable[[Ctx], None], str]] = [
    ("s1_topic", s1_topic.run, s1_topic.OUTPUT),
    ("s2_research", s2_research.run, s2_research.OUTPUT),
    ("s3_script", s3_script.run, s3_script.OUTPUT),
]
```

- [ ] **Step 4: 跑 e2e + 全量**

Run: `.venv/bin/pytest -q`
Expected: 全绿（含新 e2e）。

- [ ] **Step 5: 真实联网冒烟（Gemini，需 .env 里的 key）**

Run:
```bash
rm -rf jobs/roswell-1947
.venv/bin/python -m src.run --case roswell-1947
echo "=== research.json ==="; cat jobs/roswell-1947/research.json
echo "=== script.json (前 60 行) ==="; head -60 jobs/roswell-1947/script.json
```
Expected: 依次 `[run ] s1_topic … s2_research … s3_script … [done]`；`research.json` 有 facts + evidence_summary + `blocked:false`；`script.json` 有 title_options + segments + total_words/est_minutes。若 `[blocked]`（真实核查判定证据不足）也是**正确行为**——换个 documented 充足的案例（如 `nimitz-tictac-2004`）再验。

- [ ] **Step 6: Commit**
```bash
git add src/run.py tests/test_run.py
git commit -m "feat(run): 注册阶段②③ + 三阶段端到端"
```

---

## Self-Review

**1. Spec 覆盖（P2 范围）：**
- 阶段② I/O 契约 research.json（spec §6②）→ Task 3 ✅
- 阶段③ I/O 契约 script.json（spec §6③）→ Task 4 ✅
- 护栏#1 证据等级 + blocked（spec §7 护栏#1）→ Task 3（代码确定性判定）✅
- 编排器 blocked 控制流 + warnings 通道（P1 终审输入）→ Task 2 ✅
- LLM 迁 google-genai（P1 终审输入）→ Task 1 ✅
- DRY extract_json + 补 ValueError 测试（P1 终审 Minor）→ Task 2 ✅
- 多产物 checkpoint（P1 终审输入）→ **本阶段不需要**（②③各单产物）；**推迟到 P3**（s4 audio/ + timing.json 时再做），届时把 STAGES 的第三元素从「单文件名」升级为「判断函数」。

**2. 占位符扫描：** 无 TBD/TODO；每步含完整代码+命令+预期。

**3. 类型一致性：** `StageBlocked` / `extract_json`(util.llmjson) / `Ctx.warn`+`warnings` / `summarize` / `OUTPUT`/`INPUT` / `run(ctx)` / `STAGES`(三元组) / `SeqLLM` 在 Task 间一致；`run_pipeline` 捕获 `StageBlocked` 与 Task 3/4 抛出点对齐；s3 读 `research.blocked` 与 s2 写入字段对齐。

---

## 后续（P3–P5，本阶段后逐个写）

- **P3 音字**：`s4_tts`（edge-tts + WordBoundary 时间戳，**首个多产物阶段** → 届时升级 STAGES 的 skip 判定）→ `s5_subs`（`.ass`，无需 Whisper）。
- **P4 成片**：`s6_media`（护栏#2 白名单源 + license 校验 + Ken Burns）→ `s7_assemble`（ffmpeg）。
- **P5 发布**：`s8_meta`（标题/简介/封面）→ `s9_publish`（护栏#3 YouTube Data API，`auto_upload:true` / `visibility:private` 安全档；一次性 OAuth）。
- **LLM 升级（随时可插）**：`get_llm` 的 `claude` 分支（Anthropic SDK，`claude-opus-4-8`/`claude-sonnet-4-6`，可挂 `web_search` 让 s2 联网核查带来源），把 s2/s3 切到 Claude。
