# 探索飞碟与外星人 · P1 地基 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭好整条出片管线的地基 —— 项目骨架、配置、可换的 LLM provider、可断点续跑的编排器，并用阶段①（选题）作为打通骨架的第一个真实阶段。

**Architecture:** 模块化 CLI。`src/run.py` 编排器按注册顺序调用各阶段；阶段间通过 `jobs/<slug>/` 里的 JSON 文件交接（不共享内存）。每个阶段是 `run(ctx: Ctx) -> None`，从 job 文件夹读上一步产物、写自己的产物。编排器靠「产物文件是否已存在」实现跳过/续跑。

**Tech Stack:** Python 3.14、PyYAML（配置/种子库）、pytest（TDD）、Gemini 免费额度（真实 LLM，单测用 FakeLLM 不联网）。

## Global Constraints

- Python ≥ 3.14（系统已装）；所有新代码用 `from __future__ import annotations` + `str | None` 风格。
- **成本档 = free**：默认 LLM provider = `gemini`（免费额度）；`claude` provider 是后续升级位，本计划中只留 `NotImplementedError` 占位。
- **密钥永不进 git**：`GEMINI_API_KEY` 走环境变量 / `.env`；`.env`、`secrets/`、`jobs/` 已在 `.gitignore` 中。
- **每集一个 `jobs/<slug>/` 文件夹**，所有中间产物落在里面；编排器幂等（产物已存在则跳过，除非 `--force`）。
- 阶段统一接口：`def run(ctx: Ctx) -> None`，且导出常量 `OUTPUT`（本阶段主产物文件名，供编排器判断跳过）。
- 测试不打真实网络：LLM 走 `FakeLLM`，文件路径走 `tmp_path`。
- 编码统一 UTF-8，JSON 写入 `ensure_ascii=False`（中文可读）。

---

## File Structure

```
youtuber/
├── pyproject.toml              # 仅 pytest 配置（pythonpath=["."], testpaths=["tests"]）
├── requirements.txt            # 依赖（本计划只用到 PyYAML + pytest；其余为后续阶段预留）
├── config.yaml                 # 全局配置（成本档/声线/发布模式/护栏/provider/路径）
├── seeds/cases.yaml            # 种子案例库
├── src/
│   ├── __init__.py
│   ├── config.py               # load_config / load_env / Config(.get dotted)
│   ├── llm.py                  # LLMProvider 协议 / FakeLLM / GeminiLLM / get_llm 工厂
│   ├── context.py              # Ctx 数据类（slug/config/root/llm，.job 属性）
│   ├── run.py                  # 编排器：STAGES 注册表 + run_pipeline + main(CLI)
│   ├── stages/
│   │   ├── __init__.py
│   │   └── s1_topic.py         # 阶段①：load_seed / extract_json / build_prompt / run / OUTPUT
│   └── util/
│       ├── __init__.py
│       └── jobs.py             # write_json / read_json / exists（按 job 目录操作）
└── tests/
    ├── test_smoke.py
    ├── test_config.py
    ├── test_llm.py
    ├── test_jobs.py
    ├── test_context.py
    ├── test_s1_topic.py
    └── test_run.py
```

每个文件单一职责：`config` 只管读配置，`llm` 只管 provider，`jobs` 只管 job 文件读写，`context` 只是传参容器，`run` 只管编排，`stages/*` 各管一个阶段。

---

## Task 1: 项目骨架 + 依赖 + pytest

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `src/__init__.py`, `src/stages/__init__.py`, `src/util/__init__.py`, `tests/test_smoke.py`

**Interfaces:**
- Produces: 可用的 `.venv`；`pytest` 能从仓库根 `import src.*`。

- [ ] **Step 1: 写 `requirements.txt`**

```
PyYAML>=6.0
requests>=2.31
edge-tts>=6.1
google-generativeai>=0.8
google-api-python-client>=2.0
google-auth-oauthlib>=1.2
Pillow>=10.0
pytest>=8.0
```

- [ ] **Step 2: 写 `pyproject.toml`（pytest 配置）**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 3: 建包目录的空 `__init__.py`**

`src/__init__.py`、`src/stages/__init__.py`、`src/util/__init__.py` 三个文件，内容均为空。

- [ ] **Step 4: 写冒烟测试 `tests/test_smoke.py`**

```python
def test_smoke():
    assert True
```

- [ ] **Step 5: 建 venv 并安装依赖**

Run:
```bash
python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt
```
Expected: 安装成功，无报错。

- [ ] **Step 6: 跑冒烟测试**

Run: `.venv/bin/pytest tests/test_smoke.py -q`
Expected: `1 passed`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml requirements.txt src/__init__.py src/stages/__init__.py src/util/__init__.py tests/test_smoke.py
git commit -m "chore: 项目骨架 + pytest + 依赖"
```

---

## Task 2: 配置加载 `config.py` + `config.yaml`

**Files:**
- Create: `src/config.py`, `config.yaml`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `class Config` with `get(self, dotted: str, default=None)` — 点号路径取值（如 `config.get("llm.provider", "gemini")`）。
  - `load_env(path="./.env") -> None` — 把 `KEY=VALUE` 读进 `os.environ`（不覆盖已有）。
  - `load_config(path="config.yaml") -> Config` — 先 `load_env()` 再读 YAML。

- [ ] **Step 1: 写失败测试 `tests/test_config.py`**

```python
import os
from pathlib import Path
from src.config import Config, load_env, load_config


def test_get_dotted_and_default():
    cfg = Config({"llm": {"provider": "gemini"}, "video": {"fps": 30}})
    assert cfg.get("llm.provider") == "gemini"
    assert cfg.get("video.fps") == 30
    assert cfg.get("llm.missing", "x") == "x"
    assert cfg.get("nope.deep.path") is None


def test_load_env_no_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('A=1\n# c\nB="two"\nC=keep\n', encoding="utf-8")
    monkeypatch.setenv("C", "already")
    load_env(env)
    assert os.environ["A"] == "1"
    assert os.environ["B"] == "two"
    assert os.environ["C"] == "already"  # 不覆盖已有


def test_load_config(tmp_path):
    (tmp_path / "config.yaml").write_text("llm:\n  provider: fake\n", encoding="utf-8")
    cfg = load_config(tmp_path / "config.yaml")
    assert cfg.get("llm.provider") == "fake"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_config.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'src.config'`）

- [ ] **Step 3: 写 `src/config.py`**

```python
from __future__ import annotations

import os
from pathlib import Path

import yaml


class Config:
    def __init__(self, data: dict):
        self._data = data or {}

    def get(self, dotted: str, default=None):
        cur = self._data
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur


def load_env(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def load_config(path: str | Path = "config.yaml") -> Config:
    load_env()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config(data)
```

- [ ] **Step 4: 写 `config.yaml`（项目根，真实运行用）**

```yaml
cost_tier: free
llm:
  provider: gemini          # gemini | fake | claude
  gemini_model: gemini-2.0-flash
  script_provider: gemini
tts:
  voice: zh-CN-YunyangNeural
  rate: "-8%"
video:
  resolution: 1080p
  fps: 30
guardrails:
  min_documented_facts: 3
  allowed_licenses: [public-domain, cc0, cc-by, cc-by-sa]
publish:
  visibility: private       # private | unlisted | public（默认 private = 护栏#3）
  auto_upload: false
paths:
  jobs_root: jobs
  seeds: seeds/cases.yaml
```

- [ ] **Step 5: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_config.py -q`
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add src/config.py config.yaml tests/test_config.py
git commit -m "feat(config): 配置加载 + 点号取值 + .env"
```

---

## Task 3: LLM provider 抽象 `llm.py`

**Files:**
- Create: `src/llm.py`, `tests/test_llm.py`

**Interfaces:**
- Consumes: `Config`（来自 Task 2）。
- Produces:
  - `class LLMProvider(Protocol)`：`generate(self, system: str, prompt: str) -> str`。
  - `class FakeLLM`：`__init__(self, canned: str = "")`；`generate` 返回 `canned`，并把每次 `(system, prompt)` 记进 `self.calls`。
  - `class GeminiLLM`：`__init__(self, api_key: str, model: str)`（惰性 import google.generativeai）。
  - `get_llm(config: Config) -> LLMProvider`：按 `llm.provider` 选择；`fake`→FakeLLM，`gemini`→GeminiLLM（读 `GEMINI_API_KEY`），`claude`→`NotImplementedError`，其它→`ValueError`。

- [ ] **Step 1: 写失败测试 `tests/test_llm.py`**

```python
import pytest
from src.config import Config
from src.llm import FakeLLM, get_llm


def test_fake_llm_records_and_returns():
    llm = FakeLLM(canned="hello")
    out = llm.generate("sys", "prompt")
    assert out == "hello"
    assert llm.calls == [("sys", "prompt")]


def test_get_llm_fake():
    llm = get_llm(Config({"llm": {"provider": "fake"}}))
    assert isinstance(llm, FakeLLM)


def test_get_llm_unknown_raises():
    with pytest.raises(ValueError):
        get_llm(Config({"llm": {"provider": "nope"}}))


def test_get_llm_claude_not_implemented():
    with pytest.raises(NotImplementedError):
        get_llm(Config({"llm": {"provider": "claude"}}))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_llm.py -q`
Expected: FAIL（`No module named 'src.llm'`）

- [ ] **Step 3: 写 `src/llm.py`**

```python
from __future__ import annotations

import os
from typing import Protocol

from src.config import Config


class LLMProvider(Protocol):
    def generate(self, system: str, prompt: str) -> str: ...


class FakeLLM:
    """测试用：返回固定字符串，记录调用。不联网。"""

    def __init__(self, canned: str = ""):
        self.canned = canned
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        return self.canned


class GeminiLLM:
    """Gemini 免费额度。惰性 import，使测试无需安装 SDK。"""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = model

    def generate(self, system: str, prompt: str) -> str:
        model = self._genai.GenerativeModel(
            self._model_name, system_instruction=system
        )
        return model.generate_content(prompt).text


def get_llm(config: Config) -> LLMProvider:
    provider = config.get("llm.provider", "gemini")
    if provider == "fake":
        return FakeLLM()
    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("GEMINI_API_KEY 未设置（写进 .env 或环境变量）")
        return GeminiLLM(key, config.get("llm.gemini_model", "gemini-2.0-flash"))
    if provider == "claude":
        raise NotImplementedError("claude provider 在后续『LLM 升级』阶段实现")
    raise ValueError(f"未知 llm provider: {provider}")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_llm.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/llm.py tests/test_llm.py
git commit -m "feat(llm): provider 抽象 + FakeLLM + Gemini + 工厂"
```

---

## Task 4: job 文件读写 `util/jobs.py`

**Files:**
- Create: `src/util/jobs.py`, `tests/test_jobs.py`

**Interfaces:**
- Produces（均按「job 目录」操作，job 目录 = `jobs_root / slug`）：
  - `write_json(job: Path, name: str, data: dict) -> Path` —— 自动建目录，UTF-8、`ensure_ascii=False`、缩进 2。
  - `read_json(job: Path, name: str) -> dict`。
  - `exists(job: Path, name: str) -> bool`。

- [ ] **Step 1: 写失败测试 `tests/test_jobs.py`**

```python
from src.util.jobs import write_json, read_json, exists


def test_write_read_roundtrip(tmp_path):
    job = tmp_path / "roswell-1947"
    assert exists(job, "topic.json") is False
    write_json(job, "topic.json", {"slug": "roswell-1947", "年": 1947})
    assert exists(job, "topic.json") is True
    assert read_json(job, "topic.json") == {"slug": "roswell-1947", "年": 1947}


def test_write_json_is_utf8_unescaped(tmp_path):
    job = tmp_path / "x"
    p = write_json(job, "a.json", {"t": "飞碟"})
    assert "飞碟" in p.read_text(encoding="utf-8")  # 中文未被转义成 \uXXXX
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_jobs.py -q`
Expected: FAIL（`No module named 'src.util.jobs'`）

- [ ] **Step 3: 写 `src/util/jobs.py`**

```python
from __future__ import annotations

import json
from pathlib import Path


def write_json(job: Path, name: str, data: dict) -> Path:
    job.mkdir(parents=True, exist_ok=True)
    p = job / name
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def read_json(job: Path, name: str) -> dict:
    return json.loads((job / name).read_text(encoding="utf-8"))


def exists(job: Path, name: str) -> bool:
    return (job / name).exists()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_jobs.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/util/jobs.py tests/test_jobs.py
git commit -m "feat(jobs): job 目录 JSON 产物读写"
```

---

## Task 5: 上下文容器 `context.py`

**Files:**
- Create: `src/context.py`, `tests/test_context.py`

**Interfaces:**
- Consumes: `Config`（Task 2）、`LLMProvider`（Task 3）。
- Produces: `@dataclass class Ctx` 字段 `slug: str, config: Config, root: Path, llm: LLMProvider`；属性 `job -> Path`（= `root / slug`）。阶段函数签名统一为 `run(ctx: Ctx) -> None`。

- [ ] **Step 1: 写失败测试 `tests/test_context.py`**

```python
from pathlib import Path
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx


def test_ctx_job_path(tmp_path):
    ctx = Ctx(slug="roswell-1947", config=Config({}), root=tmp_path, llm=FakeLLM())
    assert ctx.job == tmp_path / "roswell-1947"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_context.py -q`
Expected: FAIL（`No module named 'src.context'`）

- [ ] **Step 3: 写 `src/context.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import Config
from src.llm import LLMProvider


@dataclass
class Ctx:
    slug: str
    config: Config
    root: Path
    llm: LLMProvider

    @property
    def job(self) -> Path:
        return self.root / self.slug
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_context.py -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/context.py tests/test_context.py
git commit -m "feat(context): Ctx 容器 + job 路径"
```

---

## Task 6: 阶段① 选题 `stages/s1_topic.py` + 种子库

**Files:**
- Create: `seeds/cases.yaml`, `src/stages/s1_topic.py`, `tests/test_s1_topic.py`

**Interfaces:**
- Consumes: `Ctx`（Task 5）、`write_json`（Task 4）、`ctx.llm.generate`（Task 3）。
- Produces:
  - `OUTPUT = "topic.json"`。
  - `load_seed(slug: str, seeds_path: str | Path) -> dict` —— 在种子库里按 slug 找；找不到抛 `KeyError`。
  - `extract_json(text: str) -> dict` —— 取首个 `{` 到末个 `}` 切片后 `json.loads`（容忍 ``` 围栏和前后散文）。
  - `build_prompt(seed: dict) -> str`。
  - `run(ctx: Ctx) -> None` —— 读种子 → LLM 扩展 → `extract_json` → 补 `slug` → 写 `topic.json`。

- [ ] **Step 1: 写种子库 `seeds/cases.yaml`**

```yaml
- slug: roswell-1947
  title_working: 罗斯威尔事件
  year: 1947
  location: 美国新墨西哥州
  hook: 军方先承认回收了一只"飞碟"，几小时后改口说是气象气球
  known_controversies: ["军方两次声明自相矛盾", "Project Mogul 高空气球解释"]
  seed_sources: ["RAAF 1947 新闻稿存档", "维基百科 Roswell incident"]
- slug: phoenix-lights-1997
  title_working: 凤凰城之光
  year: 1997
  location: 美国亚利桑那州
  hook: 上千人目击巨大 V 形光阵缓慢飘过城市上空
  known_controversies: ["照明弹解释 vs 固体飞行器目击", "州长事后承认也看到了"]
  seed_sources: ["维基百科 Phoenix Lights"]
- slug: nimitz-tictac-2004
  title_working: 尼米兹号 Tic-Tac UFO
  year: 2004
  location: 美国加州外海
  hook: 海军飞行员用战机红外吊舱拍下会瞬间加速的"白色 Tic-Tac"
  known_controversies: ["五角大楼承认视频真实", "物理上无法解释的机动"]
  seed_sources: ["美国国防部 2020 解密视频", "Nimitz UAP 报告"]
- slug: tunguska-1908
  title_working: 通古斯大爆炸
  year: 1908
  location: 俄罗斯西伯利亚
  hook: 一场把两千平方公里森林夷平的爆炸，却找不到任何陨石坑
  known_controversies: ["陨石/彗星空爆说", "至今无定论的能量来源"]
  seed_sources: ["维基百科 Tunguska event"]
```

- [ ] **Step 2: 写失败测试 `tests/test_s1_topic.py`**

```python
import json
import pytest
from pathlib import Path
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.stages.s1_topic import load_seed, extract_json, run, OUTPUT


SEEDS = """- slug: roswell-1947
  title_working: 罗斯威尔事件
  year: 1947
"""


def _seeds_file(tmp_path) -> Path:
    p = tmp_path / "cases.yaml"
    p.write_text(SEEDS, encoding="utf-8")
    return p


def test_load_seed_found(tmp_path):
    seed = load_seed("roswell-1947", _seeds_file(tmp_path))
    assert seed["year"] == 1947


def test_load_seed_missing_raises(tmp_path):
    with pytest.raises(KeyError):
        load_seed("does-not-exist", _seeds_file(tmp_path))


def test_extract_json_fenced_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('好的，结果是：{"a": 2} 完毕') == {"a": 2}


def test_run_writes_topic_json(tmp_path):
    seeds = _seeds_file(tmp_path)
    canned = json.dumps({"title_working": "罗斯威尔事件", "year": 1947})
    cfg = Config({"paths": {"seeds": str(seeds)}})
    ctx = Ctx(slug="roswell-1947", config=cfg, root=tmp_path / "jobs",
              llm=FakeLLM(canned=canned))
    run(ctx)
    out = json.loads((ctx.job / OUTPUT).read_text(encoding="utf-8"))
    assert out["slug"] == "roswell-1947"      # run 补上的
    assert out["year"] == 1947                 # LLM 返回的
```

- [ ] **Step 3: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_s1_topic.py -q`
Expected: FAIL（`No module named 'src.stages.s1_topic'`）

- [ ] **Step 4: 写 `src/stages/s1_topic.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.context import Ctx
from src.util.jobs import write_json

OUTPUT = "topic.json"

SYSTEM = (
    "你是中文 UFO/未解之谜频道「探索飞碟与外星人」的资料研究员。"
    "只输出一个 JSON 对象，不要任何额外解释或 Markdown 围栏外的文字。"
)


def load_seed(slug: str, seeds_path: str | Path) -> dict:
    cases = yaml.safe_load(Path(seeds_path).read_text(encoding="utf-8")) or []
    for c in cases:
        if c.get("slug") == slug:
            return c
    raise KeyError(f"种子案例未找到: {slug}（in {seeds_path}）")


def extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"LLM 输出里没有 JSON 对象: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def build_prompt(seed: dict) -> str:
    return (
        "根据以下案例线索，补全并标准化一个选题元信息对象。\n"
        f"案例线索：{json.dumps(seed, ensure_ascii=False)}\n\n"
        "输出 JSON，字段：title_working(中文标题), year(整数), location(中文), "
        "hook(一句话钩子,中文), known_controversies(中文字符串数组), "
        "seed_sources(字符串数组)。"
    )


def run(ctx: Ctx) -> None:
    seeds_path = ctx.config.get("paths.seeds", "seeds/cases.yaml")
    seed = load_seed(ctx.slug, seeds_path)
    raw = ctx.llm.generate(SYSTEM, build_prompt(seed))
    topic = extract_json(raw)
    topic.setdefault("slug", ctx.slug)
    write_json(ctx.job, OUTPUT, topic)
```

- [ ] **Step 5: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_s1_topic.py -q`
Expected: `5 passed`

- [ ] **Step 6: Commit**

```bash
git add seeds/cases.yaml src/stages/s1_topic.py tests/test_s1_topic.py
git commit -m "feat(s1): 阶段① 选题 + 种子案例库"
```

---

## Task 7: 编排器 `run.py`

**Files:**
- Create: `src/run.py`, `tests/test_run.py`

**Interfaces:**
- Consumes: `load_config`/`Config`（Task 2）、`get_llm`/`LLMProvider`（Task 3）、`Ctx`（Task 5）、`exists`（Task 4）、`s1_topic`（Task 6）。
- Produces:
  - `STAGES: list[tuple[str, Callable[[Ctx], None], str]]` —— (阶段名, run 函数, 主产物文件名)。本计划只注册 `s1_topic`。
  - `run_pipeline(slug, config, root, llm, frm=None, to=None, force=False) -> None` —— 按 `STAGES` 顺序跑；产物已存在且非 `--force` 则跳过；`frm`/`to` 限定阶段名子区间。
  - `main(argv=None) -> None` —— CLI：`--case`（必填）、`--from`、`--to`、`--force`、`--config`。

- [ ] **Step 1: 写失败测试 `tests/test_run.py`**

```python
from pathlib import Path
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.util.jobs import write_json
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_run.py -q`
Expected: FAIL（`No module named 'src.run'` 或 `STAGES` 未定义）

- [ ] **Step 3: 写 `src/run.py`**

```python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from src.config import Config, load_config
from src.context import Ctx
from src.llm import LLMProvider, get_llm
from src.stages import s1_topic
from src.util.jobs import exists

STAGES: list[tuple[str, Callable[[Ctx], None], str]] = [
    ("s1_topic", s1_topic.run, s1_topic.OUTPUT),
]


def run_pipeline(
    slug: str,
    config: Config,
    root: Path,
    llm: LLMProvider,
    frm: str | None = None,
    to: str | None = None,
    force: bool = False,
) -> None:
    names = [s[0] for s in STAGES]
    i0 = names.index(frm) if frm else 0
    i1 = names.index(to) if to else len(STAGES) - 1
    ctx = Ctx(slug=slug, config=config, root=root, llm=llm)
    for name, fn, output in STAGES[i0 : i1 + 1]:
        if not force and exists(ctx.job, output):
            print(f"[skip] {name} ({output} 已存在)")
            continue
        print(f"[run ] {name}")
        fn(ctx)
        print(f"[done] {name} -> {output}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="探索飞碟与外星人 出片管线")
    ap.add_argument("--case", required=True, help="种子案例 slug")
    ap.add_argument("--from", dest="frm", default=None, help="起始阶段名")
    ap.add_argument("--to", default=None, help="结束阶段名")
    ap.add_argument("--force", action="store_true", help="忽略已存在产物，重跑")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    root = Path(config.get("paths.jobs_root", "jobs"))
    llm = get_llm(config)
    run_pipeline(args.case, config, root, llm, args.frm, args.to, args.force)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_run.py -q`
Expected: `2 passed`

- [ ] **Step 5: 跑全量测试**

Run: `.venv/bin/pytest -q`
Expected: 全部 passed（约 18 个用例）

- [ ] **Step 6: Commit**

```bash
git add src/run.py tests/test_run.py
git commit -m "feat(run): 编排器 + 跳过/续跑/force + from/to"
```

---

## Task 8: 真实 Gemini 端到端冒烟（手动验收）

**Files:**
- Modify: 无（仅运行验证）。

**Interfaces:**
- Consumes: Task 1–7 全部。
- Produces: `jobs/roswell-1947/topic.json`（真实 LLM 产物）。

- [ ] **Step 1: 配置 Gemini key**

在仓库根建 `.env`（已被 git 忽略）：
```
GEMINI_API_KEY=你的key
```
（去 Google AI Studio 免费申请 https://aistudio.google.com/apikey ）

- [ ] **Step 2: 真实跑阶段①**

Run: `.venv/bin/python -m src.run --case roswell-1947`
Expected: 打印 `[run ] s1_topic` → `[done] s1_topic -> topic.json`

- [ ] **Step 3: 看产物**

Run: `cat jobs/roswell-1947/topic.json`
Expected: 一个合法 JSON，含 `slug=roswell-1947`、中文 `title_working`、`hook` 等字段。

- [ ] **Step 4: 验证续跑跳过**

Run: `.venv/bin/python -m src.run --case roswell-1947`
Expected: 打印 `[skip] s1_topic (topic.json 已存在)`，不再调用 LLM。

- [ ] **Step 5: 验证 force 重跑**

Run: `.venv/bin/python -m src.run --case roswell-1947 --force`
Expected: 打印 `[run ] s1_topic` 重新生成。

> 无需 commit（`jobs/` 已被 git 忽略）。这一步确认地基在真实 LLM 下端到端可用。

---

## Self-Review

**1. Spec 覆盖（本计划范围 = 地基 + 阶段①）：**
- 模块化 CLI 架构（spec §5）→ Task 1/5/7 ✅
- `config.yaml` 字段（spec §9）→ Task 2 ✅
- `llm.py` provider 抽象 gemini/claude（spec §5/§8）→ Task 3 ✅（claude 占位）
- 每集 job 文件夹 + 幂等续跑（spec §5/§10）→ Task 4/7 ✅
- 阶段① I/O 契约 `topic.json`（spec §6①）→ Task 6 ✅
- 种子案例库（spec §12）→ Task 6 ✅
- 「先全免费 Gemini」（用户确认）→ Task 3 默认 + Task 8 验收 ✅
- 阶段②–⑨、护栏①②③、TTS/字幕/素材/合成/发布 → **明确留给 P2–P5**（见下方路线）。

**2. 占位符扫描：** 无 TBD/TODO；每步都有真实代码与命令。`claude` provider 的 `NotImplementedError` 是有意的升级位占位，非计划漏洞。

**3. 类型一致性：** `Config.get` / `FakeLLM(canned=)` / `LLMProvider.generate(system, prompt)` / `Ctx(slug,config,root,llm).job` / `write_json/read_json/exists(job, name)` / `extract_json` / `load_seed` / `OUTPUT` / `STAGES`(三元组) / `run_pipeline(...)` 在 Task 间签名一致。阶段统一接口 `run(ctx: Ctx) -> None` 贯穿 Task 5/6/7。

---

## 后续计划路线（P2–P5，本计划完成后逐个写）

- **P2 内容**：`s2_research`（护栏#1 事实核查，证据等级）→ `s3_script`（原创悬疑脚本，分段+画面提示）。产物 `research.json` / `script.json`。
- **P3 音字**：`s4_tts`（edge-tts + WordBoundary 时间戳）→ `s5_subs`（文稿+时间戳生成 `.ass`，无需 Whisper）。
- **P4 成片**：`s6_media`（护栏#2 白名单源 NASA/archive.org/Wikimedia + license 校验 + Ken Burns）→ `s7_assemble`（ffmpeg：画面+配音+字幕+BGM 闪避+片头尾）。产物 `video.mp4`。
- **P5 发布**：`s8_meta`（LLM 写标题/简介/标签 + Pillow 封面）→ `s9_publish`（护栏#3 YouTube Data API，默认 private）。
- **LLM 升级（随时可插）**：实现 `llm.py` 的 `claude` 分支（Anthropic SDK，`claude-opus-4-8`/`claude-sonnet-4-6`，可挂 `web_search`），把脚本/调研切到 Claude。

### P2 设计输入（来自 P1 终审，写 P2 计划时必须处理）

1. **编排器契约要扩展** —— 当前 `run(ctx) -> None` 表达不了三件 P2 需要的事：
   - **控制流 / `blocked`（护栏#1，spec §7/§10）**：s2 若判定证据过弱写 `research.json.blocked=true`，`-> None` 阶段无法让 `run_pipeline` 停止或自动跳到下一个种子。定个机制（如抛 `Blocked` 异常，或 `run_pipeline` 每阶段后检查 sentinel 字段）。
   - **多产物 checkpoint**：s4（`audio/`+`timing.json`）、s6（`media/`+`credits.json`）产出多个文件，而跳过逻辑只认单个 `OUTPUT`。明确「主产物=checkpoint」约定，避免半写的多文件阶段被误跳。
   - **非致命 warnings 通道（spec §10）**：阶段需要能记录 warning 而不中断（`ctx` 级收集器或 per-artifact `warnings` 字段）。
2. **加 `.env.example`**（`GEMINI_API_KEY=`）——可安全提交，替代「手动建 .env」步骤。
3. **Gemini SDK 选型**：现用 legacy `google-generativeai`；Google 现行 SDK 是 `google-genai`。在 ②③⑧ 大量依赖前，明确锁定版本或迁移。
4. **依赖锁定**：`requirements.txt` 目前全 `>=` 无 lockfile；考虑 pin 以保证可复现。
