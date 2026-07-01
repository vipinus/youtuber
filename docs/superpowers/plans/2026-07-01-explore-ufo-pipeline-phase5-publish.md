# 探索飞碟与外星人 · P5 发布 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 闭环端到端：阶段⑧写 SEO 标题/简介（含素材署名）+ Pillow 封面；阶段⑨（护栏#3）用 YouTube Data API 上传 `video.mp4`，默认 `private` 草稿。跑完（+一次性 OAuth 授权）就能把成片自动发到你的频道。

**Architecture:** 上传引擎抽象成 provider（`youtube`/`fake`），与 LLM/TTS/Media 同构，单测用 `FakeUploader` 全离线。OAuth 凭据存 `secrets/`（git 忽略），一次性授权脚本 `src/auth_youtube.py` 跑浏览器同意→存 token。护栏#3：`publish.visibility` 默认 `private`，`publish.auto_upload` 关时 s9 跳过上传。s8 封面/元数据纯本地（LLM + Pillow）。

**Tech Stack:** Python 3.14；`google-api-python-client` + `google-auth-oauthlib`（已在 requirements）；Pillow（封面）；pytest（离线：FakeLLM + FakeUploader）。

## Global Constraints

- Python 3.14；`from __future__ import annotations`；UTF-8。
- **单测绝不联网/不碰 OAuth**：LLM 用 FakeLLM，上传用 FakeUploader（`youtube.provider=fake`）。
- 阶段接口 `run(ctx)->None` + `OUTPUT`/`INPUT`。s8 `OUTPUT="meta.json"`（+副产物 `thumb.jpg`）, `INPUT="script.json"`；s9 `OUTPUT="publish.json"`, `INPUT="meta.json"`。
- **护栏#3**：`publish.visibility`（默认 `private`）传给上传；`publish.auto_upload`（默认 `false` 保守，用户已在 config 设 `true`）为 false 时 s9 `ctx.warn` 并写 `publish.json {status:"skipped"}`、不上传。
- **素材署名**：s8 从 `credits.json` 收集非 fallback 的 attribution，追加到简介（满足 CC 署名）。
- 凭据 `secrets/client_secret.json` / `secrets/token.json` 永不进 git（`.gitignore` 已含 `secrets/`）。

## File Structure

```
新增：
  src/stages/s8_meta.py         # 阶段⑧ 元数据 + 封面
  src/youtube.py                # Uploader 协议 / FakeUploader / YouTubeUploader / get_uploader / load_creds
  src/auth_youtube.py           # 一次性 OAuth 授权脚本
  src/stages/s9_publish.py      # 阶段⑨ 上传（护栏#3）
  tests/test_s8_meta.py
  tests/test_youtube.py
  tests/test_s9_publish.py
  docs/youtube-oauth-setup.md   # 用户一次性授权指南
修改：
  src/run.py                    # 注册 s8/s9
  config.yaml                   # youtube 块
  tests/test_run.py             # 九阶段 e2e（fake 全链）
```

---

## Task 1: 阶段⑧ 元数据 + 封面 `src/stages/s8_meta.py`

**Files:**
- Create: `src/stages/s8_meta.py`, `tests/test_s8_meta.py`

**Interfaces:**
- Consumes: `Ctx`、`read_json`/`write_json`、`extract_json`（util.llmjson）、`ctx.llm.generate`。
- Produces: `OUTPUT="meta.json"`, `INPUT="script.json"`, `run(ctx)`, `make_thumbnail(dest, title, bg)`。副产物 `thumb.jpg`（1280×720）。简介追加 `credits.json` 非 fallback 署名。

- [ ] **Step 1: 写失败测试 `tests/test_s8_meta.py`**
```python
import json
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.util.jobs import write_json, read_json
from src.stages.s8_meta import run, OUTPUT, INPUT


def _ctx(tmp_path, canned):
    ctx = Ctx(slug="c", config=Config({}), root=tmp_path, llm=FakeLLM(canned=canned))
    write_json(ctx.job, INPUT, {"title_options": ["罗斯威尔之谜"],
                                "segments": [{"id": 1, "text": "飞碟坠落。"}]})
    write_json(ctx.job, "credits.json", {"items": [
        {"id": 1, "source": "nasa", "license": "public-domain", "attribution": "NASA / Roswell"},
        {"id": 2, "source": "fallback", "license": "public-domain", "attribution": "fallback"},
    ]})
    (ctx.job / "media").mkdir(parents=True, exist_ok=True)
    return ctx


def test_run_writes_meta_and_thumb(tmp_path):
    canned = json.dumps({"title": "罗斯威尔：军方隐瞒了什么？",
                         "description": "1947年的真相。", "tags": ["UFO", "罗斯威尔"]})
    ctx = _ctx(tmp_path, canned)
    run(ctx)
    meta = read_json(ctx.job, OUTPUT)
    assert meta["title"] == "罗斯威尔：军方隐瞒了什么？"
    assert "NASA / Roswell" in meta["description"]     # 署名追加
    assert "fallback" not in meta["description"]        # 兜底不署名
    thumb = ctx.job / "thumb.jpg"
    assert thumb.exists() and thumb.stat().st_size > 0
    from PIL import Image
    assert Image.open(thumb).size == (1280, 720)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_s8_meta.py -q`
Expected: FAIL（`No module named 'src.stages.s8_meta'`）

- [ ] **Step 3: 写 `src/stages/s8_meta.py`**
```python
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.context import Ctx
from src.util.jobs import read_json, write_json
from src.util.llmjson import extract_json

OUTPUT = "meta.json"
INPUT = "script.json"
TITLE_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc"

SYSTEM = (
    "你是「探索飞碟与外星人」频道的 YouTube 运营。为给定脚本写 SEO 元数据。"
    "只输出一个 JSON 对象，不要额外文字。"
)


def build_prompt(script: dict) -> str:
    hint = (script.get("title_options") or ["未解之谜"])[0]
    body = "".join(s.get("text", "") for s in script.get("segments", []))[:800]
    return (
        "为下面的悬疑解说写 YouTube 元数据，输出 JSON，字段："
        "title(中文,吸引点击,≤40字), description(中文,2-4句), tags(中文/英文字符串数组,5-12个)。\n"
        f"标题线索：{hint}\n脚本节选：{body}"
    )


def _fit(path: str, text: str, max_w: int, start: int = 90, lo: int = 40) -> ImageFont.FreeTypeFont:
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    s = start
    while s > lo:
        f = ImageFont.truetype(path, s)
        if probe.textbbox((0, 0), text, font=f, stroke_width=3)[2] <= max_w:
            return f
        s -= 3
    return ImageFont.truetype(path, lo)


def make_thumbnail(dest: Path, title: str, bg: Path) -> None:
    W, H = 1280, 720
    if bg.exists() and bg.stat().st_size > 0:
        img = Image.open(bg).convert("RGB").resize((W, H))
    else:
        img = Image.new("RGB", (W, H), (10, 12, 28))
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 110))          # 压暗保证字清晰
    img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    d = ImageDraw.Draw(img)
    line = title if len(title) <= 9 else title[:9]
    font = _fit(TITLE_FONT, line, int(W * 0.86))
    d.text((W // 2, H // 2), line, font=font, fill=(255, 255, 255),
           anchor="mm", stroke_width=6, stroke_fill=(8, 12, 30))
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, "JPEG", quality=88, optimize=True)


def run(ctx: Ctx) -> None:
    script = read_json(ctx.job, INPUT)
    meta = extract_json(ctx.llm.generate(SYSTEM, build_prompt(script)))
    try:
        credits = read_json(ctx.job, "credits.json").get("items", [])
    except FileNotFoundError:
        credits = []
    attrib = sorted({c["attribution"] for c in credits if c.get("source") != "fallback"})
    if attrib:
        meta["description"] = meta.get("description", "") + "\n\n素材来源：" + "；".join(attrib)
    title = meta.get("title") or (script.get("title_options") or ["未解之谜"])[0]
    make_thumbnail(ctx.job / "thumb.jpg", title, ctx.job / "media" / "seg_001.jpg")
    write_json(ctx.job, OUTPUT, meta)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_s8_meta.py -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**
```bash
git add src/stages/s8_meta.py tests/test_s8_meta.py
git commit -m "feat(s8): 阶段⑧ SEO 元数据 + Pillow 封面 + 素材署名"
```

---

## Task 2: YouTube 上传抽象 + 授权 `src/youtube.py`

**Files:**
- Create: `src/youtube.py`, `src/auth_youtube.py`, `tests/test_youtube.py`, `docs/youtube-oauth-setup.md`
- Modify: `config.yaml`

**Interfaces:**
- Produces:
  - `Uploader(Protocol)`: `upload(video: Path, meta: dict, thumb: Path, visibility: str) -> dict`（返回 `{videoId, url, status}`）。
  - `FakeUploader`（离线：返回假 videoId，不联网）。
  - `YouTubeUploader(creds)`（真 YouTube Data API v3；惰性 import googleapiclient）。
  - `load_creds(token, client_secret, scopes)` —— 读 token，过期则 refresh。
  - `get_uploader(config) -> Uploader`：`fake`→FakeUploader，`youtube`→YouTubeUploader(load_creds(...))，else `ValueError`。
  - `src/auth_youtube.py`：一次性 `InstalledAppFlow` 浏览器授权→存 `secrets/token.json`。

- [ ] **Step 1: 写失败测试 `tests/test_youtube.py`**
```python
import pytest
from pathlib import Path
from src.config import Config
from src.youtube import FakeUploader, get_uploader


def test_fake_uploader(tmp_path):
    up = FakeUploader()
    r = up.upload(tmp_path / "v.mp4", {"title": "t"}, tmp_path / "th.jpg", "private")
    assert r["videoId"] and r["status"] == "private" and "youtu" in r["url"]


def test_get_uploader_fake():
    assert isinstance(get_uploader(Config({"youtube": {"provider": "fake"}})), FakeUploader)


def test_get_uploader_unknown_raises():
    with pytest.raises(ValueError):
        get_uploader(Config({"youtube": {"provider": "nope"}}))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_youtube.py -q`
Expected: FAIL（`No module named 'src.youtube'`）

- [ ] **Step 3: 写 `src/youtube.py`**
```python
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.config import Config

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


class Uploader(Protocol):
    def upload(self, video: Path, meta: dict, thumb: Path, visibility: str) -> dict: ...


class FakeUploader:
    def upload(self, video: Path, meta: dict, thumb: Path, visibility: str) -> dict:
        return {"videoId": "FAKE12345", "url": "https://youtu.be/FAKE12345",
                "status": visibility}


def load_creds(token: str, client_secret: str, scopes=SCOPES):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    creds = Credentials.from_authorized_user_file(token, scopes)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds


class YouTubeUploader:
    def __init__(self, creds):
        from googleapiclient.discovery import build

        self._yt = build("youtube", "v3", credentials=creds)

    def upload(self, video: Path, meta: dict, thumb: Path, visibility: str) -> dict:
        from googleapiclient.http import MediaFileUpload

        body = {
            "snippet": {"title": meta.get("title", ""),
                        "description": meta.get("description", ""),
                        "tags": meta.get("tags", []), "categoryId": "22"},
            "status": {"privacyStatus": visibility, "selfDeclaredMadeForKids": False},
        }
        req = self._yt.videos().insert(
            part="snippet,status", body=body,
            media_body=MediaFileUpload(str(video), resumable=True))
        resp = req.execute()
        vid = resp["id"]
        if thumb and Path(thumb).exists():
            self._yt.thumbnails().set(
                videoId=vid, media_body=MediaFileUpload(str(thumb))).execute()
        return {"videoId": vid, "url": f"https://youtu.be/{vid}", "status": visibility}


def get_uploader(config: Config) -> Uploader:
    provider = config.get("youtube.provider", "youtube")
    if provider == "fake":
        return FakeUploader()
    if provider == "youtube":
        creds = load_creds(config.get("youtube.token", "secrets/token.json"),
                           config.get("youtube.client_secret", "secrets/client_secret.json"))
        return YouTubeUploader(creds)
    raise ValueError(f"未知 youtube provider: {provider}")
```

- [ ] **Step 4: 写 `src/auth_youtube.py`（一次性授权）**
```python
"""一次性 YouTube 授权：跑浏览器同意，把 token 存到 secrets/token.json。
用法：.venv/bin/python -m src.auth_youtube
前置：secrets/client_secret.json（从 Google Cloud 下载，见 docs/youtube-oauth-setup.md）。
"""
from __future__ import annotations

from pathlib import Path

from src.youtube import SCOPES


def main() -> None:
    from google_auth_oauthlib.flow import InstalledAppFlow

    Path("secrets").mkdir(exist_ok=True)
    flow = InstalledAppFlow.from_client_secrets_file("secrets/client_secret.json", SCOPES)
    creds = flow.run_local_server(port=0)
    Path("secrets/token.json").write_text(creds.to_json(), encoding="utf-8")
    print("已授权，token 存到 secrets/token.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: config.yaml 加 youtube 块**
```yaml
youtube:
  provider: youtube        # youtube | fake
  client_secret: secrets/client_secret.json
  token: secrets/token.json
```

- [ ] **Step 6: 写 `docs/youtube-oauth-setup.md`（用户一次性设置指南）**

内容（要点，写成 markdown 步骤）：
```markdown
# YouTube 上传一次性授权

1. 去 https://console.cloud.google.com 建个项目。
2. 「API 和服务」→启用 **YouTube Data API v3**。
3. 「凭据」→创建 **OAuth 客户端 ID**→类型选「桌面应用」→下载 JSON。
4. 把下载的 JSON 存为 `secrets/client_secret.json`（已在 .gitignore，不进库）。
5. 「OAuth 同意屏幕」把你自己的 Google 账号加为测试用户。
6. 跑：`.venv/bin/python -m src.auth_youtube` → 浏览器点授权 → 生成 `secrets/token.json`。
7. 之后管线 s9 就能自动上传（默认 private 草稿）。
注意：免费配额约 6 次上传/天（每次上传约 1600 配额，日配额 10000）；要更多可申请提额。
```

- [ ] **Step 7: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_youtube.py -q`
Expected: `3 passed`

- [ ] **Step 8: Commit**
```bash
git add src/youtube.py src/auth_youtube.py config.yaml tests/test_youtube.py docs/youtube-oauth-setup.md
git commit -m "feat(youtube): 上传 provider 抽象 + 一次性 OAuth 授权 + 设置指南"
```

---

## Task 3: 阶段⑨ 上传 `src/stages/s9_publish.py`（护栏#3）

**Files:**
- Create: `src/stages/s9_publish.py`, `tests/test_s9_publish.py`

**Interfaces:**
- Consumes: `Ctx`、`read_json`/`write_json`、`get_uploader`、`ctx.warn`、`config.publish.auto_upload`/`publish.visibility`。
- Produces: `OUTPUT="publish.json"`, `INPUT="meta.json"`, `run(ctx)`。护栏#3：auto_upload 关→跳过；visibility 默认 private。

- [ ] **Step 1: 写失败测试 `tests/test_s9_publish.py`**
```python
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.util.jobs import write_json, read_json
from src.stages.s9_publish import run, OUTPUT, INPUT


def _ctx(tmp_path, cfg):
    ctx = Ctx(slug="c", config=cfg, root=tmp_path, llm=FakeLLM())
    write_json(ctx.job, INPUT, {"title": "t", "description": "d", "tags": ["x"]})
    (ctx.job / "video.mp4").write_bytes(b"MP4")
    return ctx


def test_run_uploads_when_enabled(tmp_path):
    cfg = Config({"youtube": {"provider": "fake"},
                  "publish": {"auto_upload": True, "visibility": "private"}})
    ctx = _ctx(tmp_path, cfg)
    run(ctx)
    pub = read_json(ctx.job, OUTPUT)
    assert pub["videoId"] == "FAKE12345"
    assert pub["status"] == "private"          # 护栏#3 默认私享


def test_run_skips_when_auto_upload_off(tmp_path):
    cfg = Config({"youtube": {"provider": "fake"}, "publish": {"auto_upload": False}})
    ctx = _ctx(tmp_path, cfg)
    run(ctx)
    pub = read_json(ctx.job, OUTPUT)
    assert pub["status"] == "skipped"          # 关了就不传
    assert ctx.warnings                        # 有 warn
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_s9_publish.py -q`
Expected: FAIL（`No module named 'src.stages.s9_publish'`）

- [ ] **Step 3: 写 `src/stages/s9_publish.py`**
```python
from __future__ import annotations

from src.context import Ctx
from src.util.jobs import read_json, write_json
from src.youtube import get_uploader

OUTPUT = "publish.json"
INPUT = "meta.json"


def run(ctx: Ctx) -> None:
    if not ctx.config.get("publish.auto_upload", False):
        ctx.warn(f"{ctx.slug}: publish.auto_upload=false，跳过上传")
        write_json(ctx.job, OUTPUT, {"status": "skipped"})
        return
    meta = read_json(ctx.job, INPUT)
    visibility = ctx.config.get("publish.visibility", "private")   # 护栏#3
    uploader = get_uploader(ctx.config)
    result = uploader.upload(ctx.job / "video.mp4", meta,
                             ctx.job / "thumb.jpg", visibility)
    write_json(ctx.job, OUTPUT, result)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_s9_publish.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**
```bash
git add src/stages/s9_publish.py tests/test_s9_publish.py
git commit -m "feat(s9): 阶段⑨ YouTube 上传（护栏#3: auto_upload开关+private默认）"
```

---

## Task 4: 注册 s8/s9 + 九阶段 e2e

**Files:**
- Modify: `src/run.py`, `tests/test_run.py`

**Interfaces:**
- Produces: `STAGES` 增加 s8/s9；九阶段离线 e2e（SeqLLM 4 段 + FakeTTS + FakeMedia + FakeUploader + auto_upload=true）。

- [ ] **Step 1: 加 e2e 测试到 `tests/test_run.py`**（`SeqLLM` 现在要 4 段：topic/research/script/meta）
```python
def test_e2e_full_pipeline(tmp_path):
    import json
    from src.util.jobs import read_json
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_run.py::test_e2e_full_pipeline -q`
Expected: FAIL（STAGES 无 s8/s9，meta.json/publish.json 未生成）

- [ ] **Step 3: 注册 s8/s9 到 `src/run.py`**

import 区加 `s8_meta, s9_publish`；`STAGES` 在 `s7_assemble` 后加：
```python
    ("s8_meta", s8_meta.run, s8_meta.OUTPUT),
    ("s9_publish", s9_publish.run, s9_publish.OUTPUT),
```

- [ ] **Step 4: 全量测试**

Run: `.venv/bin/pytest -q`
Expected: 全绿（九阶段 e2e + 全部旧测试）。

> ⚠️ 审计：`grep -n "run_pipeline(" tests/ -r` —— 确认没有跑真实 STAGES 且未 `to=`/未 fake 的测试现在会落到 s8（真 LLM）或 s9（真 OAuth）。既有 e2e 已绑 `to="s6_media"`，安全；新 e2e 用 FakeUploader + fake 全链。

- [ ] **Step 5: Commit**
```bash
git add src/run.py tests/test_run.py
git commit -m "feat(run): 注册阶段⑧⑨ + 九阶段端到端"
```

> **真实上传冒烟无法自动化**：需要你的 Google 账号/频道 OAuth（见 `docs/youtube-oauth-setup.md`）。授权后 `.venv/bin/python -m src.run --case <slug>` 会把成片自动传成 private 草稿。

---

## Self-Review

**1. Spec 覆盖（P5 范围）：**
- 阶段⑧ meta.json + thumb.jpg（spec §6⑧）→ Task 1 ✅
- 阶段⑨ publish.json + YouTube 上传（spec §6⑨）→ Task 3 ✅
- 护栏#3 auto_upload 开关 + visibility private 默认（spec §7#3）→ Task 3 ✅
- CC 署名（spec §6⑧「含素材出处归属」）→ Task 1（credits 非 fallback attribution 追加简介）✅
- 离线测试纪律（fake 全链）+ 防触网/触 OAuth 回归 → Task 4 Step 4 ⚠️ 审计 ✅
- 一次性 OAuth 设置 → Task 2 auth_youtube + docs ✅

**2. 占位符扫描：** 无 TBD；每步含完整代码。真实上传冒烟为用户 OAuth 后手动步骤（已说明），非漏洞。

**3. 类型一致性：** s8 `run/make_thumbnail/build_prompt`、`Uploader.upload(video,meta,thumb,visibility)->dict`、`FakeUploader`/`YouTubeUploader`/`get_uploader`/`load_creds`、s9 `run` 读 meta 传 uploader、`OUTPUT`/`INPUT`、`STAGES` 三元组 一致。s9 auto_upload/visibility 与 config.publish 对齐。

---

## 后续（P5 之后 = 全链闭环，剩打磨/升级）

- **LLM 升级**：`get_llm` 的 `claude` 分支，绕开 Gemini 每日配额、脚本质量更高。
- **P4b 遗留**：Ken Burns/BGM/片头尾；音频 `aformat` 声道归一；s7 `n=0` 守卫。
- **素材源扩展**：archive.org/Wikimedia（真读 license）；AI 出图。
- **无人值守批量**：cron/Prefect 跑多案例；数据回流选题。
