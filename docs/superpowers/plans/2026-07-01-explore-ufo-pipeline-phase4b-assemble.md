# 探索飞碟与外星人 · P4b 合成 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 阶段⑦ 用 ffmpeg 把 `media/` 图 + `audio/` 配音 + `subs.ass` 字幕 组装成 `video.mp4` —— 每段图铺满 1080p、按**真实音频时长**（P4a 已修）对轴、拼全音轨、烧录字幕。跑完给一个案例就能产出一条完整视频。

**Architecture:** 稳健 MVP = **静态幻灯片**（每段图 scale+crop 铺满 1920×1080，展示其段时长）→ concat 视频；每段音频 aresample 归一 → concat 音轨；`subtitles=` 烧录 `subs.ass`；单条 ffmpeg `filter_complex` 出 `video.mp4`。命令构造 `build_cmd` 是纯函数（单测校验结构），真实 ffmpeg 编码走冒烟。Ken Burns 推拉（zoompan）作紧随升级，先保证稳出片。

**Tech Stack:** Python 3.14；`ffmpeg`（libx264/aac，已装 v8）；`subprocess`；pytest（单测 mock subprocess，不真编码；真实 ffmpeg 走本地冒烟，无网络）。

## Global Constraints

- Python 3.14；`from __future__ import annotations`；UTF-8。
- 阶段接口 `run(ctx)->None`；`OUTPUT="video.mp4"`, `INPUT="timing.json"`。
- s7 从 `timing.json` 取每段 `{id, audio, dur_sec}`；图片按 id 取 `media/seg_{id:03d}.jpg`；字幕用 `subs.ass`。**不需要 credits.json**（署名是 P5 s8 的事）。
- 时间轴以 `timing.json` 的真实 `dur_sec` 为准（P4a 已让它=真实音频时长）。
- **单测不真跑 ffmpeg**（慢）：`build_cmd` 纯函数校验结构；`run` 用 monkeypatch 掉 `subprocess.run` 校验调用。真实编码走 Task 2 的本地冒烟（无网络）。
- **防 STAGES 扩容触发回归**：注册 s7 后，既有跑满 STAGES 的 e2e 测试会跑到 s7→真 ffmpeg（FakeMedia 写的是无效图，会失败）。故注册时把那些 e2e **绑定 `to="s6_media"`**，s7 单独用 build_cmd 单测 + mock-run 单测 + 本地冒烟验。

## File Structure

```
新增：
  src/stages/s7_assemble.py     # 阶段⑦ ffmpeg 组装（build_cmd + run）
  tests/test_s7_assemble.py
修改：
  src/run.py                    # 注册 s7 + 把满range e2e 绑定 to=s6_media
  tests/test_run.py             # e2e 绑定修正
```

---

## Task 1: 阶段⑦ 组装 `src/stages/s7_assemble.py`

**Files:**
- Create: `src/stages/s7_assemble.py`, `tests/test_s7_assemble.py`

**Interfaces:**
- Produces:
  - `OUTPUT="video.mp4"`, `INPUT="timing.json"`, `run(ctx)`.
  - `build_cmd(job: Path, segments: list[dict], subs: Path, fps: int, w: int, h: int) -> list[str]` —— 纯函数，返回 ffmpeg argv。
- Consumes: `Ctx`、`read_json`。

- [ ] **Step 1: 写失败测试 `tests/test_s7_assemble.py`**
```python
from pathlib import Path
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.util.jobs import write_json
from src.stages.s7_assemble import build_cmd, run, OUTPUT, INPUT
import src.stages.s7_assemble as s7


def test_build_cmd_structure(tmp_path):
    segs = [
        {"id": 1, "audio": "seg_001.mp3", "dur_sec": 6.0},
        {"id": 2, "audio": "seg_002.mp3", "dur_sec": 7.0},
    ]
    cmd = build_cmd(tmp_path, segs, tmp_path / "subs.ass", fps=30, w=1920, h=1080)
    assert cmd[0] == "ffmpeg"
    assert cmd.count("-loop") == 2                        # 2 张图输入
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "concat=n=2:v=1:a=0" in fc                     # 拼 2 段视频
    assert "concat=n=2:v=0:a=1" in fc                     # 拼 2 段音频
    assert "subtitles=" in fc                             # 烧录字幕
    assert cmd[-1].endswith("video.mp4")
    assert "-t" in cmd and "6.0" in cmd and "7.0" in cmd  # 每段按 dur 展示


def test_run_invokes_ffmpeg(tmp_path, monkeypatch):
    ctx = Ctx(slug="c", config=Config({"video": {"fps": 30}}), root=tmp_path, llm=FakeLLM())
    write_json(ctx.job, INPUT, {"segments": [
        {"id": 1, "audio": "seg_001.mp3", "dur_sec": 1.0},
    ], "total_sec": 1.0})
    (ctx.job / "subs.ass").write_text("[Events]\n", encoding="utf-8")

    calls = {}

    def fake_run(cmd, **kw):
        calls["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"MP4")                 # 假装 ffmpeg 出了文件
        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(s7.subprocess, "run", fake_run)
    run(ctx)
    assert "ffmpeg" in calls["cmd"][0]
    assert calls["cmd"][-1] == str(ctx.job / OUTPUT)
    assert (ctx.job / OUTPUT).exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_s7_assemble.py -q`
Expected: FAIL（`No module named 'src.stages.s7_assemble'`）

- [ ] **Step 3: 写 `src/stages/s7_assemble.py`**
```python
from __future__ import annotations

import subprocess
from pathlib import Path

from src.context import Ctx
from src.util.jobs import read_json

OUTPUT = "video.mp4"
INPUT = "timing.json"


def build_cmd(job: Path, segments: list[dict], subs: Path,
              fps: int, w: int, h: int) -> list[str]:
    n = len(segments)
    cmd: list[str] = ["ffmpeg", "-y"]
    for seg in segments:                                  # 图片输入：每段按 dur 展示
        img = job / "media" / f"seg_{int(seg['id']):03d}.jpg"
        cmd += ["-loop", "1", "-t", f"{seg['dur_sec']}", "-i", str(img)]
    for seg in segments:                                  # 音频输入
        cmd += ["-i", str(job / "audio" / seg["audio"])]

    parts: list[str] = []
    for i in range(n):                                    # 每图 铺满 1080p
        parts.append(
            f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},setsar=1,fps={fps}[v{i}]"
        )
    parts.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vcat]")
    for i in range(n):                                    # 音频归一采样率
        parts.append(f"[{n + i}:a]aresample=44100[a{i}]")
    parts.append("".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[acat]")
    subs_esc = str(subs).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    parts.append(f"[vcat]subtitles='{subs_esc}'[vout]")

    cmd += [
        "-filter_complex", ";".join(parts),
        "-map", "[vout]", "-map", "[acat]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-r", str(fps), str(job / OUTPUT),
    ]
    return cmd


def run(ctx: Ctx) -> None:
    timing = read_json(ctx.job, INPUT)
    segments = timing.get("segments", [])
    fps = ctx.config.get("video.fps", 30)
    cmd = build_cmd(ctx.job, segments, ctx.job / "subs.ass", fps, 1920, 1080)
    subprocess.run(cmd, check=True, capture_output=True)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_s7_assemble.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**
```bash
git add src/stages/s7_assemble.py tests/test_s7_assemble.py
git commit -m "feat(s7): 阶段⑦ ffmpeg 组装 video.mp4（静态铺满+对轴+烧字幕）"
```

---

## Task 2: 注册 s7 + 绑定 e2e + 真实 ffmpeg 冒烟

**Files:**
- Modify: `src/run.py`, `tests/test_run.py`

**Interfaces:**
- Produces: `STAGES` 增加 s7；既有满 range e2e 绑定 `to="s6_media"` 防触发 s7 真 ffmpeg。

- [ ] **Step 1: 注册 s7 到 `src/run.py`**

import 区加入 `s7_assemble`；`STAGES` 在 `s6_media` 后加：
```python
    ("s7_assemble", s7_assemble.run, s7_assemble.OUTPUT),
```

- [ ] **Step 2: 绑定既有满 range e2e（防 s7 真 ffmpeg）—— 改 `tests/test_run.py`**

给 `test_e2e_through_media` 和 `test_e2e_through_subtitles` 的 `run_pipeline(...)` 调用各加 `to="s6_media"`（它们本就只验到素材/字幕，不该跑合成）。例如：
```python
    run_mod.run_pipeline("roswell-1947", cfg, tmp_path,
                         SeqLLM([topic, research, script]), to="s6_media")
```
（`test_e2e_through_subtitles` 同样加 `to="s6_media"`。）

- [ ] **Step 3: 全量测试**

Run: `.venv/bin/pytest -q`
Expected: 全绿（s7 单测 + 绑定后的 e2e，无一真跑 ffmpeg）。

> ⚠️ 审计：`grep -n "run_pipeline(" tests/ -r`，确认没有跑真实 STAGES 且未 `to=` 收尾、又会落到 s7 的测试。

- [ ] **Step 4: 真实 ffmpeg 冒烟（本地，无网络）—— 构造合法输入跑 s7 出真 mp4**

用 FakeTTS 的合法 WAV 当音频 + 兜底图当画面 + s5 出字幕，跑 s7：
```bash
mkdir -p jobs/_smoke7/media jobs/_smoke7/audio
cp assets/fallback.jpg jobs/_smoke7/media/seg_001.jpg
cp assets/fallback.jpg jobs/_smoke7/media/seg_002.jpg
.venv/bin/python - <<'PY'
from pathlib import Path
from src.tts import FakeTTS
from src.util.jobs import write_json
job = Path("jobs/_smoke7")
tts = FakeTTS()
texts = {1: "这是第一段旁白，飞碟出现在夜空。", 2: "这是第二段旁白，故事继续讲述。"}
segs = []; cursor = 0.0
for sid, text in texts.items():
    data, words = tts.synth(text)
    (job / "audio" / f"seg_{sid:03d}.mp3").write_bytes(data)
    dur = len(text) * 0.3
    segs.append({"id": sid, "audio": f"seg_{sid:03d}.mp3",
                 "start_sec": round(cursor, 3), "dur_sec": round(dur, 3),
                 "words": [{"text": c, "start_sec": round(cursor + j*0.3, 3), "dur_sec": 0.3}
                           for j, c in enumerate(text)]})
    cursor += dur
write_json(job, "timing.json", {"segments": segs, "total_sec": round(cursor, 3)})
write_json(job, "script.json", {"segments": [{"id": s, "text": t} for s, t in texts.items()]})
PY
# s5 出字幕
.venv/bin/python -c "from pathlib import Path;from src.config import Config;from src.llm import FakeLLM;from src.context import Ctx;from src.stages.s5_subs import run;run(Ctx('_smoke7',Config({}),Path('jobs'),FakeLLM()))"
# s7 真组装
.venv/bin/python -c "from pathlib import Path;from src.config import load_config;from src.llm import FakeLLM;from src.context import Ctx;from src.stages.s7_assemble import run;run(Ctx('_smoke7',load_config('config.yaml'),Path('jobs'),FakeLLM()))"
echo "=== video.mp4 ==="; ls -la jobs/_smoke7/video.mp4
echo "=== ffprobe 校验 ==="; ffprobe -v error -show_entries format=duration:stream=codec_type,codec_name,width,height -of default=noprint_wrappers=1 jobs/_smoke7/video.mp4
```
Expected: `video.mp4` 非空；ffprobe 报**有 video(h264,1920x1080) + audio(aac) 两条流**、`duration` ≈ timing.total_sec。真放能看到兜底星空图 + 静音配音 + 中文字幕对轴。若 ffmpeg 报错，按报错调 `build_cmd`（这是集成步骤，以真实 ffmpeg 为准），改完重跑本步。

- [ ] **Step 5: Commit**
```bash
git add src/run.py tests/test_run.py
git commit -m "feat(run): 注册阶段⑦ + e2e 绑定 to=s6_media（防合成触网/慢）"
```

---

## Self-Review

**1. Spec 覆盖（P4b 范围）：**
- 阶段⑦ video.mp4（spec §6⑦：图+配音+字幕合成）→ Task 1 ✅（BGM 闪避 / 片头尾 / Ken Burns 作紧随升级，MVP 先静态铺满，非漏洞）
- 时间轴以真实音频时长对轴（P4a 修）→ s7 用 timing.dur_sec ✅
- 防 STAGES 扩容触发 s7 真 ffmpeg 的测试回归 → Task 2 绑定 `to=s6_media` ✅
- 单测不真跑 ffmpeg（快）→ build_cmd 纯函数 + mock run；真编码走本地冒烟 ✅

**2. 占位符扫描：** 无 TBD；每步含完整代码+命令+预期。

**3. 类型一致性：** `build_cmd(job,segments,subs,fps,w,h)->list[str]` / `run(ctx)` / `OUTPUT=video.mp4`/`INPUT=timing.json` / s7 按 id 取 `media/seg_NNN.jpg`、按 timing 取 `audio/*` / STAGES 三元组 一致。

---

## 后续（P4b 之后）

- **Ken Burns 推拉**：把 s7 的静态 `[i:v]scale...` 换成 `zoompan`（缓慢推近），命令构造单测 + 冒烟。
- **BGM 闪避 + 片头尾**：加背景音乐（`sidechaincompress` 在旁白下压低）+ intro/outro 拼接。
- **署名**：P5 s8 用 `credits.json` 的 attribution 写进简介/片尾（满足 CC 署名）。
- **P5 发布**：s8 元数据/封面 → s9 YouTube 上传（护栏#3，`auto_upload:true`/`visibility:private`，一次性 OAuth）。

### P5 输入 / P4b 质量跟踪（来自 P4b 终审）

1. **【已做】ffmpeg stderr 可诊断**：s7 失败抛 `RuntimeError` 带 ffmpeg stderr（commit adb776c）。
2. **【已验】中文字幕真渲染**：从真实 mp4 抽帧目视确认 Noto CJK 经 libass 正常烧录（无豆腐块），render 环境已装 Noto Sans CJK。
3. **【§6⑦ 未完，跟踪】** MVP 是静态铺满；**Ken Burns(zoompan) / BGM 闪避 / 片头尾** 仍待做——§6⑦ 别算「完成」。
4. **【Minor】音频 concat 仅归一采样率**，未归一声道/格式：混源音频（单声道+立体声）会 concat 失败。加 `aformat=sample_fmts=fltp:channel_layouts=stereo`。
5. **【Minor】字幕路径转义**在含 `:`/`'` 的路径上脆弱（当前 job slug 简单不触发）；路径变用户可控时补。
6. **【Minor】s7 无 `n=0` 守卫**（空 segments → `concat=n=0` ffmpeg 拒）；真实恒 ≥1 段，加早返回更清晰。
7. **P5 s8_meta** 从 job-root 读 `video.mp4` + `credits.json`（attribution 署名）；s9_publish 上传 `video.mp4`。
