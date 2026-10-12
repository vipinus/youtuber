# 未解之谜 mystery 管线 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增第三条管线 `mystery`（m1–m8）：YouTube 检索热门未解之谜英文视频→只下字幕→LLM 深度改写成原创中文解说→LLM 分镜→zion 文生视频生成画面→CosyVoice2 固定参考音色克隆配音→按片种选 BGM→合成 8–15 分钟成片（本期不发布）。

**Architecture:** 仿 doc 管线的 stage 形态（每阶段一个产物文件，`run.py` 注册 `STAGES_MYSTERY`），最大化复用现有模块：m1 复用 `d1_source` 的 yt-dlp 帮手 + `ytsubs`，m5 复用 remote 克隆编排（新增固定参考模式），m6 复用 `d5_subs`/`s5_subs` 断句渲染，m7 复用 `d7_assemble` 的 `pick_bgm_track`/`build_assemble_cmd`/`nvenc_env`，m8 复用 `d8_meta.make_thumbnail`。新 GPU 能力（文生视频）走 `RemoteGPU` 协议新方法 `generate_video` + 新脚本 `scripts/zion_t2v.py`。

**Tech Stack:** Python(无类型标注风格同现库)、yt-dlp、ffmpeg、ollama(gemma4:31b-cloud)、diffusers LTX-Video(zion RTX 3070 Ti 8GB)、CosyVoice2、edge-tts 兜底、pytest。

**Spec:** `docs/superpowers/specs/2026-07-12-mystery-channel-design.md`

## Global Constraints

- 不碰既有 original/doc 管线的行为与测试；新代码全在新文件或明确列出的小改动点。
- 注释风格随现库：中文、记录"为什么/用户决策/踩坑"，不写"这行做什么"。
- 确定性：不用 `random`/时间戳做选择（同 slug 重跑同结果），选曲沿用 crc32 手法。
- best-effort 哲学：单段素材/配音失败回退不中止；选题/事实护栏失败 `StageBlocked` 硬停。
- 护栏：`guardrails.min_documented_facts`（默认 3）在 m2 强制执行。
- LLM 输出必须是 JSON **对象**（`src/util/llmjson.extract_json` 不解析顶层数组），批量结构一律 `{"items": [...]}` 或 `{"segments": [...]}`。
- zion GPU 脚本必须经 LD_LIBRARY_PATH wrapper 调用（`~/gpu-py` 一类），绝不直接跑裸 venv python。
- 纯函数（命令拼装、切分、解析）零 torch/网络依赖，可离线单测；真网络/GPU 函数测试必打桩。
- `assets/voice_ref/` 是克隆真人音色样本，**不进 git**（.gitignore），同 secrets 手动 scp 到 zion。
- 测试跑法：`cd /home/ddxs/Projects/youtuber && .venv/bin/python -m pytest tests/<file> -v`（下文简写 `pytest`）。
- 提交信息末尾带 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`。

## 产物契约总览（阶段间接口）

| 阶段 | OUTPUT | 结构 |
|---|---|---|
| m1_scout | `source.json`（另落 `source_subs.json3` + `transcript.json`） | `{"identifier":"yt:<id>","title","duration","view_count","channel","source_url","subs_segments"}`；transcript.json=`{"duration","segments":[{"start","end","text"}]}` |
| m2_script | `script.json` | `{"case_title","genre","facts":[...],"segments":[{"id":int,"zh":str}]}`，genre ∈ GENRES |
| m3_storyboard | `storyboard.json` | `{"segments":[{"id":int,"prompt":"english t2v prompt"}]}` |
| m4_media | `media.json`（另落 `media/seg_NNN.mp4|.png`） | `{"segments":[{"id","file","kind":"video"|"image"}]}` |
| m5_tts | `narration.json`（另落 `narration/nNNN.wav|.mp3`） | `{"segments":[{"id","file","tts_dur","zh","start","end","voice"}]}`，start/end 为**成片时间轴**（顺序累加，段间 0.35s） |
| m6_subs | `subs.ass` | ASS 字幕，时间轴同 narration |
| m7_assemble | `video.mp4`（中间产物 `base.mp4`/`narration.wav`/`bgm.json`） | 1080p30 成片 |
| m8_meta | `meta.json` + `thumb.jpg` | `{"title","description","tags"}` |

---

### Task 1: zion 文生视频脚本 + Spike A（LTX-Video 实测，决策门）

**Files:**
- Create: `scripts/zion_t2v.py`
- Test: `tests/test_zion_t2v.py`
- Create: `docs/superpowers/specs/2026-07-12-mystery-spike-notes.md`（实测数据记录）

**Interfaces:**
- Produces: CLI 契约 `<wrapper> zion_t2v.py "<prompt>" <out.mp4> [seconds] [WxH]`（Task 4 的 remote 编排按此拼命令）；纯函数 `parse_args(argv) -> tuple[str, str, float, tuple[int, int]]`、`num_frames(seconds, fps=24) -> int`（LTX 帧数须为 8n+1）。

- [ ] **Step 1: 写失败测试（纯函数，零 torch）**

```python
# tests/test_zion_t2v.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from zion_t2v import num_frames, parse_args


def test_num_frames_is_8n_plus_1():
    # LTX-Video 只接受 8n+1 帧；5s@24fps=120 → 最近的 8n+1 是 121
    assert num_frames(5.0) == 121
    assert (num_frames(3.3) - 1) % 8 == 0
    assert num_frames(0.2) >= 9          # 下限保护：至少一组


def test_parse_args_defaults_and_overrides():
    prompt, out, sec, (w, h) = parse_args(["zion_t2v.py", "a foggy night", "/tmp/o.mp4"])
    assert prompt == "a foggy night" and out == "/tmp/o.mp4"
    assert sec == 5.0 and (w, h) == (768, 512)
    _, _, sec2, (w2, h2) = parse_args(
        ["zion_t2v.py", "p", "o.mp4", "3", "704x480"])
    assert sec2 == 3.0 and (w2, h2) == (704, 480)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_zion_t2v.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'zion_t2v'`

- [ ] **Step 3: 写 scripts/zion_t2v.py**

```python
"""zion 端 LTX-Video 文生视频片段生成（mystery 管线 m4 的 GPU 侧）。

**必须**经 `~/t2v-py` 调用（t2v-venv 的 LD_LIBRARY_PATH wrapper，装法同
~/gpu-py，见 spike 笔记 docs/superpowers/specs/2026-07-12-mystery-spike-notes.md），
不能直接跑裸 venv python——同 zion_sdxl.py 的坑。

用法（在 zion 上）：
    ~/t2v-py zion_t2v.py "<prompt>" <out.mp4> [seconds] [WxH]
    默认 5 秒、768x512@24fps（LTX 原生舒适档，成片由 m7 统一放大到 1080p）。

8GB 显存配方（spike 实测数值见 spike 笔记）：bfloat16 + enable_model_cpu_offload
+ vae tiling。zion 同时是桌面机（KDE 常驻 ~3GB 显存），OOM 时先关桌面应用再试。
prompt 末尾恒追加 "no text"系负向词——SD 系画字是乱码（同 zion_sdxl 教训）。
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = "Lightricks/LTX-Video"
NEGATIVE_PROMPT = ("text, watermark, logo, subtitles, low quality, worst quality, "
                   "deformed, distorted, jittery, flickering")
FPS = 24
DEFAULT_SEC = 5.0
DEFAULT_W, DEFAULT_H = 768, 512


def num_frames(seconds: float, fps: int = FPS) -> int:
    """LTX-Video 帧数必须是 8n+1：取最接近 seconds*fps 的合法值，下限 9。"""
    n = max(round(seconds * fps / 8), 1)
    return n * 8 + 1


def parse_args(argv: list[str]) -> tuple[str, str, float, tuple[int, int]]:
    if len(argv) < 3:
        raise SystemExit("用法: zion_t2v.py <prompt> <out.mp4> [seconds] [WxH]")
    prompt, out = argv[1], argv[2]
    sec = float(argv[3]) if len(argv) > 3 else DEFAULT_SEC
    if len(argv) > 4:
        w, h = (int(x) for x in argv[4].lower().split("x"))
    else:
        w, h = DEFAULT_W, DEFAULT_H
    return prompt, out, sec, (w, h)


def generate(prompt: str, out_path: str, seconds: float, wh: tuple[int, int]) -> None:
    import torch
    from diffusers import LTXPipeline
    from diffusers.utils import export_to_video

    pipe = LTXPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()      # 8GB 显存：按需搬权重（同 zion_sdxl 配方）
    pipe.vae.enable_tiling()

    frames = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=wh[0], height=wh[1],
        num_frames=num_frames(seconds),
        num_inference_steps=40,
    ).frames[0]
    export_to_video(frames, out_path, fps=FPS)


if __name__ == "__main__":
    p, o, s, wh = parse_args(sys.argv)
    generate(p, o, s, wh)
    print("ZION_T2V_DONE")             # 编排侧探测成功标记（同 zion_gpu_asr 手法）
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_zion_t2v.py -v`
Expected: 2 PASS

- [ ] **Step 5: 提交代码**

```bash
git add scripts/zion_t2v.py tests/test_zion_t2v.py
git commit -m "feat(mystery): zion_t2v.py LTX-Video 文生视频脚本(纯函数可离线测)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Spike A——zion 装环境并实测出一段 5s 片段（决策门）**

在 zion 上建 venv + wrapper（一次性，照抄 gpu-py 模式）：

```bash
ssh fabric@zion 'python3 -m venv ~/t2v-venv && ~/t2v-venv/bin/pip install -U pip && ~/t2v-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124 && ~/t2v-venv/bin/pip install diffusers transformers accelerate sentencepiece imageio imageio-ffmpeg'
ssh fabric@zion 'printf "#!/bin/sh\nexport LD_LIBRARY_PATH=/run/opengl-driver/lib:\$LD_LIBRARY_PATH\nexec ~/t2v-venv/bin/python \"\$@\"\n" > ~/t2v-py && chmod +x ~/t2v-py'
scp scripts/zion_t2v.py fabric@zion:~/p7-work/zion_t2v.py
ssh fabric@zion '~/t2v-py ~/p7-work/zion_t2v.py "abandoned house in fog at night, dark moody documentary cinematography" ~/p7-work/t2v_test.mp4 5'
```

Expected: 末行输出 `ZION_T2V_DONE`；`scp fabric@zion:~/p7-work/t2v_test.mp4 /tmp/` 拉回人眼看画面可用。记录：显存峰值（`nvidia-smi` 另开 ssh 观察）、耗时、首次模型下载体积。

- [ ] **Step 7: 记录 spike 结果 + 决策**

把实测数字写进 `docs/superpowers/specs/2026-07-12-mystery-spike-notes.md`（模型、分辨率、秒数、显存峰值、单段耗时、60–90 段一集的外推总时长）。**决策门**：
- 通过（能出片、单段 ≤6min）→ 继续 Task 2。
- OOM/过慢 → 换 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`（`WanPipeline`，480p 档）重测一次；再不行**停下来找用户**回改设计（退 SDXL 静图+动效方案）。

```bash
git add docs/superpowers/specs/2026-07-12-mystery-spike-notes.md
git commit -m "docs(mystery): spike A LTX-Video zion 实测数据" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: 参考音色制作脚本 + 产出 assets/voice_ref

**Files:**
- Create: `scripts/make_voice_ref.py`
- Modify: `.gitignore`（追加 `assets/voice_ref/`）
- Test: `tests/test_make_voice_ref.py`

**Interfaces:**
- Produces: `assets/voice_ref/ref.wav`（16kHz mono、10–15s 干净人声）+ `assets/voice_ref/ref.txt`（该切片的中文原文，CosyVoice2 zero_shot 需要）；纯函数 `build_slice_cmd(src, out, start, end) -> list[str]`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_make_voice_ref.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_voice_ref import build_slice_cmd


def test_build_slice_cmd_slices_cleans_and_resamples():
    cmd = build_slice_cmd("/a/in.wav", "/b/ref.wav", 12.5, 26.0)
    assert cmd[0] == "ffmpeg" and cmd[-1] == "/b/ref.wav"
    i = cmd.index("-ss")
    assert cmd[i + 1] == "12.5" and cmd[cmd.index("-to") + 1] == "26.0"
    af = cmd[cmd.index("-af") + 1]
    # 同 zion_cosy_tts 生产链的清洗配方：高通去嗡 + afftdn 降噪；16k 单声道
    assert "highpass" in af and "afftdn" in af
    assert cmd[cmd.index("-ar") + 1] == "16000" and cmd[cmd.index("-ac") + 1] == "1"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_make_voice_ref.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: 写 scripts/make_voice_ref.py**

```python
"""一次性预处理：从参考视频做 CosyVoice2 固定参考音色（mystery 频道配音）。

用户拍板(2026-07-12)：克隆「科幻FANS」解说声(https://www.youtube.com/watch?v=Ilfls0FTJoM)，
风险已提示照做；引擎 CosyVoice2(不要 F5)。中文参考→中文旁白=同语种 zero_shot，
需要参考音频**和它的原文文本**(ref.txt)。

用法（人工监督跑，不进管线）：
    python scripts/make_voice_ref.py <youtube_url> --start 12.5 --end 26.0 \
        [--text "切片的中文原文"] [--out assets/voice_ref]

流程：yt-dlp 抽音频 → 按 --start/--end 切片+清洗(highpass+afftdn，克隆引擎会
连声学环境一起复刻——同 zion_cosy_tts 教训)+重采样 16k mono → ref.wav；
--text 给了就直接写 ref.txt，没给则尝试 faster-whisper(CPU, base)转写，
两者都没有就报错让人工补。**产物必须人耳听一遍**再投产：切片里不能混 BGM/
音效，否则克隆出咔咔声(能量门教训)。

assets/voice_ref/ 不进 git(.gitignore)——真人音色样本不入库，同 secrets
手动 scp 到 zion：scp -r assets/voice_ref fabric@zion:~/youtuber/assets/
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_slice_cmd(src: str, out: str, start: float, end: float) -> list[str]:
    return [
        "ffmpeg", "-y", "-i", str(src), "-ss", str(start), "-to", str(end),
        "-af", "highpass=f=80,afftdn=nf=-25",
        "-ar", "16000", "-ac", "1", str(out),
    ]


def _download_audio(url: str, dest: Path) -> None:
    subprocess.run(
        ["yt-dlp", "-x", "--audio-format", "wav", "-o", str(dest),
         "--no-playlist", url],
        check=True, timeout=600,
    )


def _transcribe(wav: Path) -> str:
    from faster_whisper import WhisperModel          # 惰性：没装就走 --text 人工路径

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segs, _info = model.transcribe(str(wav), language="zh")
    return "".join(s.text for s in segs).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="制作 CosyVoice2 固定参考音色")
    ap.add_argument("url")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--text", default="")
    ap.add_argument("--out", default="assets/voice_ref")
    args = ap.parse_args()

    if not 8.0 <= args.end - args.start <= 20.0:
        sys.exit("切片时长须在 8~20s（太短抓不住音色，太长拖慢推理）")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    full = out_dir / "_full.wav"
    if not full.exists():                            # 重试换切点时不重复下载
        _download_audio(args.url, full)
    ref = out_dir / "ref.wav"
    subprocess.run(build_slice_cmd(str(full), str(ref), args.start, args.end),
                   check=True, capture_output=True)

    text = args.text.strip()
    if not text:
        try:
            text = _transcribe(ref)
        except ImportError:
            sys.exit("未装 faster-whisper：请用 --text 手动给出切片原文")
    if not text:
        sys.exit("转写为空：换一段有清晰人声的切点，或用 --text 手动给出")
    (out_dir / "ref.txt").write_text(text, encoding="utf-8")
    print(f"OK ref={ref} text={text!r}\n务必人耳听一遍 {ref}：不能混 BGM/音效")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_make_voice_ref.py -v`
Expected: PASS

- [ ] **Step 5: .gitignore + 提交**

在 `.gitignore` 追加一行 `assets/voice_ref/`，然后：

```bash
git add scripts/make_voice_ref.py tests/test_make_voice_ref.py .gitignore
git commit -m "feat(mystery): make_voice_ref 参考音色制作脚本(voice_ref 不入库)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: 真跑产出参考音色（人工监督）**

先听原视频挑一段 10–15s **纯解说无 BGM 高潮**的区间（记下秒数），然后：

```bash
.venv/bin/python scripts/make_voice_ref.py "https://www.youtube.com/watch?v=Ilfls0FTJoM" --start <s> --end <e>
```

Expected: 打印 `OK ref=assets/voice_ref/ref.wav text='...'`。人耳听 ref.wav 确认干净、读 ref.txt 确认与音频逐字一致（不一致会毁克隆质量——F5 时代的同款教训）。不干净就换切点重跑。最后同步到 zion：

```bash
scp -r assets/voice_ref fabric@zion:~/youtuber/assets/
```

---

### Task 3: zion_cosy_tts.py 固定参考模式 + Spike B（克隆验收，决策门）

**Files:**
- Modify: `scripts/zion_cosy_tts.py`
- Test: `tests/test_zion_cosy_fixed_ref.py`

**Interfaces:**
- Produces: CLI 向后兼容扩展 `~/cosy-py zion_cosy_tts.py <zh_script.json> <ref_or_vocals> <out_dir> [ref_text.txt]`——第 4 个参数存在即"固定参考模式"：所有段用同一份参考 wav（arg2 整文件）+ ref_text 走 `inference_zero_shot`（同语种），不再逐段 `pick_reference`/`inference_cross_lingual`。产物契约不变：`out_dir/n{id:03d}.wav` + 末行 `ZION_CLONE_DONE`。
- Consumes: Task 2 的 `ref.wav`/`ref.txt`。

- [ ] **Step 1: 写失败测试（纯函数）**

```python
# tests/test_zion_cosy_fixed_ref.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from zion_cosy_tts import parse_mode


def test_parse_mode_per_speaker_when_three_args():
    m = parse_mode(["zion_cosy_tts.py", "zh.json", "vocals.mp3", "out"])
    assert m == {"mode": "per_speaker", "zh_script": "zh.json",
                 "audio": "vocals.mp3", "out_dir": "out", "ref_text": None}


def test_parse_mode_fixed_when_ref_text_given():
    m = parse_mode(["zion_cosy_tts.py", "zh.json", "ref.wav", "out", "ref.txt"])
    assert m["mode"] == "fixed" and m["ref_text"] == "ref.txt"
    assert m["audio"] == "ref.wav"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_zion_cosy_fixed_ref.py -v`
Expected: FAIL `ImportError: cannot import name 'parse_mode'`

- [ ] **Step 3: 改 scripts/zion_cosy_tts.py**

先读该文件全文再动手（本 Task 的实现者必须看过 main() 的参考选择段落）。三处改动：

(a) 模块顶部（`REF_MAX_SILENCE_DB` 常量之后）加纯函数：

```python
def parse_mode(argv: list[str]) -> dict:
    """CLI 双模式：3 参=既有逐段 per_speaker 克隆(参考=vocals 切片)；4 参=固定
    参考模式(mystery 频道:参考=arg2 整文件+ref_text，同语种 zero_shot)。"""
    if len(argv) < 4:
        raise SystemExit(
            "用法: zion_cosy_tts.py <zh_script.json> <ref_or_vocals> <out_dir> [ref_text.txt]")
    return {
        "mode": "fixed" if len(argv) > 4 else "per_speaker",
        "zh_script": argv[1], "audio": argv[2], "out_dir": argv[3],
        "ref_text": argv[4] if len(argv) > 4 else None,
    }
```

(b) 加固定参考的预处理（复用文件里已有的清洗手法，放 `_slice_reference_16k` 旁边）：

```python
def _prep_fixed_reference(ref_wav: Path, tmp_dir: Path) -> Path:
    """固定参考:整文件转 16k mono+同款降噪清洗，只做一次全段共用。
    ref.wav 制作时(make_voice_ref)已清洗过，这里再过一遍是幂等保险，
    顺带兜住用户手工放进来的未清洗样本。"""
    out = tmp_dir / "fixed_ref_16k.wav"
    if out.exists():
        return out
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(ref_wav),
         "-af", "highpass=f=80,afftdn=nf=-25", "-ar", "16000", "-ac", "1", str(out)],
        check=True, capture_output=True)
    return out
```

(c) main() 里：入口改用 `parse_mode(sys.argv)`；在逐段合成循环中，`mode=="fixed"` 时跳过 `pick_reference`/`_slice_reference_16k`，改为一次性 `ref16k = _prep_fixed_reference(...)`、`ref_text = Path(m["ref_text"]).read_text(encoding="utf-8").strip()`，每段调 CosyVoice2 的 `inference_zero_shot(normalize_zh_numbers(zh), ref_text, ref16k路径)`（本仓 CosyVoice 版本 frontend 内部自行 load_wav，传 **wav 路径**——同文件头备忘）。单段失败仍只 warn 跳过；`ZION_CLONE_DONE` 收尾逻辑不动。per_speaker 分支代码零改动。

- [ ] **Step 4: 跑测试确认通过（含既有测试不回归）**

Run: `pytest tests/test_zion_cosy_fixed_ref.py tests/test_zion_clone_tts.py -v`
Expected: 全 PASS

- [ ] **Step 5: 提交**

```bash
git add scripts/zion_cosy_tts.py tests/test_zion_cosy_fixed_ref.py
git commit -m "feat(mystery): zion_cosy_tts 固定参考模式(全片同一克隆音色)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Spike B——zion 上克隆两句人耳验收（决策门）**

```bash
scp scripts/zion_cosy_tts.py fabric@zion:~/p7-work/
ssh fabric@zion 'cd ~/p7-work && cat > spike_zh.json <<EOF
{"segments":[{"id":0,"zh":"一九四七年七月，新墨西哥州的牧场上出现了不明残骸。"},{"id":1,"zh":"官方的说法在二十四小时内改变了三次，这正是谜团的开始。"}]}
EOF
mkdir -p spike_out && ~/cosy-py zion_cosy_tts.py spike_zh.json ~/youtuber/assets/voice_ref/ref.wav spike_out ~/youtuber/assets/voice_ref/ref.txt'
scp 'fabric@zion:~/p7-work/spike_out/n*.wav' /tmp/
```

Expected: 末行 `ZION_CLONE_DONE`，拉回 n000.wav/n001.wav。**给用户听**：音色像不像科幻FANS、有没有杂音。不合格→换参考切片（回 Task 2 Step 6）重试；连续不合格→停下来找用户。把结论追加进 spike 笔记并提交（同 Task 1 Step 7 的提交方式）。

---

### Task 4: remote.py 新能力——generate_video + 固定参考克隆 + SSH 侧 cosy 分派

**Files:**
- Modify: `src/remote.py`
- Test: `tests/test_remote_t2v.py`

**Interfaces:**
- Produces: `RemoteGPU` 协议新增 `generate_video(prompt: str, out_path: Path, seconds: float) -> Path`；`clone_synthesize(zh_script, vocals, out_dir, ref_text: Path | None = None)`（None=老行为）。Fake/SSH/Local 三实现齐。`get_remote` 新读配置键：`remote.t2v_wrapper`(默认 `~/t2v-py`)、`remote.t2v_timeout_sec`(默认 900)；SSH 分支补上 cosy 分派（`tts.provider=="cosyvoice"` 时 clone_wrapper 取 `remote.cosy_wrapper` 默认 `~/cosy-py`、克隆脚本用 `_ZION_COSY_SCRIPT_NAME`——此前只有 local 分支会分派 cosy，笔记本 ssh 模式跑不了 CosyVoice，本任务顺手补齐）。
- Consumes: Task 1 的 zion_t2v.py CLI 契约、Task 3 的 4 参 CLI 契约。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_remote_t2v.py
from __future__ import annotations

import subprocess
from pathlib import Path

import src.remote as remote
from src.config import Config


def test_fake_generate_video_writes_decodable_mp4(tmp_path):
    out = tmp_path / "seg.mp4"
    remote.FakeRemoteGPU().generate_video("fog", out, seconds=1.0)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(out)], capture_output=True, text=True)
    assert probe.returncode == 0 and float(probe.stdout.strip()) >= 0.5


def test_local_generate_video_builds_wrapper_cmd(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "_run_checked",
                        lambda cmd, *a, **k: calls.append(cmd))
    gpu = remote.LocalGPU(t2v_wrapper="/x/t2v-py", repo_root=tmp_path)
    (tmp_path / "scripts").mkdir()
    gpu.generate_video("dark alley", tmp_path / "o.mp4", seconds=5.0)
    cmd = calls[0]
    assert cmd[0] == "/x/t2v-py" and cmd[1].endswith("zion_t2v.py")
    assert cmd[2] == "dark alley" and cmd[4] == "5.0"


def test_local_clone_synthesize_appends_ref_text(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "_run_checked",
                        lambda cmd, *a, **k: calls.append(cmd))
    gpu = remote.LocalGPU(repo_root=tmp_path)
    (tmp_path / "scripts").mkdir()
    gpu.clone_synthesize(tmp_path / "s.json", tmp_path / "ref.wav",
                         tmp_path / "out", ref_text=tmp_path / "ref.txt")
    assert calls[0][-1].endswith("ref.txt")
    calls.clear()
    gpu.clone_synthesize(tmp_path / "s.json", tmp_path / "v.mp3", tmp_path / "out")
    assert not calls[0][-1].endswith("ref.txt")     # 不传=老行为，命令不变


def test_get_remote_ssh_dispatches_cosy_script():
    cfg = Config({"remote": {"provider": "ssh"}, "tts": {"provider": "cosyvoice"}})
    gpu = remote.get_remote(cfg)
    assert gpu.clone_script == "zion_cosy_tts.py"
    assert gpu.clone_wrapper == "~/cosy-py"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_remote_t2v.py -v`
Expected: FAIL（`generate_video` 不存在 / `clone_script` 属性不存在）

- [ ] **Step 3: 改 src/remote.py**

模块级常量区加 `_ZION_T2V_SCRIPT_NAME = "zion_t2v.py"`。

Protocol 补两行：

```python
class RemoteGPU(Protocol):
    def transcribe_separate(self, audio: Path, out_dir: Path) -> dict: ...
    def generate_image(self, prompt: str, out_path: Path) -> Path: ...
    def clone_synthesize(self, zh_script: Path, vocals: Path, out_dir: Path,
                         ref_text: Path | None = None) -> Path: ...
    def generate_video(self, prompt: str, out_path: Path, seconds: float) -> Path: ...
```

`FakeRemoteGPU` 加（clone_synthesize 签名补 `ref_text: Path | None = None`，体内不用它）：

```python
    def generate_video(self, prompt: str, out_path: Path, seconds: float) -> Path:
        """离线确定性：ffmpeg testsrc2 现场造一段真正可解码的小 mp4（同
        transcribe_separate 用 lavfi 的教训——占位字节会把下游 ffmpeg 搞崩）。"""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i",
             f"testsrc2=size=320x180:rate=24:duration={seconds}",
             "-pix_fmt", "yuv420p", str(out_path)],
            check=True, capture_output=True)
        return out_path
```

`SSHRemoteGPU.__init__` 加参数 `t2v_wrapper: str = "~/t2v-py"`、`t2v_timeout_sec: int = 900`、`clone_script: str = _ZION_CLONE_SCRIPT_NAME`（存 self；`clone_synthesize` 里 `self._deploy_script(_ZION_CLONE_SCRIPT_NAME)` 改成 `self._deploy_script(self.clone_script)`）。加方法：

```python
    def generate_video(self, prompt: str, out_path: Path, seconds: float) -> Path:
        """同 generate_image 编排：部署 zion_t2v.py、经 t2v wrapper 跑、scp 回。"""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        remote_script = self._deploy_script(_ZION_T2V_SCRIPT_NAME)
        remote_out = f"{self.workdir}/{out_path.name}"
        self._ssh(f"{self.t2v_wrapper} {remote_script} {shlex.quote(prompt)} "
                  f"{remote_out} {seconds}", timeout_sec=self.t2v_timeout_sec)
        self._scp(f"{self.host}:{remote_out}", str(out_path),
                  timeout_sec=self.t2v_timeout_sec)
        return out_path
```

`SSHRemoteGPU.clone_synthesize` 签名补 `ref_text: Path | None = None`；体内在 scp zh_script/vocals 之后加：

```python
        ref_arg = ""
        if ref_text is not None:
            remote_ref_text = f"{self.workdir}/{Path(ref_text).name}"
            self._scp(str(ref_text), f"{self.host}:{remote_ref_text}")
            ref_arg = f" {remote_ref_text}"
```

并把跑脚本那条 `self._ssh(...)` 的命令串末尾拼上 `{ref_arg}`。

`LocalGPU.__init__` 加 `t2v_wrapper: str = "~/t2v-py"`、`t2v_timeout_sec: int = 900`（`self.t2v_wrapper = str(Path(t2v_wrapper).expanduser())`）。加方法：

```python
    def generate_video(self, prompt: str, out_path: Path, seconds: float) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        script = self.repo_root / "scripts" / _ZION_T2V_SCRIPT_NAME
        cmd = [self.t2v_wrapper, str(script), prompt, str(out_path), str(seconds)]
        _run_checked(
            cmd, self.t2v_timeout_sec,
            timeout_msg=f"local t2v 超时 {self.t2v_timeout_sec}s: {cmd!r}",
            fail_msg_prefix=f"local t2v 失败: {cmd!r}")
        return out_path
```

`LocalGPU.clone_synthesize` 签名补 `ref_text: Path | None = None`，`cmd` 组装后加：

```python
        if ref_text is not None:
            cmd.append(str(ref_text))
```

`get_remote`：ssh 分支加 cosy 分派与新键（与 local 分支同款逻辑）：

```python
    cosy = config.get("tts.provider") == "cosyvoice"
    if provider == "ssh":
        return SSHRemoteGPU(
            host=config.get("remote.host", "fabric@zion"),
            workdir=config.get("remote.workdir", "~/p7-work"),
            whisper_model=config.get("remote.whisper_model", "medium.en"),
            timeout_sec=config.get("remote.timeout_sec", 1200),
            clone_wrapper=(config.get("remote.cosy_wrapper", "~/cosy-py") if cosy
                           else config.get("remote.clone_wrapper", "~/clone-py")),
            clone_script=(_ZION_COSY_SCRIPT_NAME if cosy else _ZION_CLONE_SCRIPT_NAME),
            clone_timeout_sec=config.get("remote.clone_timeout_sec", 7200),
            t2v_wrapper=config.get("remote.t2v_wrapper", "~/t2v-py"),
            t2v_timeout_sec=config.get("remote.t2v_timeout_sec", 900),
        )
```

local 分支同样补 `t2v_wrapper=config.get("remote.t2v_wrapper", "~/t2v-py")`、`t2v_timeout_sec=config.get("remote.t2v_timeout_sec", 900)`（cosy 变量提到分支外共用）。

- [ ] **Step 4: 跑测试确认通过（含既有 remote 测试不回归）**

Run: `pytest tests/test_remote_t2v.py tests/test_remote.py tests/test_d4_tts.py -v`
Expected: 全 PASS

- [ ] **Step 5: 提交**

```bash
git add src/remote.py tests/test_remote_t2v.py
git commit -m "feat(mystery): remote generate_video + 固定参考克隆 + ssh侧cosy分派" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: m1_scout——YouTube 热门未解之谜检索 + 只下字幕

**Files:**
- Create: `src/stages/m1_scout.py`
- Test: `tests/test_m1_scout.py`

**Interfaces:**
- Consumes: `d1_source` 的 `_set_ytdlp_extra`/`_yt_download_subs`/`_yt_video_stats`（模块级复用）、`ytsubs.parse_json3_segments`、registry。
- Produces: `source.json` + `source_subs.json3` + `transcript.json`（契约见总览表）；模块常量 `OUTPUT = "source.json"`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_m1_scout.py
from __future__ import annotations

import json

import pytest

from src.config import Config
from src.context import Ctx
from src.llm import FakeLLM
from src.util.jobs import read_json, write_json

import src.stages.m1_scout as m1


def _ctx(tmp_path, cfg=None):
    return Ctx(slug="mys1", config=Config(cfg or {}), root=tmp_path, llm=FakeLLM())


def _entry(vid, title, dur, views):
    return {"id": vid, "title": title, "duration": dur, "view_count": views,
            "channel": "MysteryDocs"}


def _fake_subs(job, n=40):
    """写一份足量的 json3 字幕（n 句，每句带句号会被 parse 成 n 段）。"""
    events = [{"tStartMs": i * 2000, "dDurationMs": 1800,
               "segs": [{"utf8": f"Fact number {i}."}]} for i in range(n)]
    (job / "source_subs.json3").write_text(
        json.dumps({"events": events}), encoding="utf-8")
    return True


def test_picks_most_viewed_with_enough_subs(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(m1, "_yt_search", lambda *a: [
        _entry("low1", "Mystery A", 900, 1_000),
        _entry("hot1", "Mystery B", 900, 9_000_000),
    ])
    monkeypatch.setattr(m1, "_yt_download_subs",
                        lambda ytdlp, vid, job: _fake_subs(job))
    m1.run(ctx)
    src = read_json(ctx.job, "source.json")
    assert src["identifier"] == "yt:hot1"          # 播放量最高者先
    tr = read_json(ctx.job, "transcript.json")
    assert len(tr["segments"]) >= 30 and tr["duration"] == 900


def test_skips_registry_short_live_and_subless(tmp_path, monkeypatch):
    write_json(tmp_path, "published.json", None) if False else None
    (tmp_path / "published.json").write_text(
        json.dumps([{"identifier": "yt:seen"}]), encoding="utf-8")
    ctx = _ctx(tmp_path, {"mystery": {"min_sec": 480}})
    monkeypatch.setattr(m1, "_yt_search", lambda *a: [
        _entry("seen", "Old", 900, 9_999_999),                     # registry 去重
        _entry("live", "LIVE mystery stream replay", 900, 8_888),  # 直播词
        _entry("shrt", "Tiny", 60, 7_777),                         # 时长门
        _entry("nsub", "No subs", 900, 6_666),                     # 无字幕
        _entry("good", "Real doc", 900, 5_555),
    ])
    monkeypatch.setattr(
        m1, "_yt_download_subs",
        lambda ytdlp, vid, job: _fake_subs(job) if vid == "good" else False)
    m1.run(ctx)
    assert read_json(ctx.job, "source.json")["identifier"] == "yt:good"


def test_blocks_when_no_candidate(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(m1, "_yt_search", lambda *a: [])
    with pytest.raises(RuntimeError):
        m1.run(ctx)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_m1_scout.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'src.stages.m1_scout'`

- [ ] **Step 3: 写 src/stages/m1_scout.py**

```python
"""mystery 阶段①——YouTube 检索最受欢迎的未解之谜英文视频，**只下字幕不下视频**。

版权口径(用户拍板 2026-07-12)：热门片有版权，视频/音频一律不碰；字幕仅作
m2 深度改写的事实素材（成片文稿/画面/配音全为新生成的原创内容）。因此这里
没有 d1 的许可门/HD 门/音轨探测——没有要重发布的素材，只有参考文本。

复用 d1_source 的 yt-dlp 帮手（cookie 处理/字幕下载/逐条拉播放量）与
ytsubs.parse_json3_segments（json3→句级段）。registry 去重沿用 yt:<id>
identifier 形态，与 doc 管线共用一份登记表，两条线不会互相撞片。
"""
from __future__ import annotations

import json
import re
import subprocess

from src.context import Ctx
from src.stages import d1_source as d1
from src.stages.d1_source import _set_ytdlp_extra, _yt_download_subs, _yt_video_stats
from src.util.jobs import write_json
from src.util.registry import in_flight_identifiers, published_identifiers
from src.util.ytsubs import parse_json3_segments

OUTPUT = "source.json"
SUBS_FILE = "source_subs.json3"
TRANSCRIPT = "transcript.json"

DEFAULT_QUERY = "unsolved mystery documentary"
DEFAULT_ROWS = 50
DEFAULT_MIN_SEC = 480          # 8min：太短撑不起 8-15min 改写稿的事实量
DEFAULT_MAX_SEC = 3600
DEFAULT_MIN_SEGMENTS = 30      # 字幕句数下限：低于它事实密度不够，m2 会难产

_LIVE_RE = re.compile(r"\blive\b|\blivestream\b|\breplay\b", re.IGNORECASE)


def _yt_search(ytdlp: str, query: str, rows: int) -> list[dict]:
    """yt-dlp ytsearch 扁平列表。真网络，测试必打桩（同 d1._yt_list_channel）。"""
    out = subprocess.run(
        [ytdlp, *d1._YTDLP_EXTRA, "--flat-playlist", "-J",
         f"ytsearch{rows}:{query}"],
        capture_output=True, text=True, timeout=600, check=True)
    data = json.loads(out.stdout)
    return [e for e in (data.get("entries") or []) if e]


def run(ctx: Ctx) -> None:
    cfg = ctx.config
    ctx.job.mkdir(parents=True, exist_ok=True)
    from pathlib import Path
    ytdlp = str(Path(str(cfg.get("p7.ytdlp_bin", "yt-dlp"))).expanduser())
    _set_ytdlp_extra(cfg)                       # cookie/EJS 解算器配置与 d1 共用
    query = cfg.get("mystery.query", DEFAULT_QUERY)
    rows = int(cfg.get("mystery.rows", DEFAULT_ROWS))
    min_sec = cfg.get("mystery.min_sec", DEFAULT_MIN_SEC)
    max_sec = cfg.get("mystery.max_sec", DEFAULT_MAX_SEC)
    min_segments = int(cfg.get("mystery.min_segments", DEFAULT_MIN_SEGMENTS))
    avoid = [str(k).lower() for k in (cfg.get("mystery.avoid_keywords", []) or [])]

    skip_ids = published_identifiers(ctx.root) | in_flight_identifiers(ctx.root, ctx.slug)

    survivors: list[dict] = []
    for e in _yt_search(ytdlp, query, rows):
        vid = e.get("id")
        if not vid or f"yt:{vid}" in skip_ids:
            continue
        title = str(e.get("title") or vid)
        if e.get("live_status") in ("is_live", "was_live", "is_upcoming", "post_live"):
            continue
        if _LIVE_RE.search(title):
            continue
        if avoid and any(k in title.lower() for k in avoid):
            continue
        dur = float(e.get("duration") or 0)
        if not (min_sec <= dur <= max_sec):
            continue
        survivors.append(e)

    for e in survivors:                          # 扁平搜索缺 view_count 时逐条补
        if not e.get("view_count"):
            v, _l = _yt_video_stats(ytdlp, e["id"])
            e["view_count"] = v
    survivors.sort(key=lambda e: e.get("view_count") or 0, reverse=True)

    for e in survivors:
        vid = e["id"]
        if not _yt_download_subs(ytdlp, vid, ctx.job):
            ctx.warn(f"m1_scout: yt:{vid} 无英文字幕，跳过（字幕是唯一素材源）")
            continue
        segments = parse_json3_segments(ctx.job / SUBS_FILE)
        if len(segments) < min_segments:
            ctx.warn(f"m1_scout: yt:{vid} 字幕仅 {len(segments)} 段(<{min_segments})，"
                     "事实密度不够，跳过")
            (ctx.job / SUBS_FILE).unlink(missing_ok=True)
            continue
        dur = float(e.get("duration") or 0)
        write_json(ctx.job, TRANSCRIPT, {"duration": dur, "segments": segments})
        write_json(ctx.job, OUTPUT, {
            "identifier": f"yt:{vid}",
            "title": str(e.get("title") or vid),
            "duration": dur,
            "view_count": e.get("view_count"),
            "channel": e.get("channel") or e.get("uploader"),
            "source_url": f"https://www.youtube.com/watch?v={vid}",
            "subs_segments": len(segments),
        })
        return

    raise RuntimeError(
        f"m1_scout: 检索无合格候选(query={query!r}, {min_sec}-{max_sec}s, "
        f"字幕≥{min_segments}段) — 换 mystery.query 或放宽门槛后重试")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_m1_scout.py -v`
Expected: 3 PASS

- [ ] **Step 5: 提交**

```bash
git add src/stages/m1_scout.py tests/test_m1_scout.py
git commit -m "feat(mystery): m1_scout YouTube热门检索+只下字幕" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: m2_script——事实提取(护栏) + 原创中文解说稿改写

**Files:**
- Create: `src/stages/m2_script.py`
- Test: `tests/test_m2_script.py`

**Interfaces:**
- Consumes: `transcript.json`、`source.json`、`ctx.llm.generate(system, prompt)`、`extract_json`、`StageBlocked`(src/errors.py)。
- Produces: `script.json`（`{"case_title","genre","facts","segments":[{"id","zh"}]}`）；常量 `GENRES = ("mystery", "paranormal", "crime", "horror", "scifi", "history")`（m7 选 BGM 用同一枚举）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_m2_script.py
from __future__ import annotations

import json

import pytest

from src.config import Config
from src.context import Ctx
from src.errors import StageBlocked
from src.llm import FakeLLM
from src.util.jobs import read_json, write_json

import src.stages.m2_script as m2

CANNED = json.dumps({
    "case_title": "罗斯威尔事件", "genre": "mystery",
    "facts": ["1947年7月牧场发现残骸", "军方24小时内改口", "目击者证词互相矛盾"],
    "segments": [{"id": 0, "zh": "第一段解说。"}, {"id": 7, "zh": "第二段解说。"}],
}, ensure_ascii=False)


def _ctx(tmp_path, canned=CANNED, cfg=None):
    ctx = Ctx(slug="mys1", config=Config(cfg or {}), root=tmp_path,
              llm=FakeLLM(canned=canned))
    write_json(ctx.job, "transcript.json", {"duration": 900.0, "segments": [
        {"start": 0.0, "end": 2.0, "text": "In July 1947 a rancher found debris."},
        {"start": 2.0, "end": 4.0, "text": "The army changed its story."},
    ]})
    write_json(ctx.job, "source.json", {"identifier": "yt:x", "title": "Roswell"})
    return ctx


def test_writes_script_with_renumbered_segments_and_genre(tmp_path):
    ctx = _ctx(tmp_path)
    m2.run(ctx)
    out = read_json(ctx.job, m2.OUTPUT)
    assert out["genre"] == "mystery" and len(out["facts"]) == 3
    assert [s["id"] for s in out["segments"]] == [0, 1]   # id 重编号成连续序
    assert out["segments"][0]["zh"] == "第一段解说。"


def test_blocked_when_facts_below_guardrail(tmp_path):
    canned = json.dumps({"case_title": "x", "genre": "mystery",
                         "facts": ["只有一条"], "segments": [{"id": 0, "zh": "a。"}]},
                        ensure_ascii=False)
    ctx = _ctx(tmp_path, canned=canned,
               cfg={"guardrails": {"min_documented_facts": 3}})
    with pytest.raises(StageBlocked):
        m2.run(ctx)


def test_unknown_genre_falls_back_to_mystery(tmp_path):
    canned = CANNED.replace('"mystery"', '"telenovela"')
    ctx = _ctx(tmp_path, canned=canned)
    m2.run(ctx)
    assert read_json(ctx.job, m2.OUTPUT)["genre"] == "mystery"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_m2_script.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: 写 src/stages/m2_script.py**

```python
"""mystery 阶段②——从英文字幕提取案件事实(护栏)，再**深度改写**成原创中文解说稿。

版权口径(用户拍板 2026-07-12)：绝不逐句翻译原字幕（衍生作品风险）——字幕只当
事实清单的来源，改写稿是以事实为素材的独立叙事，不保留原片的句子/段落对应。
两步各一次 LLM 调用（gemma4:31b-cloud）：
  ①事实+题材：{"case_title","genre","facts":[...]}，facts 数量过
    guardrails.min_documented_facts 护栏，不够 StageBlocked（宁可不做也不编）。
  ②解说稿：按目标片长折算总字数（mystery.target_minutes × 60 × 4字/秒，
    4.0=CV2 实测吐字速率，见 d3 的 ZH_CHARS_PER_SEC 注释），产出分段
    {"segments":[{"id","zh"}]}，每段 2-4 句——段是 m3 分镜/m4 素材/m5 配音的
    共同粒度，段太长单个画面撑不住，太短素材生成量爆炸。

坑位备忘同 d3：extract_json 只解析 JSON 对象，LLM 必须回对象不能回顶层数组；
偶发残缺 JSON 重试 3 次。
"""
from __future__ import annotations

import json

from src.context import Ctx
from src.errors import StageBlocked
from src.util.jobs import read_json, write_json
from src.util.llmjson import extract_json

OUTPUT = "script.json"
INPUT = "transcript.json"

GENRES = ("mystery", "paranormal", "crime", "horror", "scifi", "history")
ZH_CHARS_PER_SEC = 4.0
DEFAULT_TARGET_MINUTES = 12          # 8-15min 档的中值（用户定成片长度）
DEFAULT_MAX_SOURCE_CHARS = 24000     # 源字幕喂给 LLM 的截断上限，防超上下文

SYSTEM_FACTS = (
    "你是纪录片资料研究员。从给定的英文视频字幕全文中提取案件的可核查事实，"
    "并判断题材类型。只输出一个 JSON 对象，不要额外文字、不要 markdown 代码块。"
)

SYSTEM_SCRIPT = (
    "你是「未解之谜」中文频道的解说撰稿人。根据给定的事实清单撰写一篇全新的、"
    "独立的中文解说稿：悬念开场、按线索推进、结尾留开放式疑问。这是原创写作，"
    "不是翻译——禁止逐句对应任何原文。语言口语化、适合配音朗读，不用书面腔。"
    "只输出一个 JSON 对象，不要额外文字、不要 markdown 代码块。"
)


def build_facts_prompt(title: str, transcript_text: str) -> str:
    return (
        "从下面的字幕全文提取案件事实，输出一个 JSON 对象（不是数组），格式必须是 "
        '{"case_title": "中文案件名", "genre": "题材", "facts": ["事实1", ...]}。\n'
        f"genre 只能从这些里选一个：{', '.join(GENRES)}。\n"
        "facts: 8-20 条，每条一句话，只写字幕里明确提到的可核查事实"
        "（时间/地点/人物/事件/官方说法/矛盾点），不要推测和演绎。\n"
        f"视频标题：{title}\n字幕全文：{transcript_text}"
    )


def build_script_prompt(case_title: str, genre: str, facts: list[str],
                        target_chars: int) -> str:
    facts_block = "\n".join(f"- {f}" for f in facts)
    return (
        "根据下面的事实清单写一篇中文解说稿，输出一个 JSON 对象（不是数组），"
        '格式必须是 {"segments": [{"id": 整数, "zh": "一段解说"}, ...]}。\n'
        f"总字数约 {target_chars} 字（允许 ±15%）。每段 2-4 句、50-120 字，"
        "段与段衔接自然；id 从 0 递增。\n"
        "每段只讲一个画面能承载的场景/线索（下游会按段生成画面）。\n"
        f"案件：{case_title}（题材：{genre}）\n事实清单：\n{facts_block}"
    )


def _llm_json(ctx: Ctx, system: str, prompt: str, tries: int = 3) -> dict | None:
    for attempt in range(tries):
        raw = ctx.llm.generate(system, prompt)
        try:
            return extract_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            ctx.warn(f"{ctx.slug}: m2 JSON 解析失败(第{attempt + 1}次): {e}")
    return None


def run(ctx: Ctx) -> None:
    transcript = read_json(ctx.job, INPUT)
    source = read_json(ctx.job, "source.json")
    max_chars = int(ctx.config.get("mystery.max_source_chars", DEFAULT_MAX_SOURCE_CHARS))
    text = " ".join(s.get("text", "") for s in transcript.get("segments", []))
    if len(text) > max_chars:
        ctx.warn(f"{ctx.slug}: m2 源字幕 {len(text)} 字符超上限，截断到 {max_chars}")
        text = text[:max_chars]

    facts_obj = _llm_json(ctx, SYSTEM_FACTS,
                          build_facts_prompt(str(source.get("title", "")), text))
    if facts_obj is None:
        raise StageBlocked(f"{ctx.slug}: m2 事实提取 LLM 连续失败，不出稿")
    facts = [str(f).strip() for f in (facts_obj.get("facts") or []) if str(f).strip()]
    min_facts = int(ctx.config.get("guardrails.min_documented_facts", 3))
    if len(facts) < min_facts:
        raise StageBlocked(
            f"{ctx.slug}: m2 可核查事实仅 {len(facts)} 条(<{min_facts})，"
            "宁可换题不硬编——删 source.json 重跑 m1 换一部")
    genre = facts_obj.get("genre")
    if genre not in GENRES:
        ctx.warn(f"{ctx.slug}: m2 未知题材 {genre!r}，回退 mystery")
        genre = "mystery"
    case_title = str(facts_obj.get("case_title") or source.get("title") or "未解之谜")

    minutes = float(ctx.config.get("mystery.target_minutes", DEFAULT_TARGET_MINUTES))
    target_chars = int(minutes * 60 * ZH_CHARS_PER_SEC)
    script_obj = _llm_json(ctx, SYSTEM_SCRIPT,
                           build_script_prompt(case_title, genre, facts, target_chars))
    raw_segments = (script_obj or {}).get("segments") or []
    segments = [{"id": i, "zh": str(s.get("zh", "")).strip()}
                for i, s in enumerate(raw_segments) if str(s.get("zh", "")).strip()]
    if not segments:
        raise StageBlocked(f"{ctx.slug}: m2 解说稿生成为空，不出稿")

    write_json(ctx.job, OUTPUT, {
        "case_title": case_title, "genre": genre,
        "facts": facts, "segments": segments,
    })
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_m2_script.py -v`
Expected: 3 PASS

- [ ] **Step 5: 提交**

```bash
git add src/stages/m2_script.py tests/test_m2_script.py
git commit -m "feat(mystery): m2_script 事实护栏+原创中文解说稿改写" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: m3_storyboard——逐段文生视频提示词

**Files:**
- Create: `src/stages/m3_storyboard.py`
- Test: `tests/test_m3_storyboard.py`

**Interfaces:**
- Consumes: `script.json`。
- Produces: `storyboard.json` `{"segments":[{"id","prompt"}]}`——prompt 已含统一画风后缀，m4 直接透传给 `remote.generate_video`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_m3_storyboard.py
from __future__ import annotations

import json

from src.config import Config
from src.context import Ctx
from src.llm import FakeLLM
from src.util.jobs import read_json, write_json

import src.stages.m3_storyboard as m3


def _ctx(tmp_path, canned):
    ctx = Ctx(slug="mys1", config=Config({}), root=tmp_path, llm=FakeLLM(canned=canned))
    write_json(ctx.job, "script.json", {
        "case_title": "罗斯威尔", "genre": "mystery", "facts": ["a", "b", "c"],
        "segments": [{"id": 0, "zh": "牧场上出现残骸。"},
                     {"id": 1, "zh": "军方连夜封锁现场。"}],
    })
    return ctx


def test_prompts_get_style_suffix(tmp_path):
    canned = json.dumps({"items": [
        {"id": 0, "prompt": "debris scattered on a desert ranch at dusk"},
        {"id": 1, "prompt": "military trucks sealing off a field at night"},
    ]})
    ctx = _ctx(tmp_path, canned)
    m3.run(ctx)
    out = read_json(ctx.job, m3.OUTPUT)
    assert len(out["segments"]) == 2
    for seg in out["segments"]:
        assert seg["prompt"].endswith(m3.STYLE_SUFFIX)
        assert "no text" in seg["prompt"]


def test_missing_id_falls_back_to_default_prompt(tmp_path):
    canned = json.dumps({"items": [{"id": 0, "prompt": "desert ranch"}]})  # 缺 id=1
    ctx = _ctx(tmp_path, canned)
    m3.run(ctx)
    out = read_json(ctx.job, m3.OUTPUT)
    assert out["segments"][1]["prompt"].startswith(m3.DEFAULT_PROMPT)
    assert any("m3" in w for w in ctx.warnings)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_m3_storyboard.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: 写 src/stages/m3_storyboard.py**

```python
"""mystery 阶段③——gemma4:31b-cloud 按解说段写英文文生视频提示词（分镜）。

用户定(2026-07-12)：m3 与 m2 用同一个 31b-cloud（不再需要 zion 跑 12b）。
每段一条英文 prompt + 统一画风后缀（暗调纪录片、电影感）——后缀锁全片视觉
一致性，LLM 只负责场景内容。负向词交给 zion_t2v.py 的 NEGATIVE_PROMPT，
"no text" 在后缀里再压一道（SD 系画字是乱码的老教训）。

批处理/JSON 对象/重试的坑位同 d3_translate（CHUNK 防长稿截断）。
"""
from __future__ import annotations

import json

from src.context import Ctx
from src.util.jobs import read_json, write_json
from src.util.llmjson import extract_json

OUTPUT = "storyboard.json"
INPUT = "script.json"
CHUNK_SIZE = 40

STYLE_SUFFIX = (", dark moody documentary cinematography, cinematic lighting, "
                "photorealistic, 35mm film grain, slow camera movement, "
                "no text, no watermark, no logo")
DEFAULT_PROMPT = "foggy abandoned place at night, mysterious atmosphere, single flashlight beam"

SYSTEM = (
    "你是纪录片视觉导演。为每段中文解说写一条英文的文生视频提示词，描述一个"
    "能承载这段内容的具体画面：场景、主体、时间/天气、镜头感。只描述画面，"
    "不要人名、不要文字元素、不要多个镜头切换。"
    "只输出一个 JSON 对象，不要额外文字、不要 markdown 代码块。"
)


def build_prompt(items: list[dict]) -> str:
    payload = json.dumps({"items": items}, ensure_ascii=False)
    return (
        "为下面每段解说各写一条英文文生视频提示词，输出一个 JSON 对象（不是数组），"
        '格式必须是 {"items": [{"id": 整数, "prompt": "english prompt"}, ...]}，'
        "id 与输入一一对应，不要遗漏、不要新增。每条 15-35 个英文单词。\n"
        f"输入：{payload}"
    )


def _batch(ctx: Ctx, items: list[dict]) -> dict[int, str]:
    parsed = None
    for attempt in range(3):
        raw = ctx.llm.generate(SYSTEM, build_prompt(items))
        try:
            parsed = extract_json(raw)
            break
        except (json.JSONDecodeError, ValueError) as e:
            ctx.warn(f"{ctx.slug}: m3 批次 JSON 解析失败(第{attempt + 1}次): {e}")
    if parsed is None:
        return {}
    out: dict[int, str] = {}
    for it in parsed.get("items", []):
        try:
            out[int(it.get("id"))] = str(it.get("prompt", "")).strip()
        except (TypeError, ValueError):
            continue
    return out


def run(ctx: Ctx) -> None:
    segments = read_json(ctx.job, INPUT).get("segments", [])
    prompts: dict[int, str] = {}
    items = [{"id": s["id"], "zh": s["zh"]} for s in segments]
    for start in range(0, len(items), CHUNK_SIZE):
        prompts.update(_batch(ctx, items[start:start + CHUNK_SIZE]))

    missing = 0
    out_segments = []
    for seg in segments:
        p = prompts.get(seg["id"], "")
        if not p:
            missing += 1
            p = DEFAULT_PROMPT
        out_segments.append({"id": seg["id"], "prompt": p.rstrip(". ") + STYLE_SUFFIX})
    if missing:
        ctx.warn(f"{ctx.slug}: m3 有 {missing}/{len(segments)} 段缺提示词，用默认画面顶上")
    write_json(ctx.job, OUTPUT, {"segments": out_segments})
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_m3_storyboard.py -v`
Expected: 2 PASS

- [ ] **Step 5: 提交**

```bash
git add src/stages/m3_storyboard.py tests/test_m3_storyboard.py
git commit -m "feat(mystery): m3_storyboard 分镜提示词(统一画风后缀)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: m4_media——逐段文生视频 + SDXL 静图兜底

**Files:**
- Create: `src/stages/m4_media.py`
- Test: `tests/test_m4_media.py`

**Interfaces:**
- Consumes: `storyboard.json`、`get_remote(config).generate_video/generate_image`（Task 4）。
- Produces: `media.json` `{"segments":[{"id","file","kind"}]}`，file 为 job 相对路径 `media/seg_NNN.mp4|.png`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_m4_media.py
from __future__ import annotations

from pathlib import Path

from src.config import Config
from src.context import Ctx
from src.llm import FakeLLM
from src.util.jobs import read_json, write_json

import src.stages.m4_media as m4


def _ctx(tmp_path, cfg=None):
    base = {"remote": {"provider": "fake"}}
    base.update(cfg or {})
    ctx = Ctx(slug="mys1", config=Config(base), root=tmp_path, llm=FakeLLM())
    write_json(ctx.job, "storyboard.json", {"segments": [
        {"id": 0, "prompt": "desert ranch"}, {"id": 1, "prompt": "military trucks"},
    ]})
    return ctx


def test_generates_video_per_segment_with_fake_remote(tmp_path):
    ctx = _ctx(tmp_path)
    m4.run(ctx)
    out = read_json(ctx.job, m4.OUTPUT)
    assert [s["kind"] for s in out["segments"]] == ["video", "video"]
    for s in out["segments"]:
        p = ctx.job / s["file"]
        assert p.exists() and p.stat().st_size > 0
    assert ctx.warnings == []


def test_falls_back_to_image_when_video_fails(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)

    class Boom:
        def generate_video(self, prompt, out_path, seconds):
            raise RuntimeError("t2v OOM")

        def generate_image(self, prompt, out_path):
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_bytes(b"png")
            return out_path

    monkeypatch.setattr(m4, "get_remote", lambda cfg: Boom())
    m4.run(ctx)
    out = read_json(ctx.job, m4.OUTPUT)
    assert all(s["kind"] == "image" and s["file"].endswith(".png")
               for s in out["segments"])
    assert len(ctx.warnings) >= 2                 # 每段一条回退告警
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_m4_media.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: 写 src/stages/m4_media.py**

```python
"""mystery 阶段④——逐段调 zion 文生视频（LTX-Video）生成画面素材。

每段一个 ~5s 片段（mystery.clip_sec），时长与配音不必对齐——m7 用慢放/来回放
把片段延展到该段配音时长（省 GPU：60-90 段一集只生成 60-90 个短片段）。
单段失败回退 SDXL 静图（m7 对静图走 Ken Burns 动效），只告警不中止——
best-effort 哲学同 d 系列；两级都失败落 fallback 占位图，绝不让一段坏素材
拖垮整集。
"""
from __future__ import annotations

from pathlib import Path

from src.context import Ctx
from src.remote import get_remote
from src.util.jobs import read_json, write_json

OUTPUT = "media.json"
INPUT = "storyboard.json"
MEDIA_DIR = "media"
DEFAULT_CLIP_SEC = 5.0


def run(ctx: Ctx) -> None:
    segments = read_json(ctx.job, INPUT).get("segments", [])
    if not segments:
        from src.errors import StageBlocked
        raise StageBlocked(f"{ctx.slug}: m4 分镜 0 段，检查 m2/m3 产物")
    clip_sec = float(ctx.config.get("mystery.clip_sec", DEFAULT_CLIP_SEC))
    remote = get_remote(ctx.config)
    media_dir = ctx.job / MEDIA_DIR
    media_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    for seg in segments:
        sid = int(seg.get("id", 0))
        prompt = str(seg.get("prompt", ""))
        mp4 = media_dir / f"seg_{sid:03d}.mp4"
        try:
            remote.generate_video(prompt, mp4, clip_sec)
            items.append({"id": sid, "file": f"{MEDIA_DIR}/{mp4.name}", "kind": "video"})
            continue
        except Exception as e:                     # noqa: BLE001 — 单段失败不中止
            ctx.warn(f"{ctx.slug}: m4 段 {sid} 文生视频失败({type(e).__name__})，"
                     "回退 SDXL 静图")
        png = media_dir / f"seg_{sid:03d}.png"
        try:
            remote.generate_image(prompt, png)
        except Exception as e:                     # noqa: BLE001 — 静图也挂→占位图
            ctx.warn(f"{ctx.slug}: m4 段 {sid} SDXL 静图也失败({type(e).__name__})，"
                     "用兜底占位图")
            fb = Path(ctx.config.get("paths.fallback_image", "assets/fallback.jpg"))
            png.write_bytes(fb.read_bytes() if fb.exists() else b"")
        items.append({"id": sid, "file": f"{MEDIA_DIR}/{png.name}", "kind": "image"})

    write_json(ctx.job, OUTPUT, {"segments": items})
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_m4_media.py -v`
Expected: 2 PASS

- [ ] **Step 5: 提交**

```bash
git add src/stages/m4_media.py tests/test_m4_media.py
git commit -m "feat(mystery): m4_media 逐段文生视频+SDXL静图兜底" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: m5_tts——固定参考克隆配音 + 顺序时间轴

**Files:**
- Create: `src/stages/m5_tts.py`
- Test: `tests/test_m5_tts.py`

**Interfaces:**
- Consumes: `script.json`、`remote.clone_synthesize(..., ref_text=...)`（Task 4）、`get_tts`、`duration_sec`；配置 `mystery.voice_ref`(默认 `assets/voice_ref/ref.wav`)、`mystery.voice_ref_text`(默认 `assets/voice_ref/ref.txt`)。
- Produces: `narration.json`——**start/end 是成片时间轴**（第 0 段从 0 起，顺序累加 `tts_dur + GAP_SEC(0.35)`），m6/m7 直接消费；中间文件 `tts_script.json`（喂克隆脚本的 `{"segments":[{"id","zh"}]}`）。常量 `GAP_SEC = 0.35`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_m5_tts.py
from __future__ import annotations

from src.config import Config
from src.context import Ctx
from src.llm import FakeLLM
from src.util.jobs import read_json, write_json

import src.stages.m5_tts as m5


def _ctx(tmp_path, cfg):
    ctx = Ctx(slug="mys1", config=Config(cfg), root=tmp_path, llm=FakeLLM())
    write_json(ctx.job, "script.json", {"case_title": "x", "genre": "mystery",
                                        "facts": ["a", "b", "c"], "segments": [
        {"id": 0, "zh": "第一段。"}, {"id": 1, "zh": "第二段。"}]})
    return ctx


def test_fake_tts_sequential_timeline(tmp_path):
    ctx = _ctx(tmp_path, {"tts": {"provider": "fake"}})
    m5.run(ctx)
    segs = read_json(ctx.job, m5.OUTPUT)["segments"]
    assert segs[0]["start"] == 0.0
    # 第二段起点 = 第一段真实时长 + 段间隙（FakeTTS: 4字×0.3s=1.2s）
    assert abs(segs[1]["start"] - (segs[0]["tts_dur"] + m5.GAP_SEC)) < 1e-6
    assert all((ctx.job / s["file"]).exists() for s in segs)


def test_clone_mode_with_fixed_ref_and_fallback(tmp_path):
    ref_dir = tmp_path / "voice_ref"
    ref_dir.mkdir()
    (ref_dir / "ref.wav").write_bytes(b"x")
    (ref_dir / "ref.txt").write_text("参考文本", encoding="utf-8")
    ctx = _ctx(tmp_path, {
        "tts": {"provider": "cosyvoice", "fallback_provider": "fake"},
        "remote": {"provider": "fake"},
        "mystery": {"voice_ref": str(ref_dir / "ref.wav"),
                    "voice_ref_text": str(ref_dir / "ref.txt")},
    })
    m5.run(ctx)
    segs = read_json(ctx.job, m5.OUTPUT)["segments"]
    assert len(segs) == 2 and all(s["voice"] == "clone" for s in segs)


def test_clone_mode_missing_ref_falls_back_whole_batch(tmp_path):
    ctx = _ctx(tmp_path, {
        "tts": {"provider": "cosyvoice", "fallback_provider": "fake"},
        "remote": {"provider": "fake"},
        "mystery": {"voice_ref": str(tmp_path / "nope.wav"),
                    "voice_ref_text": str(tmp_path / "nope.txt")},
    })
    m5.run(ctx)
    segs = read_json(ctx.job, m5.OUTPUT)["segments"]
    assert len(segs) == 2 and all(s["voice"] != "clone" for s in segs)
    assert any("参考音色缺失" in w for w in ctx.warnings)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_m5_tts.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: 写 src/stages/m5_tts.py**

```python
"""mystery 阶段⑤——CosyVoice2 固定参考克隆配音 + 顺序时间轴。

与 d4_tts 的本质区别：doc 管线的时间轴锚在源片(start 来自转录)；mystery 没有
源片，**配音时长定义时间轴**——逐段合成后按顺序累加(段间 GAP_SEC 呼吸)得出
start/end，m6 字幕/m7 画面都跟这条轴走。

克隆参考(用户拍板 2026-07-12)：全片固定用科幻FANS音色切片(assets/voice_ref/，
Task 2 产出、不入库)，不再逐段 pick_reference——remote.clone_synthesize 传
ref_text 即固定参考模式(Task 3/4)。参考文件缺失→整批回退 fallback_provider
只告警(笔记本没同步 voice_ref 时管线仍能跑通出片验流程)；单段克隆缺失→
单段回退(同 d4 的绝不让一段坏的拖垮整集)。
"""
from __future__ import annotations

from pathlib import Path

from src.context import Ctx
from src.remote import get_remote
from src.tts import TTSEngine, get_tts
from src.util.ffprobe import duration_sec
from src.util.jobs import read_json, write_json

OUTPUT = "narration.json"
INPUT = "script.json"
NARRATION_DIR = "narration"
TTS_SCRIPT = "tts_script.json"      # 喂克隆脚本的中间文件(契约同 zh_script.json 的 segments)
GAP_SEC = 0.35                       # 段间呼吸(同 d7 SENT_PAUSE_SEC 手感)


def _synth_fallback(ctx: Ctx, engine_cache: dict, zh: str, sid: int,
                    narration_dir: Path, provider: str) -> tuple[str, str]:
    voice = ctx.config.get("tts.voice", "zh-CN-YunyangNeural")
    engine: TTSEngine = engine_cache.get(voice) or get_tts(
        ctx.config, voice=voice, provider=provider)
    engine_cache[voice] = engine
    audio, _words = engine.synth(zh)
    name = f"n{sid:03d}.mp3"
    (narration_dir / name).write_bytes(audio)
    return name, voice


def run(ctx: Ctx) -> None:
    segments = read_json(ctx.job, INPUT).get("segments", [])
    narration_dir = ctx.job / NARRATION_DIR
    narration_dir.mkdir(parents=True, exist_ok=True)

    provider = ctx.config.get("tts.provider", "edge")
    fallback = ctx.config.get("tts.fallback_provider", "edge")
    clone = provider in ("clone", "cosyvoice")
    if clone:
        ref_wav = Path(ctx.config.get("mystery.voice_ref", "assets/voice_ref/ref.wav"))
        ref_text = Path(ctx.config.get("mystery.voice_ref_text",
                                       "assets/voice_ref/ref.txt"))
        if not (ref_wav.exists() and ref_text.exists()):
            ctx.warn(f"{ctx.slug}: m5 参考音色缺失({ref_wav})，整批回退 {fallback}"
                     "（voice_ref 不入库，需手动同步，见 make_voice_ref.py 文件头）")
            clone = False
        else:
            write_json(ctx.job, TTS_SCRIPT,
                       {"segments": [{"id": s["id"], "zh": s["zh"]} for s in segments]})
            get_remote(ctx.config).clone_synthesize(
                ctx.job / TTS_SCRIPT, ref_wav, narration_dir, ref_text=ref_text)

    engines: dict[str, TTSEngine] = {}
    out_segments: list[dict] = []
    cursor = 0.0
    for seg in segments:
        zh = (seg.get("zh") or "").strip()
        if not zh:
            continue
        sid = int(seg.get("id", 0))
        clone_path = narration_dir / f"n{sid:03d}.wav"
        if clone and clone_path.exists():
            name, voice = clone_path.name, "clone"
        else:
            if clone:
                ctx.warn(f"{ctx.slug}: m5 段 {sid} 克隆缺失，回退 {fallback} 补上")
            name, voice = _synth_fallback(
                ctx, engines, zh, sid, narration_dir,
                fallback if clone or provider in ("clone", "cosyvoice") else provider)
        dur = round(duration_sec(narration_dir / name), 3)
        out_segments.append({
            "id": sid, "file": f"{NARRATION_DIR}/{name}", "tts_dur": dur, "zh": zh,
            "start": round(cursor, 3), "end": round(cursor + dur, 3), "voice": voice,
        })
        cursor += dur + GAP_SEC

    if not out_segments:
        ctx.warn(f"{ctx.slug}: m5 没有可用配音段(script 为空)")
    write_json(ctx.job, OUTPUT, {"segments": out_segments})
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_m5_tts.py -v`
Expected: 3 PASS

- [ ] **Step 5: 提交**

```bash
git add src/stages/m5_tts.py tests/test_m5_tts.py
git commit -m "feat(mystery): m5_tts 固定参考克隆配音+顺序时间轴" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: m6_subs——按配音时间轴出字幕

**Files:**
- Create: `src/stages/m6_subs.py`
- Test: `tests/test_m6_subs.py`

**Interfaces:**
- Consumes: `narration.json`（m5 的成片时间轴）、复用 `d5_subs._seg_cue_lines`、`s5_subs.render_ass`。
- Produces: `subs.ass`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_m6_subs.py
from __future__ import annotations

from src.config import Config
from src.context import Ctx
from src.llm import FakeLLM
from src.util.jobs import write_json

import src.stages.m6_subs as m6


def _ctx(tmp_path):
    return Ctx(slug="mys1", config=Config({"subs": {"max_chars": 20}}),
               root=tmp_path, llm=FakeLLM())


def test_renders_ass_on_narration_timeline(tmp_path):
    ctx = _ctx(tmp_path)
    write_json(ctx.job, "narration.json", {"segments": [
        {"id": 0, "zh": "牧场上出现了不明残骸。", "start": 0.0, "end": 3.0,
         "tts_dur": 3.0, "file": "narration/n000.wav"},
        {"id": 1, "zh": "军方连夜封锁了现场。", "start": 3.35, "end": 6.0,
         "tts_dur": 2.65, "file": "narration/n001.wav"},
    ]})
    m6.run(ctx)
    txt = (ctx.job / m6.OUTPUT).read_text(encoding="utf-8")
    assert "[Events]" in txt and txt.count("Dialogue:") == 2
    assert "0:00:03.35" in txt        # 第二段 cue 从配音时间轴的 start 起


def test_empty_narration_writes_empty_ass_with_warning(tmp_path):
    ctx = _ctx(tmp_path)
    write_json(ctx.job, "narration.json", {"segments": []})
    m6.run(ctx)
    assert (ctx.job / m6.OUTPUT).exists()
    assert any("0 段" in w for w in ctx.warnings)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_m6_subs.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: 写 src/stages/m6_subs.py**

```python
"""mystery 阶段⑥——按 m5 的配音时间轴渲染中文字幕。

比 d5 简单得多：mystery 的时间轴就是配音轴本身（m5 已顺序排定 start/end），
不存在 doc 管线"源片轴 vs 配音轴"的重排对齐问题。断句/渲染复用 d5/s5 的
成熟件（句末标点必断、屏显去句号等用户定的规则都在里面），不重新发明。
"""
from __future__ import annotations

from src.context import Ctx
from src.stages.d5_subs import _seg_cue_lines
from src.stages.s5_subs import render_ass
from src.util.jobs import read_json

OUTPUT = "subs.ass"
INPUT = "narration.json"


def run(ctx: Ctx) -> None:
    max_chars = ctx.config.get("subs.max_chars", 20)
    segments = read_json(ctx.job, INPUT).get("segments", [])
    lines: list[tuple[float, float, str]] = []
    for seg in segments:
        zh = (seg.get("zh") or "").strip()
        if not zh:
            continue
        lines += _seg_cue_lines(zh, float(seg.get("start", 0.0)),
                                float(seg.get("end", 0.0)), max_chars)
    if not segments:
        ctx.warn(f"{ctx.slug}: m6_subs narration 0 段，生成空字幕")
    (ctx.job / OUTPUT).write_text(render_ass(lines), encoding="utf-8")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_m6_subs.py -v`
Expected: 2 PASS

- [ ] **Step 5: 提交**

```bash
git add src/stages/m6_subs.py tests/test_m6_subs.py
git commit -m "feat(mystery): m6_subs 配音时间轴字幕(复用d5断句)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: m7_assemble——片段延展/拼接 + BGM(按题材) + 合成

**Files:**
- Create: `src/stages/m7_assemble.py`
- Test: `tests/test_m7_assemble.py`

**Interfaces:**
- Consumes: `narration.json`、`media.json`、`subs.ass`、`script.json`(genre 选 BGM)；复用 `d7_assemble` 的 `pick_bgm_track`/`build_assemble_cmd`/`nvenc_env`/`JOB_BGM`。
- Produces: `video.mp4`（1080p30）；中间产物 `base.mp4`（无声画面轨）、`narration.wav`、`bgm.json`（供 m8 署名）。纯函数：`build_boomerang_cmd(src, out)`、`build_hold_video_cmd(src, out, dur)`、`build_kenburns_cmd(img, out, dur)`、`build_narration_track_cmd(job, segments, out)`、`seg_duration(seg, is_last)`。
- 关键约定：每段画面时长 = `tts_dur + GAP_SEC`（末段再加 `TAIL_SEC` 片尾留白），与 m5 时间轴严格一致，字幕/配音/画面三轨同源不漂移。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_m7_assemble.py
from __future__ import annotations

import subprocess

from src.config import Config
from src.context import Ctx
from src.llm import FakeLLM
from src.util.jobs import read_json, write_json

import src.stages.m7_assemble as m7


def test_seg_duration_matches_m5_timeline():
    seg = {"tts_dur": 3.0}
    assert m7.seg_duration(seg, is_last=False) == 3.0 + m7.GAP_SEC
    assert m7.seg_duration(seg, is_last=True) == 3.0 + m7.GAP_SEC + m7.TAIL_SEC


def test_boomerang_and_hold_cmds():
    bo = m7.build_boomerang_cmd("/a/seg.mp4", "/a/loop.mp4")
    assert "reverse" in " ".join(bo) and "concat=n=2:v=1:a=0" in " ".join(bo)
    hold = m7.build_hold_video_cmd("/a/loop.mp4", "/a/ext.mp4", 7.5)
    joined = " ".join(hold)
    assert "-stream_loop -1" in joined and "-t 7.5" in joined
    assert "scale=1920:1080" in joined and "fps=30" in joined and "-an" in joined


def test_kenburns_cmd_zooms_still():
    cmd = " ".join(m7.build_kenburns_cmd("/a/seg.png", "/a/ext.mp4", 4.0))
    assert "-loop 1" in cmd and "zoompan" in cmd and "s=1920x1080" in cmd


def test_narration_track_cmd_concats_with_gaps(tmp_path):
    segs = [{"file": "narration/n000.wav"}, {"file": "narration/n001.wav"}]
    cmd = m7.build_narration_track_cmd(tmp_path, segs, tmp_path / "narration.wav")
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert f"apad=pad_dur={m7.GAP_SEC}" in fc
    assert "concat=n=2:v=0:a=1" in fc and "loudnorm" in fc


def test_run_assembles_video_end_to_end(tmp_path):
    """全 fake 素材真跑 ffmpeg：2 段小片段+静音配音 → video.mp4 可解码。"""
    ctx = Ctx(slug="mys1", config=Config({"p7": {"video_watermark": False},
                                          "mystery": {}}),
              root=tmp_path, llm=FakeLLM())
    (ctx.job / "media").mkdir(parents=True)
    (ctx.job / "narration").mkdir()
    for i in range(2):
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                        "testsrc2=size=320x180:rate=24:duration=1",
                        "-pix_fmt", "yuv420p",
                        str(ctx.job / "media" / f"seg_{i:03d}.mp4")],
                       check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                        "anullsrc=r=24000:cl=mono", "-t", "1",
                        str(ctx.job / "narration" / f"n{i:03d}.wav")],
                       check=True, capture_output=True)
    write_json(ctx.job, "media.json", {"segments": [
        {"id": 0, "file": "media/seg_000.mp4", "kind": "video"},
        {"id": 1, "file": "media/seg_001.mp4", "kind": "video"}]})
    write_json(ctx.job, "narration.json", {"segments": [
        {"id": 0, "zh": "一。", "file": "narration/n000.wav", "tts_dur": 1.0,
         "start": 0.0, "end": 1.0},
        {"id": 1, "zh": "二。", "file": "narration/n001.wav", "tts_dur": 1.0,
         "start": 1.35, "end": 2.35}]})
    write_json(ctx.job, "script.json", {"genre": "mystery", "case_title": "x",
                                        "facts": [], "segments": []})
    (ctx.job / "subs.ass").write_text(
        "[Script Info]\n\n[Events]\nFormat: Layer, Start, End, Style, Name, "
        "MarginL, MarginR, MarginV, Effect, Text\n", encoding="utf-8")

    m7.run(ctx)

    out = ctx.job / m7.OUTPUT
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(out)], capture_output=True, text=True)
    assert probe.returncode == 0
    assert float(probe.stdout.strip()) >= 2.0     # ≈两段时长+留白
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_m7_assemble.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: 写 src/stages/m7_assemble.py**

```python
"""mystery 阶段⑦——把逐段生成的画面素材延展到配音时长、拼接成画面轨，再走
d7 的成熟合成链（BGM 床+侧链 duck+烧字幕+台标水印+nvenc 回退）出成片。

画面延展：t2v 片段固定 ~5s，配音段普遍 8-20s——直接慢放会拖成鬼片，改用
"来回放"(boomerang：正放+倒放拼成回文再无缝循环)延展任意时长；静图兜底段
走 zoompan Ken Burns 缓推。每段延展目标=该段配音时长+GAP_SEC（与 m5 时间轴
严格同源，字幕/配音/画面三轨不漂移），末段加 TAIL_SEC 片尾留白。

统一转码 1920x1080@30fps yuv420p 后用 concat demuxer `-c copy` 拼接（参数
完全一致才能 copy，省一次全片重编码；最终编码在 build_assemble_cmd 的第②遍）。

BGM 按题材(用户定 2026-07-12)：m2 的 genre 直接当 pick_bgm_track 的 mood 桶
（bgm.json 清单条目 moods 里标 mystery/horror/... 即入桶），命不中自动退全池，
清单缺失退静音床——绝不因配乐中止 run。选中曲目写 job/bgm.json 供 m8 署名，
与 d7/d8 同一契约。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from src.context import Ctx
from src.errors import StageBlocked
from src.stages.d7_assemble import (
    DEFAULT_BGM_DIR,
    JOB_BGM,
    build_assemble_cmd,
    nvenc_env,
    pick_bgm_track,
)
from src.util.ffprobe import duration_sec
from src.util.jobs import read_json, write_json

OUTPUT = "video.mp4"
NARRATION = "narration.json"
MEDIA = "media.json"
SUBS = "subs.ass"
BASE = "base.mp4"                    # 中间产物：拼好的无声画面轨
NARRATION_TRACK = "narration.wav"
GAP_SEC = 0.35                       # 必须与 m5_tts.GAP_SEC 一致（时间轴同源）
TAIL_SEC = 1.5
W, H, FPS = 1920, 1080, 30


def seg_duration(seg: dict, is_last: bool) -> float:
    d = float(seg.get("tts_dur", 0.0) or 0.0) + GAP_SEC
    return round(d + (TAIL_SEC if is_last else 0.0), 3)


def build_boomerang_cmd(src: str | Path, out: str | Path) -> list[str]:
    return ["ffmpeg", "-y", "-i", str(src), "-filter_complex",
            "[0:v]split[f][b];[b]reverse[r];[f][r]concat=n=2:v=1:a=0[v]",
            "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]


def _scale_pad() -> str:
    return (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={FPS}")


def build_hold_video_cmd(src: str | Path, out: str | Path, dur: float) -> list[str]:
    return ["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(src), "-t", str(dur),
            "-vf", _scale_pad(), "-an",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]


def build_kenburns_cmd(img: str | Path, out: str | Path, dur: float) -> list[str]:
    frames = max(int(dur * FPS), 1)
    # 先放大再 zoompan 防抖(zoompan 低分辨率缩放会锯齿抖动，ffmpeg 社区惯用招)
    vf = (f"scale=3840:-2,zoompan=z='min(zoom+0.0008,1.25)':d={frames}"
          f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={FPS}")
    return ["ffmpeg", "-y", "-loop", "1", "-i", str(img), "-t", str(dur),
            "-vf", vf, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]


def build_narration_track_cmd(job: Path, segments: list[dict], out: Path) -> list[str]:
    """配音轨=逐段 wav 顺序拼接(每段尾垫 GAP_SEC)+响度归一——时间轴与 m5 的
    start/end 天然一致，不需要 d7 那套 adelay/amix 重排。"""
    cmd: list[str] = ["ffmpeg", "-y"]
    for seg in segments:
        cmd += ["-i", str(job / seg["file"])]
    n = len(segments)
    parts = [f"[{i}:a]aresample=48000,apad=pad_dur={GAP_SEC}[p{i}]" for i in range(n)]
    parts.append("".join(f"[p{i}]" for i in range(n))
                 + f"concat=n={n}:v=0:a=1,loudnorm=I=-16:TP=-1.5:LRA=11[out]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[out]", str(out)]
    return cmd


def _run_ff(cmd: list[str], what: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg {what} 失败 (exit {proc.returncode}):\n"
                           f"{(proc.stderr or '')[-2000:]}")


def run(ctx: Ctx) -> None:
    job = ctx.job
    segments = read_json(job, NARRATION).get("segments", [])
    if not segments:
        raise StageBlocked(f"{ctx.slug}: m7 无配音段——mystery 成片时间轴由配音定义，"
                           "查 m2/m5 产物后重跑")
    media = {int(m["id"]): m for m in read_json(job, MEDIA).get("segments", [])}

    # ① 逐段延展画面到配音时长
    ext_dir = job / "media"
    concat_list = ext_dir / "concat.txt"
    lines: list[str] = []
    for i, seg in enumerate(segments):
        sid = int(seg["id"])
        dur = seg_duration(seg, is_last=(i == len(segments) - 1))
        ext = ext_dir / f"ext_{sid:03d}.mp4"
        m = media.get(sid)
        if m and m["kind"] == "video" and (job / m["file"]).exists():
            loop = ext_dir / f"loop_{sid:03d}.mp4"
            _run_ff(build_boomerang_cmd(job / m["file"], loop), f"段{sid} 回文")
            _run_ff(build_hold_video_cmd(loop, ext, dur), f"段{sid} 延展")
            loop.unlink(missing_ok=True)
        else:
            if not m:
                ctx.warn(f"{ctx.slug}: m7 段 {sid} 无素材记录，用兜底占位图")
            img = (job / m["file"]) if m else Path(
                ctx.config.get("paths.fallback_image", "assets/fallback.jpg"))
            _run_ff(build_kenburns_cmd(img, ext, dur), f"段{sid} KenBurns")
        lines.append(f"file '{ext.name}'")
    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ② 拼画面轨（参数统一，-c copy 免二次全片编码）
    base = job / BASE
    _run_ff(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
             "-c", "copy", str(base)], "画面轨拼接")
    video_dur = round(duration_sec(base), 3)

    # ③ 配音轨
    narration_path = job / NARRATION_TRACK
    _run_ff(build_narration_track_cmd(job, segments, narration_path), "配音轨")

    # ④ BGM 按题材选曲；缺清单/缺曲退静音床(不 duck 也不崩)
    genre = (read_json(job, "script.json").get("genre")
             if (job / "script.json").exists() else None)
    bgm_dir = Path(ctx.config.get("p7.bgm_dir", DEFAULT_BGM_DIR))
    picked = pick_bgm_track(bgm_dir, ctx.slug,
                            lambda m: ctx.warn(f"{ctx.slug}: {m}"), mood=genre)
    if picked is not None:
        bg = picked["path"]
        write_json(job, JOB_BGM, picked["entry"])
    else:
        bg = job / "_silence.wav"
        _run_ff(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                 "-t", "1", str(bg)], "静音床")
    bg_input_opts = ["-stream_loop", "-1", "-t", f"{video_dur}"]

    # ⑤ 终混：复用 d7 的第②遍(床+duck+烧字幕+水印+nvenc 回退)
    encoder = ctx.config.get("p7.encoder", "libx264")

    def wm_warn(msg: str) -> None:
        ctx.warn(f"{ctx.slug}: {msg}")

    def _cmd(enc: str) -> list[str]:
        return build_assemble_cmd(
            base, bg, narration_path, job / SUBS, job / OUTPUT,
            ctx.config.get("p7.bg_gain", 0.5), bg_input_opts, encoder=enc,
            video_dur=video_dur, slug=ctx.slug, config=ctx.config, warn=wm_warn)

    env = nvenc_env(ctx.config) if encoder == "nvenc" else None
    proc = subprocess.run(_cmd(encoder), capture_output=True, text=True, env=env)
    if proc.returncode != 0 and encoder == "nvenc":
        ctx.warn(f"{ctx.slug}: m7 nvenc 失败(exit {proc.returncode})，回退 libx264: "
                 f"{(proc.stderr or '')[-500:]}")
        proc = subprocess.run(_cmd("libx264"), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 合成失败 (exit {proc.returncode}):\n"
                           f"{(proc.stderr or '')[-2000:]}")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_m7_assemble.py -v`
Expected: 5 PASS（最后一个是真 ffmpeg 集成测试，慢一点属正常）

- [ ] **Step 5: 提交**

```bash
git add src/stages/m7_assemble.py tests/test_m7_assemble.py
git commit -m "feat(mystery): m7_assemble 素材延展拼接+题材BGM+复用d7终混" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: m8_meta——缩略图 + 元数据（含 AI 生成声明）

**Files:**
- Create: `src/stages/m8_meta.py`
- Test: `tests/test_m8_meta.py`

**Interfaces:**
- Consumes: `script.json`、`bgm.json`(可缺)、`remote.generate_image`、复用 `d8_meta.make_thumbnail`/`_bgm_attribution_line`。
- Produces: `meta.json` `{"title","description","tags"}` + `thumb.jpg`——契约与 s9_publish 兼容（下一期发布直接可用）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_m8_meta.py
from __future__ import annotations

import json

from src.config import Config
from src.context import Ctx
from src.llm import FakeLLM
from src.util.jobs import read_json, write_json

import src.stages.m8_meta as m8

CANNED = json.dumps({"title": "罗斯威尔之谜", "description": "官方说法三度反转。",
                     "tags": ["未解之谜", "UFO"],
                     "image_prompt": "crashed disc in desert night"},
                    ensure_ascii=False)


def _ctx(tmp_path):
    ctx = Ctx(slug="mys1", config=Config({"remote": {"provider": "fake"}}),
              root=tmp_path, llm=FakeLLM(canned=CANNED))
    write_json(ctx.job, "script.json", {
        "case_title": "罗斯威尔", "genre": "mystery", "facts": ["a", "b", "c"],
        "segments": [{"id": 0, "zh": "第一段。"}]})
    return ctx


def test_writes_meta_thumb_and_ai_disclosure(tmp_path):
    ctx = _ctx(tmp_path)
    m8.run(ctx)
    meta = read_json(ctx.job, m8.OUTPUT)
    assert meta["title"] == "罗斯威尔之谜"
    assert m8.AI_DISCLOSURE in meta["description"]      # AI 生成声明必须在
    assert (ctx.job / "thumb.jpg").stat().st_size > 0


def test_appends_bgm_attribution_when_present(tmp_path):
    ctx = _ctx(tmp_path)
    write_json(ctx.job, "bgm.json", {"file": "x.mp3",
                                     "attribution": "Music: Kevin MacLeod (CC-BY)"})
    m8.run(ctx)
    assert "Kevin MacLeod" in read_json(ctx.job, m8.OUTPUT)["description"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_m8_meta.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: 写 src/stages/m8_meta.py**

```python
"""mystery 阶段⑧——SDXL 缩略图 + LLM 标题/简介/标签。

与 d8 的差异：没有源片(不写"素材来源"署名、SDXL 挂了没有帧可抽只回退纯色底)；
简介固定追加 AI_DISCLOSURE——画面全是 AI 生成演绎、文稿是基于公开报道的原创
整理，主动声明既是平台政策要求(YouTube 合成内容披露)也是对观众诚实。
缩略图合成/BGM 署名直接复用 d8 的 make_thumbnail/_bgm_attribution_line。
"""
from __future__ import annotations

from pathlib import Path

from src.context import Ctx
from src.errors import StageBlocked
from src.remote import get_remote
from src.stages.d8_meta import DEFAULT_WATERMARK, _bgm_attribution_line, make_thumbnail
from src.stages.s8_meta import TITLE_FONT
from src.util.jobs import read_json, write_json
from src.util.llmjson import extract_json

OUTPUT = "meta.json"
INPUT = "script.json"

AI_DISCLOSURE = "本片解说由 AI 基于公开报道整理撰写，画面为 AI 生成的演绎场景。"
DEFAULT_IMAGE_PROMPT = ("mysterious foggy night scene, single light source, dark "
                        "documentary key art, cinematic, ultra detailed, no text")

SYSTEM = (
    "你是「未解之谜」中文纪录片频道的 YouTube 运营。为给定的解说稿写吸引点击的"
    "元数据。只输出一个 JSON 对象，不要额外文字、不要 markdown 代码块。"
)


def build_prompt(script: dict) -> str:
    body = "".join(seg.get("zh", "") for seg in script.get("segments", []))[:1200]
    return (
        "为下面这集未解之谜解说写 YouTube 元数据，输出一个 JSON 对象（不是数组），"
        '格式必须是 {"title": "...", "description": "...", "tags": ["...", ...], '
        '"image_prompt": "..."}。\n'
        "title: 中文标题，悬念感强，不超过约20字，不标题党造假。\n"
        "description: 中文简介，2-4句，抛出核心谜团但不剧透结论。\n"
        "tags: 5-12个中文/英文标签字符串数组。\n"
        "image_prompt: 一句英文 Stable Diffusion 提示词，描述本案最标志性的"
        "画面，要求 dark, mysterious, cinematic, no text, no watermark。\n"
        f"案件：{script.get('case_title', '')}（题材：{script.get('genre', '')}）\n"
        f"解说稿节选：{body}"
    )


def run(ctx: Ctx) -> None:
    script = read_json(ctx.job, INPUT)
    if not script.get("segments"):
        raise StageBlocked(f"{ctx.slug}: m8 解说稿 0 段，不出片")

    meta = extract_json(ctx.llm.generate(SYSTEM, build_prompt(script)))
    desc = (meta.get("description") or "") + "\n\n" + AI_DISCLOSURE
    bgm_attrib = _bgm_attribution_line(ctx.job)
    if bgm_attrib:
        desc += "\n\n" + bgm_attrib

    image_prompt = meta.get("image_prompt") or DEFAULT_IMAGE_PROMPT
    bg_path = ctx.job / "thumb_bg.png"
    try:
        get_remote(ctx.config).generate_image(image_prompt, bg_path)
    except Exception as e:                # noqa: BLE001 — 缩略图挂了片子照出
        ctx.warn(f"{ctx.slug}: m8 SDXL 封面失败({type(e).__name__})，回退纯色底")
        bg_path = ctx.job / "thumb_bg_missing.png"   # 不存在→make_thumbnail 纯色兜底

    title = meta.get("title") or script.get("case_title") or "未解之谜"
    make_thumbnail(ctx.job / "thumb.jpg", title, bg_path,
                   Path(ctx.config.get("p7.watermark", DEFAULT_WATERMARK)),
                   font_path=ctx.config.get("paths.title_font", TITLE_FONT))

    write_json(ctx.job, OUTPUT, {
        "title": title, "description": desc, "tags": meta.get("tags", []),
    })
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_m8_meta.py -v`
Expected: 2 PASS

- [ ] **Step 5: 提交**

```bash
git add src/stages/m8_meta.py tests/test_m8_meta.py
git commit -m "feat(mystery): m8_meta 缩略图+元数据(含AI生成声明)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 13: run.py 注册 mystery 管线 + 两份 config

**Files:**
- Modify: `src/run.py`
- Create: `config-mystery.yaml`（笔记本档）、`config-zion-mystery.yaml`（zion 生产档）
- Test: `tests/test_run_mystery.py`

**Interfaces:**
- Produces: `STAGES_MYSTERY`、`--pipeline mystery`；两份可直接跑的 config。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_run_mystery.py
from __future__ import annotations

import src.run as run_mod


def test_stages_mystery_order_and_lookup():
    names = [s[0] for s in run_mod.STAGES_MYSTERY]
    assert names == ["m1_scout", "m2_script", "m3_storyboard", "m4_media",
                     "m5_tts", "m6_subs", "m7_assemble", "m8_meta"]
    assert run_mod._stages_for("mystery") is run_mod.STAGES_MYSTERY


def test_existing_pipelines_untouched():
    assert run_mod._stages_for("original") is run_mod.STAGES
    assert run_mod._stages_for("doc") is run_mod.STAGES_P7
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_run_mystery.py -v`
Expected: FAIL `AttributeError: ... no attribute 'STAGES_MYSTERY'`

- [ ] **Step 3: 改 src/run.py**

import 区追加 `m1_scout, m2_script, m3_storyboard, m4_media, m5_tts, m6_subs, m7_assemble, m8_meta`（并入现有 `from src.stages import (...)`）。`STAGES_P7` 定义之后加：

```python
# mystery「未解之谜」管线(2026-07-12 用户定,设计见 docs/superpowers/specs/
# 2026-07-12-mystery-channel-design.md)：YouTube 热门检索只下字幕 → 事实护栏+
# 原创中文改写 → 分镜 → zion 文生视频(失败退 SDXL 静图) → CosyVoice2 固定参考
# 克隆配音(时间轴由配音定义) → 字幕 → 延展拼接+题材 BGM 终混 → 缩略图/元数据。
# 本期到成片为止(用户人工审看)，发布链下一期接 s9_publish 同款契约(meta.json+
# video.mp4+thumb.jpg 已对齐)。
STAGES_MYSTERY: list[tuple[str, Callable[[Ctx], None], str]] = [
    ("m1_scout", m1_scout.run, m1_scout.OUTPUT),
    ("m2_script", m2_script.run, m2_script.OUTPUT),
    ("m3_storyboard", m3_storyboard.run, m3_storyboard.OUTPUT),
    ("m4_media", m4_media.run, m4_media.OUTPUT),
    ("m5_tts", m5_tts.run, m5_tts.OUTPUT),
    ("m6_subs", m6_subs.run, m6_subs.OUTPUT),
    ("m7_assemble", m7_assemble.run, m7_assemble.OUTPUT),
    ("m8_meta", m8_meta.run, m8_meta.OUTPUT),
]
```

`_stages_for` 加分支（`raise` 前）：

```python
    if pipeline == "mystery":
        return STAGES_MYSTERY
    raise ValueError(f"未知 pipeline: {pipeline!r}（可选 original|doc|mystery）")
```

argparse 的 `--pipeline` 改 `choices=["original", "doc", "mystery"]`，help 文案追加 `| mystery(未解之谜 m1-m8)`。

- [ ] **Step 4: 写 config-mystery.yaml（笔记本档，GPU 走 ssh 到 zion）**

```yaml
pipeline: mystery            # 未解之谜频道(m1-m8,2026-07-12)；到成片为止,本期不发布
llm:
  provider: ollama           # m2改写/m3分镜/m8元数据全走 gemma4:31b-cloud(用户定:m3与m2同模型)
  ollama_host: http://eros:11434
  ollama_model: gemma4:31b-cloud
tts:
  provider: cosyvoice        # 固定参考克隆(科幻FANS音色,用户拍板);引擎 CosyVoice2(不要F5)
  fallback_provider: edge    # 单段克隆失败/voice_ref 未同步时的回退
  voice: zh-CN-YunyangNeural
mystery:
  query: "unsolved mystery documentary"
  rows: 50
  min_sec: 480               # 8min:太短撑不起改写稿的事实量
  max_sec: 3600
  min_segments: 30           # 字幕句数下限(事实密度门)
  target_minutes: 12         # 成片 8-15min 档中值
  clip_sec: 5                # 每段 t2v 片段秒数(m7 延展到配音时长)
  voice_ref: assets/voice_ref/ref.wav        # make_voice_ref.py 产出;不入git,手动同步
  voice_ref_text: assets/voice_ref/ref.txt
subs:
  max_chars: 20
remote:
  provider: ssh              # 笔记本没 GPU:t2v/SDXL/克隆全 ssh 到 zion
  host: fabric@zion
  workdir: ~/p7-work
  timeout_sec: 1200
  t2v_wrapper: ~/t2v-py      # Task 1 Spike A 建的 LTX-Video venv wrapper
  t2v_timeout_sec: 900       # 单段 t2v 上限;超时段自动退 SDXL 静图
  cosy_wrapper: ~/cosy-py
  clone_timeout_sec: 7200
p7:                          # m7/m8 复用 d7/d8 的键(水印/BGM/编码),沿用 p7 命名空间
  bg_gain: 0.5
  bgm_dir: assets/bgm
  watermark: assets/watermark_tr.png   # TODO 下一期换新频道台标;当前先复用现有水印
  video_watermark: true
  encoder: libx264
guardrails:
  min_documented_facts: 3
paths:
  jobs_root: jobs
  fallback_image: assets/fallback.jpg
  title_font: /usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc
```

- [ ] **Step 5: 写 config-zion-mystery.yaml（zion 生产档）**

与 config-mystery.yaml 逐项相同，仅差三处（文件头注释注明"与 config-mystery.yaml 唯三差异"）：`remote.provider: local`、`remote.wrapper: ~/gpu-py` 保持默认并加 `remote.t2v_wrapper: ~/t2v-py`（同值，local 模式本机 subprocess）、`p7.encoder: nvenc`、`paths.title_font: /run/current-system/sw/share/X11/fonts/NotoSansCJK-Bold.ttc`（照抄 config-zion.yaml 里 zion 的实际字体路径，写前先 `grep title_font config-zion.yaml` 核对）。

- [ ] **Step 6: 跑测试确认通过（含全量回归）**

Run: `pytest tests/ -v --timeout=300 -x -q 2>&1 | tail -20`（无 timeout 插件就去掉该参数）
Expected: 全 PASS（既有 original/doc 测试零回归）

- [ ] **Step 7: 提交**

```bash
git add src/run.py config-mystery.yaml config-zion-mystery.yaml tests/test_run_mystery.py
git commit -m "feat(mystery): 注册 STAGES_MYSTERY 管线+双机config" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 14: BGM 曲库打题材标签

**Files:**
- Modify: `assets/bgm/bgm.json`

- [ ] **Step 1: 看现有清单**

Run: `cat assets/bgm/bgm.json`

- [ ] **Step 2: 给适合悬疑氛围的曲目追加 moods 标签**

m7 用 m2 的 genre（`mystery|paranormal|crime|horror|scifi|history`）当 mood 桶查 `pick_bgm_track`。逐条听现有曲目（`ffplay assets/bgm/<file>`），把暗调/氛围/低沉的曲目的 `moods` 数组追加相应题材词，例如：

```json
{"file": "dark_ambient_1.mp3", "attribution": "Music: ... (CC-BY)",
 "moods": ["somber", "mystery", "paranormal", "crime", "horror"]}
```

没有一首合适就从 incompetech.com（Kevin MacLeod，CC-BY）下载 2-3 首 dark ambient/suspense 曲目入池（署名写进 attribution）。命不中桶会自动退全池（`pick_bgm_track` 现有行为），所以这步不阻塞其它任务，但样片前必须完成，否则悬疑片配上太空昂扬曲会穿帮。

- [ ] **Step 3: 提交**

```bash
git add assets/bgm/bgm.json assets/bgm/
git commit -m "chore(mystery): bgm 清单加题材 moods 标签(+新增悬疑曲目)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 15: 全链离线 smoke 测试

**Files:**
- Test: `tests/test_smoke_mystery.py`

**Interfaces:**
- Consumes: 全部 m 系 stage + FakeLLM/FakeTTS/FakeRemoteGPU；m1 的网络函数打桩。

- [ ] **Step 1: 写失败(第一次跑必真跑通全链)测试**

```python
# tests/test_smoke_mystery.py
"""mystery 管线全链离线 smoke：m1 打桩、其余全 fake provider，真跑 ffmpeg。

FakeLLM 对所有调用回同一份 canned——把 m2 两次调用(facts/segments)、m3(items)、
m8(title/...) 需要的键全塞进一个 JSON 对象，各阶段只取自己认识的键。
"""
from __future__ import annotations

import json

from src.config import Config
from src.context import Ctx
from src.llm import FakeLLM
from src.run import run_pipeline

import src.stages.m1_scout as m1

CANNED = json.dumps({
    "case_title": "测试之谜", "genre": "mystery",
    "facts": ["事实一", "事实二", "事实三"],
    "segments": [{"id": 0, "zh": "第一段解说。"}, {"id": 1, "zh": "第二段解说。"}],
    "items": [{"id": 0, "prompt": "foggy ranch"}, {"id": 1, "prompt": "night road"}],
    "title": "测试之谜", "description": "简介。", "tags": ["t"],
    "image_prompt": "dark scene",
}, ensure_ascii=False)

CFG = {
    "pipeline": "mystery",
    "llm": {"provider": "fake"},
    "tts": {"provider": "fake"},
    "remote": {"provider": "fake"},
    "mystery": {"min_segments": 2, "clip_sec": 1},
    "p7": {"video_watermark": False},
    "guardrails": {"min_documented_facts": 3},
}


def _fake_subs(job):
    events = [{"tStartMs": i * 2000, "dDurationMs": 1800,
               "segs": [{"utf8": f"Fact number {i}."}]} for i in range(5)]
    (job / "source_subs.json3").write_text(json.dumps({"events": events}),
                                           encoding="utf-8")
    return True


def test_full_mystery_pipeline_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(m1, "_yt_search", lambda *a: [
        {"id": "vid1", "title": "Great Mystery", "duration": 900,
         "view_count": 1000, "channel": "Docs"}])
    monkeypatch.setattr(m1, "_yt_download_subs",
                        lambda ytdlp, vid, job: _fake_subs(job))

    config = Config(CFG)
    run_pipeline("smoke-mys", config, tmp_path, FakeLLM(canned=CANNED),
                 pipeline="mystery")

    job = tmp_path / "smoke-mys"
    for artifact in ("source.json", "script.json", "storyboard.json", "media.json",
                     "narration.json", "subs.ass", "video.mp4", "thumb.jpg",
                     "meta.json"):
        assert (job / artifact).exists(), f"缺产物 {artifact}"
    assert (job / "video.mp4").stat().st_size > 10_000
```

- [ ] **Step 2: 跑通**

Run: `pytest tests/test_smoke_mystery.py -v`
Expected: PASS（首跑若失败，按报错修对应 stage——这是集成层的最后一道网）

- [ ] **Step 3: 全量回归 + 提交**

Run: `pytest tests/ -q 2>&1 | tail -5`
Expected: 全 PASS

```bash
git add tests/test_smoke_mystery.py
git commit -m "test(mystery): 全链离线 smoke" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 16: 真跑第一集样片（人工审看门）

**Files:** 无代码改动；产出 `jobs/<slug>/video.mp4`。

- [ ] **Step 1: 预检**

- Task 1 Spike A 通过（zion 能出 t2v 片段）、Task 3 Spike B 通过（克隆音色用户认可）。
- `assets/voice_ref/` 已同步到 zion（Task 2 Step 6）。
- 代码同步到 zion：`git push zion master && ssh fabric@zion 'cd ~/youtuber && git pull'`（zion remote 用法见 docs/zion-deploy.md）。

- [ ] **Step 2: 在 zion 上跑（生产档，GPU 本机直跑最快）**

```bash
ssh fabric@zion
cd ~/youtuber
~/pipe-venv/bin/python -m src.run --case mystery-001 --config config-zion-mystery.yaml
```

Expected: 各阶段 `[run ]`/`[done]` 顺序推进；m4 是大头（60-90 段 × 单段分钟级，参考 spike 外推值，可能要跑几小时——建议 `tmux` 里跑）。中途单段失败只出 `[warn]` 回退静图，不中断。

- [ ] **Step 3: 拉回样片给用户审**

```bash
scp fabric@zion:~/youtuber/jobs/mystery-001/{video.mp4,thumb.jpg,meta.json} /tmp/mystery-001/
```

请用户看：画面质量/风格统一度、克隆音色、BGM 贴不贴题材、字幕节奏、缩略图。**用户说效果 OK 才算本期完成**；不 OK 按反馈定位阶段（画面→m3 提示词/m4 模型参数；声音→Task 2 换切片；节奏→m2 target_minutes/段长），改完 `--force --from <stage>` 重跑。

---

## Self-Review 结果

- **Spec 覆盖**：m1–m8、双 spike、voice_ref 预处理、BGM 题材标签、双机 config、fake 全链测试、真跑审看——spec 各节均有对应任务。发布链/新频道 OAuth 为 spec 明示的下一期，不在本计划。
- **类型/命名一致性**：`GAP_SEC=0.35` 在 m5/m7 各自定义且测试断言相等语义（m7 注释标明必须一致）；`narration.json` 的 start/end 契约在 Task 9/10/11 三处一致；`clone_synthesize(..., ref_text=None)` 签名在 Protocol/Fake/SSH/Local 四处一致；`OUTPUT` 常量与 STAGES_MYSTERY 注册一一对应。
- **占位符扫描**：Task 3 Step 3(c) 要求实现者先读 zion_cosy_tts.py 再改 main()（该文件的 main 未在计划中逐行给出，属于"改现有文件先读文件"的常规要求，非 TBD）；config-zion-mystery.yaml 的字体路径要求先 grep 核对——均给出了确切的核对命令。
