# 探索飞碟与外星人 · 自动化视频管线设计 (v1)

- **日期**: 2026-06-30
- **状态**: 设计已确认，待写实现计划
- **作者**: fabric-soul7 + Claude

---

## 1. 概述 (Overview)

一个面向 YouTube 的**无人出镜中文频道**「**探索飞碟与外星人**」，自动化生产 8–20 分钟的
**原创悬疑解说视频**，讲述真实的 UFO / 外星人案例（罗斯威尔、凤凰城之光、Nimitz Tic-Tac…）。

核心管线：`选题 → 调研核查 → 写脚本 → 配音 → 字幕 → 取合法素材 → 合成 → 元数据/封面 → 发布`。

设计目标是**全自动**，但在三个关键点焊死护栏（事实核查、素材合法性、发布暂停开关），
并采用**「先全免费跑通 MVP，再按瓶颈逐环节升级付费」**的成本策略。

---

## 2. 目标与非目标 (Goals / Non-Goals)

### Goals
- 一条 CLI 命令把一个案例从「选题」跑到「成片」（发布可选自动）。
- 每个环节**可断点续跑、可单独调试、可独立替换**（免费实现 ↔ 付费实现，仅改 config）。
- 内容**合法、可过 YouTube 合作计划 (YPP) 变现、不踩版权与 spam 政策**。
- 免费档零 API 成本即可出片；升级路径清晰。

### Non-Goals (v1 明确不做)
- ❌ 下载/搬运/翻译他人受版权保护的视频 —— 这是频道封号的头号风险，整套设计就是为绕开它。
- ❌ 本地 AI 出图（机器无独立显卡，CPU 出图不可行）。
- ❌ 多平台分发（Bilibili/抖音/TikTok）—— 未来再说。
- ❌ 真人配音、真人剪辑。

---

## 3. 关键约束：版权 (Critical Constraint — 整套架构的根因)

> 这一节解释「为什么管线长这样」。任何实现都不得违反本节。

直接下载英文纪录片配中文发布 = 侵权 + 必被 Content ID 识别 + 被判 **Reused Content** 无法变现 +
三次版权警告封号。因此本项目走**原创解说**：
- **脚本**：我们自己写中文原创文稿（演绎/解说属原创表达）。
- **画面**：**只允许**来自合法白名单源（见 §6 阶段⑥ 与 §7 护栏）。
- **配音/字幕**：基于我们自己的文稿生成，无第三方音轨。

这样产出的是**原创演绎作品**，而非搬运。

---

## 4. 内容形态规格 (Format Spec)

| 维度 | 规格 |
|---|---|
| 格式 | 悬疑故事解说（单案例叙事） |
| 时长 | 8–20 分钟（约 1500–3500 中文字脚本） |
| 配音声线 | 沉稳、好奇、带悬念的「探索腔」；候选 `zh-CN-YunyangNeural`(云扬,叙事) / `zh-CN-YunxiNeural`(云希)，语速略放慢 |
| 语言 | 简体中文配音 + 中文字幕（可选英文字幕做扩量） |
| 画面 | 公版素材 + Ken Burns 推拉镜头 + 字幕 + 氛围 BGM + 片头/片尾 |
| 受众 | YouTube 上的中文 UFO/未解之谜 兴趣观众 |

---

## 5. 架构 (Architecture)

**模块化 CLI 管线**。每个阶段是一个独立、可单测的 Python 模块，编排器按顺序调用，
**通过 job 文件夹里的文件交接**（不共享内存状态），因此天然可续跑、可调试、可替换。

```
youtuber/
├── config.yaml                # 全局配置：成本档、声线、发布模式、护栏严格度、provider 选择
├── seeds/cases.yaml           # 选题种子库（well-documented 案例清单）
├── src/
│   ├── run.py                 # 编排器：串联 9 个阶段，支持 --from/--to/--resume
│   ├── config.py              # 读 config.yaml + .env，提供统一配置对象
│   ├── llm.py                 # LLM provider 抽象 (gemini|claude|openai)，generate(system,prompt)
│   ├── stages/
│   │   ├── s1_topic.py        # 选题
│   │   ├── s2_research.py     # 调研 + 事实核查 🛡️
│   │   ├── s3_script.py       # 写脚本
│   │   ├── s4_tts.py          # 配音 (edge-tts) + 字级时间戳
│   │   ├── s5_subs.py         # 字幕生成 (无需 Whisper)
│   │   ├── s6_media.py        # 取合法素材 🛡️
│   │   ├── s7_assemble.py     # ffmpeg 合成
│   │   ├── s8_meta.py         # 标题/简介/标签/封面
│   │   └── s9_publish.py      # YouTube 上传 🛡️
│   ├── sources/               # 合法素材源适配器 (nasa, archive_org, wikimedia)
│   └── util/                  # ffmpeg 封装、kenburns、srt/ass、http 缓存、日志
├── jobs/<slug>/               # 每集一个文件夹，存全部中间产物（git 忽略）
├── assets/                    # 片头/片尾/水印/字体/默认BGM（可复用）
├── secrets/                   # OAuth token、API key（git 忽略）
└── tests/
```

**编排器契约**：`python -m src.run --case roswell-1947 [--from s3 --to s7] [--resume] [--publish private|public|skip]`
每阶段读上一阶段产物、写自己的产物到 `jobs/<slug>/`，幂等（已存在产物默认跳过，除非 `--force`）。

---

## 6. 九阶段 I/O 契约 (Stage Contracts)

> 每个阶段定义：**输入文件 → 做什么 → 输出文件**。这是实现的「接口」，让各单元可独立开发与测试。

### ① s1_topic — 选题
- **In**: `seeds/cases.yaml` 或 `--case <slug>`
- **Do**: 选定一个案例；LLM 补全标准化元信息（名称、时间、地点、一句话钩子、已知争议点）。
- **Out**: `jobs/<slug>/topic.json`
```json
{ "slug": "roswell-1947", "title_working": "罗斯威尔事件",
  "year": 1947, "location": "新墨西哥州", "hook": "军方先承认是飞碟，几小时后改口气象气球",
  "known_controversies": ["军方两次声明矛盾", "Project Mogul 解释"], "seed_sources": ["..."] }
```

### ② s2_research — 调研 + 事实核查 🛡️ 护栏#1
- **In**: `topic.json`
- **Do**: LLM 围绕案例产出**带证据等级的事实清单**，交叉核查（区分「有记录的事实 / 目击者主张 / 阴谋论推测」）。
  若整体证据等级过低（全是 unverified），**标记 `blocked` 并跳过该选题**（记录原因）。
- **Out**: `jobs/<slug>/research.json`
```json
{ "facts": [
    { "claim": "1947年7月军方新闻稿称回收一只'飞碟'", "evidence": "documented", "source": "RAAF 新闻稿存档" },
    { "claim": "目击者称看到非人类残骸", "evidence": "claim", "source": "证人回忆，无实证" } ],
  "evidence_summary": { "documented": 6, "claim": 4, "speculation": 2 },
  "blocked": false }
```
- **证据等级**：`documented`(有档案/记录) > `claim`(目击/证词) > `speculation`(推测)。脚本里必须如实区分口吻。

### ③ s3_script — 写脚本
- **In**: `research.json`
- **Do**: LLM 写**原创中文悬疑解说**，分段，每段附「画面提示」(visual_query) 与时长预估；
  口吻遵守证据等级（不把 speculation 说成事实）。结构：钩子 → 背景 → 事件经过 → 疑点 → 各方解释 → 开放式收尾 + 订阅引导。
- **Out**: `jobs/<slug>/script.json`
```json
{ "title_options": ["...","..."],
  "segments": [
    { "id": 1, "text": "1947年7月的新墨西哥沙漠，一个牧场主发现了一片奇怪的金属碎片……",
      "visual_query": "1947 new mexico desert ranch debris", "est_sec": 9 } ],
  "total_words": 2400, "est_minutes": 12 }
```

### ④ s4_tts — 配音
- **In**: `script.json`
- **Do**: 逐段用 **edge-tts** 合成音频，并采集 **WordBoundary 字级时间戳**。语速/音调按「探索腔」配置。
- **Out**: `jobs/<slug>/audio/seg_*.mp3` + `jobs/<slug>/timing.json`（每段每词的 offset/duration）

### ⑤ s5_subs — 字幕（无需 Whisper）
- **In**: `script.json` + `timing.json`
- **Do**: 用我们已知的文稿 + TTS 时间戳，切分成阅读友好的字幕行，生成带样式的 `.ass`。
- **Out**: `jobs/<slug>/subs.ass`（+ 可选 `subs.srt`）

### ⑥ s6_media — 取合法素材 🛡️ 护栏#2
- **In**: `script.json`（各段 visual_query）
- **Do**: 对每段 visual_query，依次查询**白名单源**：NASA Images API / archive.org / Wikimedia Commons；
  **每个素材必须读取并记录其 license**（仅接受 Public Domain / CC），保存归属信息；
  查不到则回退到通用公版图 + Ken Burns。**绝不抓取非白名单源（尤其他人 YouTube）。**
- **Out**: `jobs/<slug>/media/seg_*.{jpg,mp4}` + `jobs/<slug>/media/credits.json`（含每素材来源+license+attribution）

### ⑦ s7_assemble — 合成
- **In**: `audio/`, `timing.json`, `subs.ass`, `media/`, `assets/`(片头尾/BGM/字体)
- **Do**: ffmpeg 构建时间轴：每段画面（图片走 `zoompan` 推拉）对齐其音频；叠字幕；BGM 在旁白下闪避降音；加片头/片尾。
- **Out**: `jobs/<slug>/video.mp4`（1080p, H.264/AAC）

### ⑧ s8_meta — 元数据 + 封面
- **In**: `script.json`, `research.json`, `media/credits.json`
- **Do**: LLM 写 SEO 标题/简介（含素材出处归属，满足 CC 署名）/标签；Pillow 合成封面（公版图 + 大字标题）。
- **Out**: `jobs/<slug>/meta.json` + `jobs/<slug>/thumb.jpg`

### ⑨ s9_publish — 发布 🛡️ 护栏#3
- **In**: `video.mp4`, `meta.json`, `thumb.jpg`
- **Do**: YouTube Data API v3 `videos.insert` 上传 + 设封面。**发布可见性由 config/CLI 控制，默认 `private`（草稿）**。
- **Out**: `jobs/<slug>/publish.json`（videoId, url, status, 时间）

---

## 7. 三条护栏 (Guardrails)

1. **事实核查关 (②)**：每个说法标证据等级；证据过弱的选题自动 `blocked`，避免产出伪科学/阴谋论被限流。
2. **素材合法性锁 (⑥)**：素材来源**白名单制**，逐个校验 license 仅收 PD/CC 并记录归属；非白名单源一律拒绝。
3. **发布暂停开关 (⑨)**：默认上传为 `private`。频道早期人工过一眼标题/封面/事实，跑顺后再在 config 改 `public` 或开自动。
   规避 YouTube 对「批量自动上传低质内容」的 spam 政策。

护栏严格度在 `config.yaml` 可调（如 `guardrails.min_documented_facts: 3`）。

---

## 8. 技术选型 & 升级路径 (Tech Stack)

| 环节 | 免费档 (MVP) | 升级路径（按瓶颈优先级） |
|---|---|---|
| 脚本/调研/元数据 LLM | **Gemini 免费额度**（Toronto 可直连） | **① 换 Claude API 写脚本+调研**：`claude-opus-4-8`≈¥0.7/集 或 `claude-sonnet-4-6`≈¥0.5/集；中文悬疑叙事明显更好；可挂 `web_search` 工具联网核查、带来源，把②③两步合一 |
| 配音 TTS | **edge-tts**（免费微软中文神经音 + 字级时间戳） | Azure TTS / MiniMax / 豆包（更强情感/克隆） |
| 字幕 | 文稿 + TTS 时间戳（零成本、精确） | —（已足够） |
| 画面素材 | NASA / archive.org / Wikimedia 公版 + Ken Burns | ② 免费云出图(Pollinations) → 付费即梦/Recraft |
| 合成 | **ffmpeg**（已装 v8） | moviepy / 模板化转场 |
| 封面 | Pillow 合成 | AI 封面生成 |
| 上传 | YouTube Data API v3 | 同（加定时/批量） |
| 编排 | `run.py` CLI | Prefect / cron 无人值守批量 |

已具备：python3.14, node24, ffmpeg8, yt-dlp, Pillow, requests, curl, jq, git。
需补 pip：`edge-tts`, `google-api-python-client google-auth-oauthlib`（上传）, `google-generativeai`(或所选 LLM SDK), `PyYAML`。
`yt-dlp` 仅用于**明确合法源**（CC 授权 / 你自己的内容），不用于搬运。

---

## 9. 配置 (config.yaml 关键字段)

```yaml
cost_tier: free                 # free | mixed | paid
llm:
  provider: gemini              # gemini | claude | openai
  script_provider: gemini       # 可单独给脚本指定更强模型
tts:
  voice: zh-CN-YunyangNeural
  rate: "-8%"
video: { resolution: 1080p, fps: 30 }
guardrails:
  min_documented_facts: 3       # 低于则 block 选题
  allowed_licenses: [public-domain, cc0, cc-by, cc-by-sa]
publish:
  visibility: private           # private | unlisted | public  (默认 private = 护栏#3)
  auto_upload: false
```
密钥走 `.env` / `secrets/`（不进 git）。

---

## 10. 错误处理 & 可续跑

- 每阶段产物即 checkpoint；失败重跑只从失败阶段继续（`--resume`）。
- 阶段内失败（如某段素材查不到）→ 回退策略（通用公版图），并写 `warnings` 到产物，不中断整条。
- `②` 选题被 `blocked` → 编排器记录并（可选）自动取下一个种子案例。
- 所有外部请求带本地缓存（避免重复抓取/重复花额度）。

---

## 11. 测试策略 (Testing)

- 每个 stage 模块对「样例输入 json → 期望输出形状」做单测（可用 fixture，不打真实网络）。
- `llm.py` / `sources/*` / `s9_publish` 走 mock，验证契约而非外部服务。
- 一个 `--dry-run` 端到端：用内置 fixture 案例跑完 ①–⑦ 出一个短样片，不上传。
- ffmpeg 合成做「能生成有效 mp4 且时长≈音频总长」的冒烟测试。

---

## 12. 已定默认值（确认时采用，spec 审阅可改）

- LLM 免费档 = **Gemini 免费额度**（用户确认「先全免费跑通」，0 成本验证整条管线）。脚本质量是第一升级位 → **Claude API**（`claude-opus-4-8`≈¥0.7/集 或 `claude-sonnet-4-6`≈¥0.5/集，一天一集约 ¥15–25/月）。注意区分：**Claude Code 交互式 = 半自动**（你在场）；**全自动需走 Anthropic SDK 程序化调用**，并可挂 `web_search` 让脚本联网、带来源，直接服务护栏#1。`llm.py` 的 provider 抽象已为此预留（gemini ↔ claude 仅改 config）。
- 选题来源 = 我先建一个**种子案例库** `seeds/cases.yaml`（罗斯威尔1947 / 凤凰城之光1997 / Nimitz Tic-Tac 2004 / 通古斯1908 / 兰德申森林1980 / Travis Walton 1975 / 比利时UFO波1989 / Kenneth Arnold 1947 …），LLM 在其上扩展；自动发现选题列为后续。
- 发布默认 `private`，`auto_upload: false`。

---

## 13. 风险 (Risks)

| 风险 | 缓解 |
|---|---|
| 免费 LLM 脚本「AI 味」重、留存差 | 优先升级脚本 LLM；强叙事模板与钩子规范 |
| 公版素材与具体案例画面不够贴 | Ken Burns + 氛围化通用素材 + 后续 AI 出图 |
| YouTube API 上传配额（默认 ~6 上传/天） | 默认够用；批量再申请提额 |
| 误收非合法 license 素材 | ⑥ 强制逐个校验 license，白名单制 |
| 事实错误/阴谋论被限流 | ② 证据等级 + block 机制 + private 默认人工复核 |

---

## 14. 未来 (Out of Scope, Future)

付费档全开（云 TTS + AI 出图）、Prefect/cron 无人值守批量、多平台分发、A/B 标题封面、数据回流选题（看哪类案例数据好就多做）。
