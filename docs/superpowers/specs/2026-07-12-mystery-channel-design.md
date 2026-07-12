# 新频道「未解之谜」管线设计（mystery / m 系列）

日期：2026-07-12　状态：待用户审阅

## 背景与目标

在现有 original(s1–s9) 和 doc(d1–d11) 之外新增第三条管线 `mystery`，做一个
未解之谜纪录片新频道。与 doc 管线的本质区别：**不下载源视频**——只取热门
英文未解之谜视频的字幕当事实素材，文稿深度改写成原创中文解说，画面全部由
文生视频模型新生成。成片 8–15 分钟。

**本期范围到成片为止**（video.mp4 + thumb.jpg + meta.json，用户人工看效果）。
发布（新频道 OAuth/B 站/互链/归档）是下一期，不在本 spec 内。

## 已拍板的决策

| 决策点 | 结论 |
|---|---|
| 素材形态 | LLM 分镜 + 文生视频模型生成动态片段（非 SDXL 静图方案） |
| 文稿版权口径 | 字幕只当事实源，LLM **深度改写**为独立中文解说稿，不逐句翻译 |
| m2/m3 的 LLM | 均用现有 gemma4:31b-cloud（eros ollama 云端）；**不再需要在 zion 跑 12b** |
| 成片长度 | 8–15 分钟 |
| 配音 | 克隆「科幻FANS」解说音色（参考视频 https://www.youtube.com/watch?v=Ilfls0FTJoM ，中文→中文同语种克隆）。引擎用 **CosyVoice2**（用户明确不要 F5，CosyVoice2 本就是 2026-07-05 A/B 胜出的生产引擎），失败段回退 edge-tts。风险已提示（模仿真人声音可能被本人申诉），用户拍板照做，冷启动私享审看护栏不变 |
| BGM | 按片种选：`assets/bgm/bgm.json` 清单加情绪/类型标签（悬疑/惊悚/科幻/历史…），m2 给本集标类型，m7 按标签从曲目池选配 |
| 文生视频模型 | 首选 LTX-Video 2B，备选 Wan2.1-1.3B——由先行 spike 在 zion（RTX 3070 Ti 8GB）实测定夺 |

## 管线阶段

`python -m src.run --case <slug> --pipeline mystery --config config-mystery.yaml`

| 阶段 | 产物 | 说明 |
|---|---|---|
| m1_scout | source.json + en_subs.json3 | yt-dlp `ytsearch50` 检索未解之谜纪录片，按播放量排序，过滤（时长≥8min、英文、有字幕、registry 未发过），选一部只下字幕（`--skip-download --write-auto-subs --sub-format json3`）。复用 `src/util/ytsubs.py` 解析成句级转录段 |
| m2_script | script.json | gemma4:31b-cloud 两步：①从字幕提取案件事实清单（护栏 `min_documented_facts`，不够即 StageBlocked）；②以事实为素材重写原创中文解说稿，分段（每段 2–4 句），并给全集标类型标签（bgm 选曲用） |
| m3_storyboard | storyboard.json | 同一 31b-cloud 逐段写英文文生视频提示词，附加统一画风后缀（暗调、电影感、纪录片质感），负向词排除文字/水印（SD 系画字是乱码的老教训同样适用） |
| m4_media | media/seg_*.mp4 | 新 `scripts/zion_t2v.py`（经 `~/gpu-py` 或独立 wrapper，同 zion_sdxl 模式）逐段生成 ~5s 片段；每段 1 个片段省 GPU。单段失败回退 SDXL 静图（复用 zion_sdxl.py），只告警不中止 |
| m5_tts | narration.json + narration/*.wav | CosyVoice2 克隆，**固定参考模式**：参考音频不再逐段 pick_reference，恒用 `assets/voice_ref/` 下的科幻FANS 音色切片。段级失败回退 edge-tts（沿用 `tts.fallback_provider` 机制） |
| m6_subs | subs.ass | 复用现有断句规则（`subs.max_chars: 20`，句末标点必断） |
| m7_assemble | video.mp4 | 每段片段延展到配音时长（慢放+来回放/boomerang），拼接；按 m2 类型标签选 BGM 铺底（sidechain duck，`p7.bg_gain` 同款）；烧字幕；zion 上 nvenc（回退 libx264） |
| m8_meta | thumb.jpg + meta.json | 复用 d8 套路：SDXL 缩略图背景 + Pillow 叠中文标题 + LLM 标题/简介/标签 |

## 一次性预处理（不进管线，脚本化）

`scripts/make_voice_ref.py`：yt-dlp 下载参考视频音频 → demucs 分离人声 →
自动挑一段 10–15s 干净切片（复用能量门思想，max_volume < -45dB 拒用）→
whisper 转出参考文本 → 存 `assets/voice_ref/ref.wav + ref.json`（CosyVoice2
zero-shot 同语种克隆需要参考文本）。人工听一遍确认切片干净再投产。

## 先行验证 spike（写管线代码之前做）

1. **LTX-Video 2B 在 zion 出一段 5s 768×512 片段**：测显存峰值、耗时。
   过不了换 Wan2.1-1.3B 再测；都过不了回来改设计（退 SDXL+动效方案）。
2. **CosyVoice2 固定参考克隆**：拿科幻FANS 切片合成两句中文，人耳验收音色。

GPU 时长粗估：8–15min 成片 ≈ 60–90 段 ≈ 60–90 个片段，按每段 2–5 分钟生成
即 2–7.5 小时批跑，可过夜——spike 实测后修正。

## 错误处理

- 检索无合格候选 / 字幕拉不下来 / 事实不足护栏 → StageBlocked 停链（选题环节必须硬失败）。
- m4 单段生成失败 → SDXL 静图兜底；m5 单段克隆失败 → edge-tts 兜底。均只告警。
- bgm.json 缺类型标签或池空 → 回退随机选曲，告警不中止（同 p7.bgm 回退哲学）。

## 配置与多机

- 新 `config-mystery.yaml`（笔记本档：remote.provider=ssh 走 zion GPU）+
  `config-zion-mystery.yaml`（生产档：local + nvenc），沿用现有双机模式。
- registry 去重沿用 `YOUTUBER_REGISTRY`（NAS published.json），m1 选题查它防重复做题。

## 测试

沿用项目套路：m1 的 yt 检索、m4 的 T2V、LLM、TTS 全配 fake provider，每个
stage 一组 pytest（产物契约 + 护栏分支 + 回退分支）。`zion_t2v.py` 里可本地
单测的纯函数（时长延展计算等）保持零 torch 依赖。

## 开放问题（下一期）

- 新频道 YouTube 账号 OAuth（照 youtube_en 模式独立 token/client_secret）。
- 发布链（m9 系列）、Shorts、归档接入。
- BGM 曲库按类型扩充（现有池是否够悬疑/惊悚类）。
