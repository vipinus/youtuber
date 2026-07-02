# P7 「整片翻译」管线 实现计划

**方向**：找完整的公有领域纪录片(首题材 太空/NASA)→ 整片翻译成中文视频重新发布。
合法前提：只用真 PD/CC 源(护栏#2 复用 P6 的 license 严判 + 内容黑名单 + 可信集合)。

## 已实测验通(zion GPU,样片端到端跑通)
- faster-whisper 转录:GPU 27x 实时、时间轴精确。demucs 去人声:CPU 51s/8min、旁白除~92%、背景留。
- 样片证明整链可行:PD源片 → 转录 → Gemini翻译 → edge-tts中文配音 → demucs背景垫 → 烧中文字幕 → 成片。
- zion 环境见 memory `zion-gpu-asr-setup`(faster-whisper GPU / demucs CPU --mp3 / LD_LIBRARY_PATH 坑)。

## 管线阶段(job 文件夹 file-based;复用现有 s4/s5/s8/s9 思路)
| 阶段 | 位置 | 产物 | 说明 |
|------|------|------|------|
| d1_source | 本地 | source.mp4, source.json | archive.org 挑 PD 纪录片(可信集合+黑名单+有效metadata+优选HD)并下载。复用 ArchiveVideo 门 |
| d2_gpu | **zion(SSH)** | transcript.json, no_vocals.mp3 | 推音频→zion 跑 faster-whisper(GPU)+demucs(CPU)→拉回。src/remote.py 编排 |
| d3_translate | 本地 | zh_script.json | Gemini 逐段 EN→ZH 保时间轴(对象包数组避免 extract_json 数组坑)。复用 LLM+503重试 |
| d4_tts | 本地 | audio/*.mp3, timing | edge-tts 中文旁白(rate -18%)。复用 EdgeTTS |
| d5_subs | 本地 | subs.ass | 中文字幕(复用 s5 断句) |
| d7_assemble | 本地 | video.mp4 | 源片视频 + (no_vocals背景 sidechain-duck + 中文旁白按句起点) + 烧字幕。扩展 s7 |
| d8_meta/d9_publish | 本地 | meta/publish.json | 复用 s8/s9 |

## 任务分解
- **T1 · src/remote.py + zion GPU 脚本**：SSH/scp 编排(带 LD_LIBRARY_PATH),部署 zion 端 gpu_asr.py(音频→transcript.json+no_vocals.mp3)。离线单测用 Fake remote。
- **T2 · d1_source**：archive.org PD 纪录片挑选+下载(可信集合、有效metadata、时长/清晰度过滤、内容黑名单)。
- **T3 · d2_gpu**：调 remote 完成转录+去人声,落 transcript.json/no_vocals.mp3。
- **T4 · d3_translate**：Gemini 批量翻译,保 id/时间轴。
- **T5 · d4_tts + d5_subs**：中文配音 + 中文字幕(复用)。
- **T6 · d7_assemble**：混音(背景 ducking + 旁白定位)+ 烧字幕 → video.mp4。
- **T7 · 编排 + config + e2e**：新 STAGES(或独立 run),全 fake e2e 不触网,roswell/NASA live。

## 关键设计点(已与用户对齐)
- 中文旁白锚定英文句起始时间;背景 sidechain ducking(说话压低)。
- 时长不齐:MVP 锚定起点;后续可语速自适应/时长压缩。
- production 用 NASA HD 源 + whisper large-v3;样片用 SD + medium.en。
- YouTube「重复利用内容」:翻译+配音+字幕算本地化转化;想更稳可加中文片头导读。

## 遗留/风险
- zion 依赖(远程 GPU 机必须在线);demucs GPU cuDNN 不兼容→固定 CPU。
- 长片(90min)转录~3min、demucs CPU~10min、下载大;首版限时长(如≤30min)。
- 源片清晰度参差(样片 SD)——d1 优先选高分辨率变体。
