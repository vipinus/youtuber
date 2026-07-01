# P6 视频素材源 + s7 视频合成 实现计划

目标：把空镜从「NASA 静图」升级为「archive.org 公有领域影片(视频片段)+ 免费商用空镜补」，
让画面既是**动态视频**又**贴题**，且**绝不碰版权**（护栏#2 按 license 严判）。

## 实测结论（2026-07-01 live 探明，已去风险）
- archive.org `advancedsearch.php?q=(QUERY) AND mediatype:movies&output=json` 可搜到影片。
- **一半结果是坑**：`collection` 为 altcensored / mirrortube / PeriscopeFilm 等 = 镜像的 YouTube 视频、
  **无 licenseurl 字段** → 很可能有版权，必须拒。
- **安全的可识别**：有明确 `licenseurl`（如 `creativecommons.org/publicdomain/zero/1.0/`、`/licenses/by/`）
  或政府/PD 集合（FedFlix / usgovfilms / us-national-archives / prelinger / nasa）。
- `metadata/<id>` 返回 `files[]`，含 `format`（"h.264" / "MPEG4" / "512Kb MPEG4" / "Ogg Video"）；
  下载 URL = `archive.org/download/<id>/<filename>`。示例 `gov.archives.arc.1663615.mp4`。

## 护栏#2（合法素材）—— 视频源必须严守
只接受满足**任一**且**其余一律拒**：
1. `licenseurl` 命中 PD/CC 白名单模式：`publicdomain`/`/zero/`→"public-domain"；`/licenses/by/`→"cc-by"；
   `/licenses/by-sa/`→"cc-by-sa"。
2. `collection` 命中硬编码 PD 集合白名单（FedFlix, usgovfilms, us-national-archives, prelinger, nasa, publicmovies）。
无 license 且不在 PD 集合 → **丢弃**（不是兜底，是根本不作为候选）。source 字段标 "archive"。

## 任务分解
- **T1 · ArchiveVideo 媒体源**（src/media.py）：`ArchiveVideo` 实现 MediaSource 协议
  （search 严过 license→候选，fetch 选最优 mp4 下载）；`get_media_source` 加 provider "archive"。
  ext="mp4"。离线单测（Fake HTTP）+ 一次 live 探针。优先选较小 mp4 变体（避免下 59MB 用 6 秒）。
- **T2 · s6 按候选扩展名命名**：seg_NNN.<ext>（视频 .mp4 / 图 .jpg），credits.json 记 file 全名。
  兜底仍 .jpg。保持护栏#2 逐词拆分逻辑不动。
- **T3 · s7 支持混合图/视频**：改为「逐段归一化成定长静音片段(按段音频时长 trim/loop + scale/crop 铺满 1080p)
  → concat → 混旁白音轨」。视频比图更需要这套归一化。从 credits.json 读每段真实 file（图或视频）。
- **T4 · 接线 + e2e + live**：config media.provider 支持 archive；九阶段 e2e 全 fake；roswell 真跑对比画面。

## 免费商用空镜补（后续 T5，需用户 API key）
Pexels/Pixabay/Coverr 补通用空镜（沙漠/夜空/军事）。需用户申请免费 API key → 仅在 key 存在时启用。
先不阻塞 archive.org 主线。

## 遗留/风险
- 下载体积：整片下载只用几秒，浪费带宽；MVP 选最小 mp4 变体，后续可 byte-range/时段截取优化。
- archive.org PD 影片多是长纪录片，随机截取的片段未必精准贴合旁白句意；比 NASA 静图强，但非逐句对齐。
