# 首集上线 Checklist —— 「探索飞碟与外星人」

首集建议用 **`roswell-1947`**（证据最硬，稳过护栏#1 的「≥3 条 documented 事实」）。
全程命令都在仓库根目录、用 `.venv/bin/python`。

---

## 阶段 0 · 一次性设置（只做一次）

- [ ] **YouTube 授权（唯一人工步骤）** —— 照 `docs/youtube-oauth-setup.md`：
  1. Google Cloud 建项目 → 启用 **YouTube Data API v3**
  2. 建 **OAuth 客户端 ID（桌面应用）** → 下载 JSON 存为 `secrets/client_secret.json`
  3. 「OAuth 同意屏幕」把自己的 Google 账号加为**测试用户**
  4. 跑 `​.venv/bin/python -m src.auth_youtube` → 浏览器点授权 → 生成 `secrets/token.json`
  5. **验证拿到了 refresh_token**（否则无人值守次日会 401）：
     `grep -o refresh_token secrets/token.json` 应有输出
- [ ] **频道手机验证**（youtube.com/verify）—— 不验证的话自定义封面会 403（视频仍会上传，
      失败信息记在 `publish.json` 的 `thumbnail_error`，不影响主流程）
- [ ] **确认 Gemini key** 在 `.env`（`GEMINI_API_KEY=...`，已 gitignore）
- [ ] **体检**：`​.venv/bin/pytest -q` → **67 passed**

---

## 阶段 1 · 生成（不上传），本地过一眼

先跑到 s8 为止，产物全落地但**不发布**，人工审核后再决定上传：

```bash
.venv/bin/python -m src.run --case roswell-1947 --to s8_meta
```

产物在 `jobs/roswell-1947/`：
`topic.json → research.json → script.json → 音频 → 字幕 → media/ + credits.json → video.mp4 → meta.json + thumb.jpg`

- [ ] **护栏#1**：若 s2 抛 `StageBlocked`（documented 事实 < 3）→ 换证据更硬的案例，
      **不要**去调低 `min_documented_facts`（那等于关掉事实核查）
- [ ] **成片自检** `jobs/roswell-1947/video.mp4`：字幕烧对了？配音跟画面同步？画面是合法素材？
- [ ] **元数据自检** `meta.json`：标题吸引点击且 ≤ ~40 字、简介通顺、tags 合理；`thumb.jpg` 能看
- [ ] **版权自检** `credits.json`：每条 `license` 都在白名单
      （public-domain / cc0 / cc-by / cc-by-sa），`attribution` 有出处

> 配额提醒：一集约 4–6 次 Gemini 调用（s1/s2/s3/s8），免费档 20 次/天 → **一天别连跑太多集**。
> 要连续量产就先接 Claude LLM（`config.yaml` 的 `llm.provider` 改一行）。

---

## 阶段 2 · 上传（私享草稿）

审核通过后，只跑发布这一阶段（护栏#3：`auto_upload: true` + `visibility: private`）：

```bash
.venv/bin/python -m src.run --case roswell-1947 --from s9_publish
```

- [ ] 看 `jobs/roswell-1947/publish.json` → 里面有 `videoId` / `url` / `status: private`
- [ ] 打开那个 url（登录你的频道）确认是**私享草稿**
- [ ] 若有 `thumbnail_error` 字段 → 去做频道手机验证后，删掉 `publish.json` 用 `--force` 重跑 s9

---

## 阶段 3 · 人工闸门 → 公开

- [ ] 在 YouTube Studio 完整看一遍首集
- [ ] 满意 → 手动改成 Public（首集建议手动，别急着全自动公开）
- [ ] 稳定几集后，若要**全自动公开**：把 `config.yaml` 的 `publish.visibility` 改为 `public`

---

## 出问题时

| 现象 | 原因 / 处理 |
|------|------|
| s2 `StageBlocked` | 案例证据不足 3 条 → 换案例（nimitz-tictac / phoenix-lights 也可） |
| s9 `FileNotFoundError: token.json` | 没做阶段 0 的 OAuth，跑 `python -m src.auth_youtube` |
| `publish.json` 有 `thumbnail_error` | 频道未手机验证；视频已上传，验证后重传封面即可 |
| Gemini 429 / 配额 | 当天配额用尽 → 明天再跑，或切 Claude LLM |
| 某阶段想重跑 | 删掉该阶段的产物文件，或加 `--force`（配 `--from/--to` 限定范围） |

一次性设置做完后，日常出片就是一条命令：
`.venv/bin/python -m src.run --case <slug>`（存在产物的阶段自动跳过，可断点续跑）。
