# YouTube 上传一次性授权

1. 去 https://console.cloud.google.com 建个项目。
2. 「API 和服务」→启用 **YouTube Data API v3**。
3. 「凭据」→创建 **OAuth 客户端 ID**→类型选「桌面应用」→下载 JSON。
4. 把下载的 JSON 存为 `secrets/client_secret.json`（已在 .gitignore，不进库）。
5. 「OAuth 同意屏幕」把你自己的 Google 账号加为测试用户。
6. 跑：`.venv/bin/python -m src.auth_youtube` → 浏览器点授权 → 生成 `secrets/token.json`。
7. 之后管线 s9 就能自动上传（默认 private 草稿）。

注意：免费配额约 6 次上传/天（每次上传约 1600 配额，日配额 10000）；要更多可申请提额。

注意：设置**自定义封面**要求频道已完成手机验证（否则 `thumbnails().set` 会返回 403）。
管线对此做了容错——封面失败不会作废已上传的视频，videoId 照常返回，失败信息记在
`publish.json` 的 `thumbnail_error` 字段。先去 youtube.com/verify 验证一下即可。
