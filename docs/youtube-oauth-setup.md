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

## 播放列表自动归类（需要重新授权一次）

`publish.playlist` 配置项打开后，上传成功的视频会自动加入指定标题的播放列表（找不到就新建）。
这需要比"仅上传"更大的权限，所以 `SCOPES` 从 `youtube.upload` 换成了完整的 `youtube`
（后者本身已经包含上传权限，不用两个 scope 叠加）。

**已有 `secrets/token.json` 的老授权不会自动失效**，上传仍然照常工作；但它没有播放列表权限，
第一次调用 `add_to_playlist` 会拿到 403 `insufficientPermissions`——管线不会因此丢视频
（照 thumbnail_error 的路数，videoId 正常写进 `publish.json`），只是在 `playlist_error` 字段
和运行日志里 warn 一句，提示你重新授权。

跑一次就好：

```
.venv/bin/python -m src.auth_youtube
```

浏览器会再弹一次同意屏幕（`prompt=consent` 保证拿到新 scope 的 refresh_token），同意后
`secrets/token.json` 被新 token 覆盖，之后播放列表就能正常写入了。不想用播放列表功能的话，
把 `config.yaml` 里的 `publish.playlist` 留空即可，完全跳过这一步、老 token 继续用。
