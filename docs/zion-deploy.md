# 部署到 zion 原生跑（P7 T14）

管线从「笔记本 ssh 到 zion 跑 GPU 步骤」迁移成「整条管线直接在 zion 上跑」——
zion 本身就是 GPU 主机（RTX 3070 Ti，NixOS），不用再走 scp/ssh 网络往返。

zion 侧环境已经配好（见 memory `zion-gpu-asr-setup`），本文档只记部署/运行流程。

## 1. 推代码到 zion

zion 上的仓库是 `~/youtuber`（`~/repos/youtuber.git` 的 clone），笔记本这边直接
`push` 到 zion 的裸仓库，再在 zion 上 `pull` 一下工作树：

```bash
# 笔记本
git push zion master

# zion（工作树 pull 一下，把裸仓库的新提交同步到 ~/youtuber）
ssh fabric@zion 'cd ~/youtuber && git pull'
```

`git push zion master` 要求笔记本的 `.git/config` 里已经配了 `zion` 这个 remote
（指向 `fabric@zion:~/repos/youtuber.git`）；没配过就先 `git remote add zion
fabric@zion:~/repos/youtuber.git`。

## 2. 跑管线

在 zion 上，用管线自己的 venv（`~/pipe-venv`，依赖已装好），配上专用的
`config-zion.yaml`（`remote.provider: local` 本机跑 GPU 脚本、`p7.encoder: nvenc`
硬件编码、`paths.title_font` 指向 zion 上装的 CJK 字体）：

```bash
ssh fabric@zion
cd ~/youtuber
~/pipe-venv/bin/python -m src.run --case <slug> --config config-zion.yaml
```

不用再传 `--remote`/额外参数——`config-zion.yaml` 里已经把所有 zion 专属项配好了，
跟笔记本用的 `config.yaml` 唯二的行为差异是：GPU 步骤走本机 subprocess（不 ssh/scp）、
视频编码走 nvenc（GPU 硬编，失败自动回退 libx264，不丢片子）。

## 3. secrets

`.env`、`secrets/client_secret.json`、`secrets/token.json` 都是 `.gitignore` 的，
`git push`/`pull` 不会带过去。zion 的 `~/youtuber` 工作树里已经手动放了一份（首次
部署时 `scp` 过去的）——**这些文件不会随 git 更新**，改了笔记本这边的 `.env`/密钥
要记得手动同步一份到 zion。

## 4. YouTube token 重新授权后的同步

`secrets/token.json` 一旦在笔记本上重新授权过（比如加了播放列表 scope、或者
token 过期重新走了一遍 `python -m src.auth_youtube`），zion 上那份旧 token 就跟着
失效了——**必须手动把新 token 覆盖过去**，不会自动同步：

```bash
scp secrets/token.json fabric@zion:~/youtuber/secrets/token.json
```

不同步的话 zion 上跑 s9/d9 上传阶段会拿着过期 token 报 401/403（不会崩管线，
`publish.json` 里会记 `playlist_error`/上传失败信息，但片子传不上去）。

## 参考

- `src/remote.py` `LocalGPU`：本机（zion）直接 subprocess 经 `~/gpu-py` 跑
  `scripts/zion_gpu_asr.py`/`scripts/zion_sdxl.py`，不 scp/不 ssh。
- `src/stages/d7_assemble.py` `p7.encoder`：`nvenc` 时用 `h264_nvenc`，需要
  `LD_LIBRARY_PATH` 带上 `/run/opengl-driver/lib`（`nvenc_env()` 已处理，
  可用 `p7.encoder_lib_path` 覆盖）。
- `config-zion.yaml`：zion 专用 config，跟 `config.yaml` 逐项对比差异见文件头注释。
