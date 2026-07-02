"""一次性 YouTube 授权：跑浏览器同意，把 token 存到指定路径。
用法：.venv/bin/python -m src.auth_youtube [token输出路径]
     默认 secrets/token.json;第二账号(如英文原片频道)传 secrets/token_en.json,
     浏览器里选对应的 Google 账号即可——同一个 client_secret 可给任意账号授权。
前置：secrets/client_secret.json（从 Google Cloud 下载，见 docs/youtube-oauth-setup.md）。
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.youtube import SCOPES


def main() -> None:
    from google_auth_oauthlib.flow import InstalledAppFlow

    token_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("secrets/token.json")
    token_path.parent.mkdir(parents=True, exist_ok=True)
    flow = InstalledAppFlow.from_client_secrets_file("secrets/client_secret.json", SCOPES)
    # access_type=offline + prompt=consent 确保拿到 refresh_token，否则无人值守次日会 401
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    token_path.write_text(creds.to_json(), encoding="utf-8")
    if not creds.refresh_token:
        print("警告：未拿到 refresh_token，长期自动上传会失效；请在同意屏幕重新授权")
    print(f"已授权，token 存到 {token_path}")


if __name__ == "__main__":
    main()
