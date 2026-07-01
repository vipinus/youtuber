"""一次性 YouTube 授权：跑浏览器同意，把 token 存到 secrets/token.json。
用法：.venv/bin/python -m src.auth_youtube
前置：secrets/client_secret.json（从 Google Cloud 下载，见 docs/youtube-oauth-setup.md）。
"""
from __future__ import annotations

from pathlib import Path

from src.youtube import SCOPES


def main() -> None:
    from google_auth_oauthlib.flow import InstalledAppFlow

    Path("secrets").mkdir(exist_ok=True)
    flow = InstalledAppFlow.from_client_secrets_file("secrets/client_secret.json", SCOPES)
    # access_type=offline + prompt=consent 确保拿到 refresh_token，否则无人值守次日会 401
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    Path("secrets/token.json").write_text(creds.to_json(), encoding="utf-8")
    if not creds.refresh_token:
        print("警告：未拿到 refresh_token，长期自动上传会失效；请在同意屏幕重新授权")
    print("已授权，token 存到 secrets/token.json")


if __name__ == "__main__":
    main()
