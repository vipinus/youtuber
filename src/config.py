from __future__ import annotations

import os
from pathlib import Path

import yaml


class Config:
    def __init__(self, data: dict):
        self._data = data or {}

    def get(self, dotted: str, default=None):
        cur = self._data
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur


def load_env(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def load_config(path: str | Path = "config.yaml") -> Config:
    load_env()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config(data)
