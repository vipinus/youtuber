from __future__ import annotations

import json
from pathlib import Path


def write_json(job: Path, name: str, data: dict) -> Path:
    job.mkdir(parents=True, exist_ok=True)
    p = job / name
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def read_json(job: Path, name: str) -> dict:
    return json.loads((job / name).read_text(encoding="utf-8"))


def exists(job: Path, name: str) -> bool:
    return (job / name).exists()
