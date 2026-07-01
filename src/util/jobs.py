from __future__ import annotations

import json
from pathlib import Path


def write_json(job: Path, name: str, data: dict) -> Path:
    import os
    import tempfile

    job.mkdir(parents=True, exist_ok=True)
    p = job / name
    fd, tmp = tempfile.mkstemp(dir=job, prefix=f".{name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))
        os.replace(tmp, p)          # 原子替换
    except BaseException:
        os.unlink(tmp)
        raise
    return p


def read_json(job: Path, name: str) -> dict:
    return json.loads((job / name).read_text(encoding="utf-8"))


def exists(job: Path, name: str) -> bool:
    return (job / name).exists()
