from __future__ import annotations

import json


def extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"LLM 输出里没有 JSON 对象: {text[:200]!r}")
    return json.loads(text[start : end + 1])
