from __future__ import annotations

import os
import time
from typing import Protocol

from src.config import Config


def _is_transient(e: Exception) -> bool:
    """Gemini 瞬时服务端错误(高峰 503 / 500 / 502)——可退避重试；配额 429 不在此列，快速失败。"""
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    if code in (500, 502, 503):
        return True
    if type(e).__name__ in ("ConnectionError", "Timeout", "ReadTimeout", "ConnectTimeout"):
        return True                       # 网络抖动/对端重启(如 eros 的 ollama)——可重试
    s = str(e)
    return any(k in s for k in ("UNAVAILABLE", "INTERNAL", "503", "500", "high demand"))


class LLMProvider(Protocol):
    def generate(self, system: str, prompt: str) -> str: ...


class FakeLLM:
    """测试用：返回固定字符串，记录调用。不联网。"""

    def __init__(self, canned: str = ""):
        self.canned = canned
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        return self.canned


class GeminiLLM:
    """Gemini（google-genai）。惰性 import，使单测无需 SDK / 网络。"""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, system: str, prompt: str, _tries: int = 6) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(system_instruction=system)
        last: Exception | None = None
        for attempt in range(_tries):
            try:
                resp = self._client.models.generate_content(
                    model=self._model, contents=prompt, config=cfg)
                return resp.text or ""
            except Exception as e:                        # noqa: BLE001
                if not _is_transient(e) or attempt == _tries - 1:
                    raise
                last = e
                time.sleep(min(3 * 2 ** attempt, 40))     # 3,6,12,24,40,40s —— 扛 Gemini 高峰尖峰
        raise last  # pragma: no cover — 循环要么 return 要么 raise


class OllamaLLM:
    """Ollama(/api/chat, stream=false)。惰性 import requests，单测离线安全。

    生产指向 eros 的 Ollama(云端模型如 gemma4:31b-cloud 在 Ollama 云上跑,
    本机零权重下载、无 Gemini 免费档 20次/天 的日配额枷锁)。瞬时 5xx 沿用
    GeminiLLM 同款退避重试;4xx(如云端模型 403 需订阅)立即抛,不空转。"""

    def __init__(self, host: str, model: str, temperature: float = 0.3):
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature

    def generate(self, system: str, prompt: str, _tries: int = 6) -> str:
        import requests

        last: Exception | None = None
        for attempt in range(_tries):
            try:
                r = requests.post(
                    f"{self.host}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [{"role": "system", "content": system},
                                      {"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {"temperature": self.temperature},
                    },
                    timeout=300,
                )
                if r.status_code >= 500:
                    raise RuntimeError(f"ollama {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                return (r.json().get("message") or {}).get("content", "") or ""
            except Exception as e:                    # noqa: BLE001
                if not _is_transient(e) or attempt == _tries - 1:
                    raise
                last = e
                time.sleep(min(3 * 2 ** attempt, 40))
        raise last  # pragma: no cover


def get_llm(config: Config) -> LLMProvider:
    provider = config.get("llm.provider", "gemini")
    if provider == "fake":
        return FakeLLM()
    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("GEMINI_API_KEY 未设置（写进 .env 或环境变量）")
        return GeminiLLM(key, config.get("llm.gemini_model", "gemini-2.5-flash"))
    if provider == "ollama":
        return OllamaLLM(
            config.get("llm.ollama_host", "http://eros:11434"),
            config.get("llm.ollama_model", "gemma4:31b-cloud"),
            config.get("llm.temperature", 0.3),
        )
    if provider == "claude":
        raise NotImplementedError("claude provider 在后续『LLM 升级』阶段实现")
    raise ValueError(f"未知 llm provider: {provider}")
