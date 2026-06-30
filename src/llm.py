from __future__ import annotations

import os
from typing import Protocol

from src.config import Config


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
    """Gemini 免费额度。惰性 import，使测试无需安装 SDK。"""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = model

    def generate(self, system: str, prompt: str) -> str:
        model = self._genai.GenerativeModel(
            self._model_name, system_instruction=system
        )
        return model.generate_content(prompt).text


def get_llm(config: Config) -> LLMProvider:
    provider = config.get("llm.provider", "gemini")
    if provider == "fake":
        return FakeLLM()
    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("GEMINI_API_KEY 未设置（写进 .env 或环境变量）")
        return GeminiLLM(key, config.get("llm.gemini_model", "gemini-2.0-flash"))
    if provider == "claude":
        raise NotImplementedError("claude provider 在后续『LLM 升级』阶段实现")
    raise ValueError(f"未知 llm provider: {provider}")
