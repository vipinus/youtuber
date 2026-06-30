from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import Config
from src.llm import LLMProvider


@dataclass
class Ctx:
    slug: str
    config: Config
    root: Path
    llm: LLMProvider

    @property
    def job(self) -> Path:
        return self.root / self.slug
