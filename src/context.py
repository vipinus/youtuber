from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config import Config
from src.llm import LLMProvider


@dataclass
class Ctx:
    slug: str
    config: Config
    root: Path
    llm: LLMProvider
    warnings: list[str] = field(default_factory=list)

    @property
    def job(self) -> Path:
        return self.root / self.slug

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"[warn] {msg}")
