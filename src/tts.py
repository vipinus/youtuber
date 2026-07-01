from __future__ import annotations

from typing import Protocol

from src.config import Config


class TTSEngine(Protocol):
    def synth(self, text: str) -> tuple[bytes, list[dict]]: ...


class FakeTTS:
    """确定性离线 TTS：每字一个 word，每字 0.3s。"""

    def synth(self, text: str) -> tuple[bytes, list[dict]]:
        words = []
        t = 0.0
        for ch in text:
            words.append({"text": ch, "offset_sec": round(t, 3), "dur_sec": 0.3})
            t += 0.3
        return b"FAKEAUDIO", words


class EdgeTTS:
    """edge-tts 免费微软中文神经音 + 字级时间戳。惰性 import。"""

    def __init__(self, voice: str = "zh-CN-YunyangNeural", rate: str = "-8%"):
        self.voice = voice
        self.rate = rate

    def synth(self, text: str) -> tuple[bytes, list[dict]]:
        import asyncio

        import edge_tts

        async def _go() -> tuple[bytes, list[dict]]:
            # edge-tts>=7.x 默认 boundary="SentenceBoundary"，
            # 需显式传 boundary="WordBoundary" 才能拿到字级时间戳。
            comm = edge_tts.Communicate(
                text, voice=self.voice, rate=self.rate, boundary="WordBoundary"
            )
            audio = bytearray()
            words: list[dict] = []
            async for ch in comm.stream():
                if ch["type"] == "audio":
                    audio.extend(ch["data"])
                elif ch["type"] == "WordBoundary":
                    words.append({
                        "text": ch["text"],
                        "offset_sec": round(ch["offset"] / 1e7, 3),
                        "dur_sec": round(ch["duration"] / 1e7, 3),
                    })
            return bytes(audio), words

        return asyncio.run(_go())


def get_tts(config: Config) -> TTSEngine:
    provider = config.get("tts.provider", "edge")
    if provider == "fake":
        return FakeTTS()
    if provider == "edge":
        return EdgeTTS(
            config.get("tts.voice", "zh-CN-YunyangNeural"),
            config.get("tts.rate", "-8%"),
        )
    raise ValueError(f"未知 tts provider: {provider}")
