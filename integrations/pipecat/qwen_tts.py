"""Pipecat TTS → single voice server /v1/tts/ws."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import AsyncGenerator

from loguru import logger

from pipecat.frames.frames import Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService, TextAggregationMode
from pipecat.transcriptions.language import Language

from integrations.pipecat.voice_client import AsyncTtsWebSocketClient


@dataclass
class Qwen3TTSSettings(TTSSettings):
    """Qwen3-TTS: ``voice`` is the CustomVoice speaker name (e.g. serena)."""
    model: str | None = "qwen3-tts"


class Qwen3TTSService(TTSService):
    Settings = Qwen3TTSSettings
    _settings: Qwen3TTSSettings

    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        speaker: str | None = None,
        language: str | None = None,
        settings: Qwen3TTSSettings | None = None,
        **kwargs,
    ):
        speaker = speaker or os.getenv("QWEN3_TTS_SPEAKER", "serena")
        language = language or os.getenv("QWEN3_TTS_LANGUAGE", "English")
        settings = settings or Qwen3TTSSettings(
            model="qwen3-tts",   # 👈 ADD THIS
            voice=speaker,
            language=language,
        )
        super().__init__(
            sample_rate=sample_rate,
            # Qwen3-TTS synthesizes full phrases; per-token calls are very slow.
            text_aggregation_mode=TextAggregationMode.SENTENCE,
            push_start_frame=True,
            push_stop_frames=True,
            settings=settings,
            **kwargs,
        )
        self._client = AsyncTtsWebSocketClient()

    def _speaker(self) -> str:
        return self._settings.voice or os.getenv("QWEN3_TTS_SPEAKER", "serena")

    def _language(self) -> str:
        lang = self._settings.language
        if lang is None:
            return os.getenv("QWEN3_TTS_LANGUAGE", "English")
        if isinstance(lang, Language):
            return str(lang.value)
        return str(lang)

    async def on_turn_context_created(self, context_id: str) -> None:
        try:
            await self._client.connect()
        except Exception as exc:
            logger.warning(f"TTS WebSocket pre-connect failed: {exc}")

    def language_to_service_language(self, language: Language) -> str | None:
        return str(language.value) if language else self._language()

    async def run_tts(
        self, text: str, context_id: str
    ) -> AsyncGenerator[Frame | None, None]:
        logger.debug(f"Qwen3 TTS: {len(text)} chars")
        print(f"🤖 Agent: {text}")
        async for msg in self._client.stream_audio(
            text,
            speaker=self._speaker(),
            language=self._language(),
            context_id=context_id,
        ):
            # logger.debug(f"Qwen3 TTS message: {msg}")
            if msg.get("type") == "audio":
                yield TTSAudioRawFrame(
                    audio=base64.b64decode(msg["pcm_base64"]),
                    sample_rate=int(msg.get("sample_rate", self.sample_rate)),
                    num_channels=int(msg.get("num_channels", 1)),
                )
        yield None
