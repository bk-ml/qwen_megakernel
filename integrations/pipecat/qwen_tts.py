"""Pipecat TTS → single voice server /v1/tts/ws."""

from __future__ import annotations

import base64
import os
from typing import AsyncGenerator

from loguru import logger

from pipecat.frames.frames import Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService, TextAggregationMode
from pipecat.transcriptions.language import Language

from integrations.pipecat.voice_client import AsyncTtsWebSocketClient


class Qwen3TTSService(TTSService):
    class Settings(TTSSettings):
        speaker: str | None = None
        language: str | None = None

    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        speaker: str | None = None,
        language: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        settings = settings or Qwen3TTSService.Settings(
            speaker=speaker or os.getenv("QWEN3_TTS_SPEAKER", "serena"),
            language=language or os.getenv("QWEN3_TTS_LANGUAGE", "English"),
        )
        super().__init__(
            sample_rate=sample_rate,
            text_aggregation_mode=TextAggregationMode.TOKEN,
            push_start_frame=True,
            push_stop_frames=True,
            settings=settings,
            **kwargs,
        )
        self._client = AsyncTtsWebSocketClient()

    def language_to_service_language(self, language: Language) -> str | None:
        return str(language.value) if language else self._settings.language

    async def run_tts(
        self, text: str, context_id: str
    ) -> AsyncGenerator[Frame | None, None]:
        logger.debug(f"Qwen3 TTS: {len(text)} chars")
        async for msg in self._client.stream_audio(
            text,
            speaker=self._settings.speaker,
            language=self._settings.language,
            context_id=context_id,
        ):
            if msg.get("type") == "audio":
                yield TTSAudioRawFrame(
                    audio=base64.b64decode(msg["pcm_base64"]),
                    sample_rate=int(msg.get("sample_rate", self.sample_rate)),
                    num_channels=int(msg.get("num_channels", 1)),
                )
        yield None
