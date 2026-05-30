"""Qwen3-TTS — text in, PCM audio chunks out (same process as LLM server)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

import numpy as np

SAMPLE_RATE = 24000


@lru_cache(maxsize=1)
def _synthesizer():
    from qwen_megakernel.tts.synthesizer import Qwen3TTSSynthesizer

    return Qwen3TTSSynthesizer(
        model_name=os.getenv(
            "QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        )
    )


def is_tts_available() -> bool:
    try:
        from qwen_megakernel.tts.synthesizer import is_qwen_tts_available

        return is_qwen_tts_available()
    except ImportError:
        return False


def stream_pcm_chunks(
    text: str,
    *,
    speaker: str | None = None,
    language: str | None = None,
    chunk_samples: int = 4800,
    **kwargs: Any,
) -> Iterator[tuple[bytes, int]]:
    """Yield (int16_pcm_bytes, sample_rate) for streaming playback."""
    speaker = speaker or os.getenv("QWEN3_TTS_SPEAKER", "serena")
    language = language or os.getenv("QWEN3_TTS_LANGUAGE", "English")
    synth = _synthesizer()
    for chunk, sr in synth.synthesize_chunks(
        text,
        speaker=speaker,
        language=language,
        chunk_samples=chunk_samples,
        **kwargs,
    ):
        pcm = np.clip(chunk, -32768, 32767).astype(np.int16)
        yield pcm.tobytes(), sr
