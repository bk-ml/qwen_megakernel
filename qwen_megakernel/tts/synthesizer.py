"""Text → PCM chunks via Qwen3-TTS (HF talker or megakernel talker server)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import numpy as np

DEFAULT_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_SAMPLE_RATE = 24000


class Qwen3TTSSynthesizer:
    """Wraps official Qwen3-TTS generation and yields int16 PCM chunks."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device_map: str | dict[str, Any] = "cuda",
        dtype: Any = None,
    ):
        self.model_name = model_name or os.getenv("QWEN3_TTS_MODEL", DEFAULT_TTS_MODEL)
        self._wrapper = None
        self._device_map = device_map
        self._dtype = dtype

    def _load(self) -> None:
        if self._wrapper is not None:
            return
        import torch
        from qwen_tts import Qwen3TTSModel

        kwargs: dict[str, Any] = {"device_map": self._device_map}
        if self._dtype is not None:
            kwargs["dtype"] = self._dtype
        else:
            kwargs["dtype"] = torch.bfloat16

        self._wrapper = Qwen3TTSModel.from_pretrained(self.model_name, **kwargs)

    @property
    def sample_rate(self) -> int:
        return DEFAULT_SAMPLE_RATE

    def synthesize(
        self,
        text: str,
        *,
        speaker: str | None = None,
        language: str | None = None,
        **generate_kwargs: Any,
    ) -> np.ndarray:
        """Return mono float32 waveform in [-1, 1]."""
        self._load()
        speaker = speaker or os.getenv("QWEN3_TTS_SPEAKER", "serena")
        language = language or os.getenv("QWEN3_TTS_LANGUAGE", "English")
        wavs, sr = self._wrapper.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            **generate_kwargs,
        )
        if sr != self.sample_rate:
            raise RuntimeError(f"unexpected sample rate {sr}, expected {self.sample_rate}")
        wav = wavs[0]
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)
        return wav

    def synthesize_chunks(
        self,
        text: str,
        *,
        chunk_samples: int = 4800,
        **kwargs: Any,
    ) -> Iterator[tuple[np.ndarray, int]]:
        """Yield (int16_pcm, sample_rate) chunks for streaming playback."""
        wav = self.synthesize(text, **kwargs)
        pcm = np.clip(wav, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        for start in range(0, len(pcm), chunk_samples):
            yield pcm[start : start + chunk_samples], self.sample_rate


def is_qwen_tts_available() -> bool:
    try:
        import qwen_tts  # noqa: F401

        return True
    except ImportError:
        return False
