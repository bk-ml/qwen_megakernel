#!/usr/bin/env python3
"""
Minimal Pipecat pipeline:  Speech → STT → megakernel LLM → Qwen3-TTS → speakers

GPU host (one process):
  uv pip install -r requirements-voice.txt
  python -m qwen_megakernel.server.app          # :8000

Client (same machine or laptop with mic):
  uv pip install -r requirements-pipecat.txt
  export VOICE_SERVER_URL=http://<gpu-host>:8000
  python examples/pipecat_voice_agent.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from pipecat.audio.vad.vad_analyzer import VADParams
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import wave
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from loguru import logger
from pipecat.frames.frames import InputAudioRawFrame
import threading
import pyaudio
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.frames.frames import StartFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
try:
    import pyaudio
except ImportError:
    pyaudio = None  # type: ignore[assignment]

from integrations.pipecat.megakernel_llm import create_megakernel_llm_service
from integrations.pipecat.qwen_tts import Qwen3TTSService
from integrations.pipecat.voice_client import check_voice_server, voice_server_base

import numpy as np
import pyaudio

pa = pyaudio.PyAudio()
# for i in range(pa.get_device_count()):
#     print(i, pa.get_device_info_by_index(i)["name"])


class WaveFileSink(FrameProcessor):
    def __init__(self, filename="output.wav", sample_rate=24000):
        super().__init__()
        self._filename = filename
        self._sample_rate = sample_rate
        self._wav = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if hasattr(frame, "audio") and frame.audio:
            if self._wav is None:
                self._wav = wave.open(self._filename, "w")
                self._wav.setnchannels(1)
                self._wav.setsampwidth(2)
                self._wav.setframerate(self._sample_rate)
            self._wav.writeframes(frame.audio)
        await self.push_frame(frame, direction)

    async def cleanup(self):
        if self._wav:
            self._wav.close()

class DebugAllFrames(FrameProcessor):
    def _check_started(self, frame):
        return True
    
    async def process_frame(self, frame, direction):
        frame_name = type(frame).__name__
        if "Speaking" in frame_name or "VAD" in frame_name or "Speech" in frame_name:
            print(f"VAD EVENT: {frame_name}")  # already have this
        if hasattr(frame, "audio"):
            arr = np.frombuffer(frame.audio, dtype=np.int16)
            level = abs(arr).mean()
            if level > 200:  # only print high levels
                print(f"AUDIO LEVEL: {level}")
        await self.push_frame(frame, direction)

    


class TranscriptionLogger(FrameProcessor):
    """Log STT output so it's clear the mic pipeline is working."""

    async def process_frame(self, frame, direction: FrameDirection):
        # print(f"FRAME TYPE: {type(frame).__name__}")  # add this
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            logger.info(f"Heard: {frame.text.strip()!r}")
            print(f"TranscriptionLogger - User: {frame.text.strip()}")
        await self.push_frame(frame, direction)


def ensure_local_microphone() -> None:
    """Fail fast on headless GPU boxes (no ALSA/PyAudio input device)."""
    if os.getenv("PIPCAT_SKIP_AUDIO_CHECK"):
        return
    if pyaudio is None:
        raise SystemExit(
            "PyAudio is required for the mic client. Install with:\n"
            "  pip install -r requirements-pipecat.txt\n"
            "(needs pipecat-ai[local]; on macOS: brew install portaudio)"
        )
    pa = pyaudio.PyAudio()
    try:
        pa.get_default_input_device_info()
    except OSError as exc:
        raise SystemExit(
            "No microphone on this machine.\n"
            "Run the voice server on the GPU host, then run this script on your "
            "laptop with an SSH tunnel:\n"
            "  ssh -N -p 46210 root@HOST -L 8080:127.0.0.1:8080\n"
            "  export VOICE_SERVER_URL=http://127.0.0.1:8080\n"
            "  python examples/pipecat_voice_agent.py\n"
            f"({exc})"
        ) from exc
    finally:
        pa.terminate()


async def ensure_voice_server() -> None:
    base = voice_server_base()
    logger.info(f"Checking voice server at {base} ...")
    try:
        health = await check_voice_server()
    except Exception as exc:
        detail = str(exc).strip() or type(exc).__name__
        raise SystemExit(
            f"Voice server not reachable at {base}: {detail}\n\n"
            "On the GPU host:\n"
            "  python -m qwen_megakernel.server.app --port 8080\n\n"
            "On your Mac (separate terminal, keep open):\n"
            "  ssh -N -p 46210 root@142.171.48.138 -L 8080:127.0.0.1:8080\n\n"
            "Then verify on Mac:\n"
            "  curl http://127.0.0.1:8080/health\n"
        ) from exc

    if not health.get("tts"):
        logger.warning("Server reports TTS unavailable — install qwen-tts on the GPU host")
    logger.info(f"Voice server OK (llm={health.get('llm')}, tts={health.get('tts')})")


async def main() -> None:
    await ensure_voice_server()
    ensure_local_microphone()

    logger.info("Initializing local audio (mic + speakers)...")
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(
                confidence=0.8,
                start_secs=0.5,
                stop_secs=0.8,
                min_volume=0.6
            )),
            vad_enabled=True,  # ← ADD THIS, might be missing!
            min_silence_duration_ms=300,
            start_immediately=True,
        )
    )

    
    # converts audio → text.
    whisper_model = os.getenv("WHISPER_MODEL", "base")
    whisper_device = os.getenv("WHISPER_DEVICE", "cpu")
    logger.info(
        f"Loading Whisper STT model={whisper_model!r} device={whisper_device!r} "
        "(first run may download the model)..."
    )
    stt = WhisperSTTService(
        settings=WhisperSTTService.Settings(model=whisper_model),
        device=whisper_device,
    )
    if getattr(stt, "_model", None) is None:
        raise SystemExit(
            "Whisper failed to load. Install STT deps:\n"
            "  pip install 'pipecat-ai[whisper]'"
        )
    logger.info("Whisper STT ready.")

    print("LLM started")

    # llm to generate text tokens from text
    llm = create_megakernel_llm_service()

    print("LLM finished")
    # tts to generate audio chunks from text tokens
    tts = Qwen3TTSService(sample_rate=24000, language="english")

    # context for system prompt
    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly and helpful voice assistant. /no_think "
                    "Keep all responses under two sentences. "
                    "Be direct and conversational — no lists, no markdown, no bullet points. "
                    "Never say 'As an AI' or similar phrases. "
                    "If you don't know something, say so briefly."
                ),
            }
        ]
    )

    # aggregators to collect user and assistant messages
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(
                confidence=0.3,
                start_secs=0.1,
                stop_secs=0.3,
                min_volume=0.0
            )),
        ),
    )

    # pipeline to combine all the components
    pipeline = Pipeline([
        transport.input(), 
        # DebugAllFrames(),
        stt,
        TranscriptionLogger(),
        user_agg,
        llm,
        tts,
        WaveFileSink("output.wav"),
        transport.output(),
        assistant_agg,
    ])

    # task to run the pipeline
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
    )

    logger.info(
        "Ready — speak into the microphone. "
        "Pause briefly after you finish; VAD detects end-of-speech."
    )
    await PipelineRunner().run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
