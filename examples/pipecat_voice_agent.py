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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

try:
    import pyaudio
except ImportError:
    pyaudio = None  # type: ignore[assignment]

from integrations.pipecat.megakernel_llm import create_megakernel_llm_service
from integrations.pipecat.qwen_tts import Qwen3TTSService
from integrations.pipecat.voice_client import check_voice_server, voice_server_base


class TranscriptionLogger(FrameProcessor):
    """Log STT output so it's clear the mic pipeline is working."""

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            logger.info(f"Heard: {frame.text.strip()!r}")
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
        raise SystemExit(
            f"Voice server not reachable at {base}: {exc}\n"
            "Start it with: python -m qwen_megakernel.server.app"
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
            vad_analyzer=SileroVADAnalyzer(),
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

    # llm to generate text tokens from text
    llm = create_megakernel_llm_service()

    # tts to generate audio chunks from text tokens
    tts = Qwen3TTSService(sample_rate=24000)

    # context for system prompt
    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise voice assistant. "
                    "Reply in one or two short sentences. No markdown."
                ),
            }
        ]
    )

    # aggregators to collect user and assistant messages
    user_agg, assistant_agg = LLMContextAggregatorPair(context)

    # pipeline to combine all the components
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            TranscriptionLogger(),
            user_agg,
            llm,
            tts,
            transport.output(),
            assistant_agg,
        ]
    )

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
