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
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from integrations.pipecat.megakernel_llm import create_megakernel_llm_service
from integrations.pipecat.qwen_tts import Qwen3TTSService
from integrations.pipecat.voice_client import voice_server_base


async def main() -> None:
    logger.info(f"Voice server: {voice_server_base()}")

    # handles audio input/output (mic and speakers)
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    # converts audio → text.
    stt = WhisperSTTService(
        model=os.getenv("WHISPER_MODEL", "base"),
        device=os.getenv("WHISPER_DEVICE", "cpu"),
    )

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

    logger.info("Listening — speak into the microphone.")
    await PipelineRunner().run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
