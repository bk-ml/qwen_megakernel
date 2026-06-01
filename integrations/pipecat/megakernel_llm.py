"""Pipecat LLM → voice server OpenAI-compatible /v1/chat/completions."""

from __future__ import annotations

import os

from integrations.pipecat.voice_client import voice_server_base


def create_megakernel_llm_service():
    """
    OpenAILLMService pointed at the local megakernel voice server.

    Pipecat streams tokens from POST /v1/chat/completions (SSE).
    """
    from pipecat.services.openai.llm import OpenAILLMService

    base = voice_server_base()
    # OpenAILLMService calls POST /v1/chat/completions internally
    return OpenAILLMService(
        api_key=os.getenv("MEGAKERNEL_API_KEY", "megakernel"),
        base_url=f"{base}/v1",
        model=os.getenv("MEGAKERNEL_MODEL_NAME", "qwen-megakernel"),
        stream=True,
    )
