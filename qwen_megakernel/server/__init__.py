"""Single-process voice server: megakernel LLM + Qwen3-TTS."""

from qwen_megakernel.server.app import app, main

__all__ = ["app", "main"]
