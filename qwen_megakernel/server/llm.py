"""Megakernel LLM decode — prompt in, streamed text tokens out."""

from __future__ import annotations

import os
from collections.abc import Iterator

from qwen_megakernel.model import Decoder


class MegakernelLLM:
    """Single-session megakernel decoder for streaming generation."""

    def __init__(self, model_name: str | None = None):
        name = model_name or os.getenv("MEGAKERNEL_MODEL", "Qwen/Qwen3-0.6B")
        self._decoder = Decoder(model_name=name, verbose=True)
        self._eos_id = self._decoder.tokenizer.eos_token_id

    def stream_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
    ) -> Iterator[str]:
        """Yield decoded text fragments as each token is generated."""
        tok = self._decoder.tokenizer
        ids = tok.encode(prompt, add_special_tokens=True)
        if not ids:
            return

        dec = self._decoder
        dec.reset()
        for tid in ids[:-1]:
            dec.step(tid)

        current = ids[-1]
        for _ in range(max_tokens):
            current = dec.step(current)
            if current == self._eos_id:
                break
            piece = tok.decode([current], skip_special_tokens=True)
            if piece:
                yield piece

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
    ) -> Iterator[str]:
        """Format chat messages with the model tokenizer template, then stream."""
        tok = self._decoder.tokenizer
        if hasattr(tok, "apply_chat_template"):
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            parts = []
            for m in messages:
                parts.append(f"{m.get('role', 'user')}: {m.get('content', '')}")
            prompt = "\n".join(parts) + "\nassistant:"
        yield from self.stream_text(prompt, max_tokens=max_tokens)
