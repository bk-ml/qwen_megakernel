"""HTTP/WebSocket helpers for the single voice server."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any


def voice_server_base() -> str:
    return os.getenv("VOICE_SERVER_URL", "http://127.0.0.1:8000").rstrip("/")


def voice_server_ws() -> str:
    explicit = os.getenv("VOICE_SERVER_WS_URL")
    if explicit:
        return explicit.rstrip("/")
    base = voice_server_base()
    return base.replace("https://", "wss://").replace("http://", "ws://") + "/v1/tts/ws"


class AsyncTtsWebSocketClient:
    async def stream_audio(
        self,
        text: str,
        *,
        speaker: str | None = None,
        language: str | None = None,
        context_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        import websockets

        async with websockets.connect(voice_server_ws(), max_size=None) as ws:
            await ws.send(
                json.dumps(
                    {
                        "text": text,
                        "speaker": speaker,
                        "language": language,
                        "context_id": context_id,
                    }
                )
            )
            while True:
                msg = json.loads(await ws.recv()) # waits for next chunk of audio data
                print("TTS WebSocket message:" , msg)
                yield msg # immediately sends it forward to the client
                if msg.get("type") in ("done", "error"):
                    if msg.get("type") == "error":
                        raise RuntimeError(msg.get("message", "TTS error"))
                    break
