"""HTTP/WebSocket helpers for the single voice server."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx


def voice_server_base() -> str:
    return os.getenv("VOICE_SERVER_URL", "http://127.0.0.1:8000").rstrip("/")


def voice_server_ws() -> str:
    explicit = os.getenv("VOICE_SERVER_WS_URL")
    if explicit:
        return explicit.rstrip("/")
    base = voice_server_base()
    return base.replace("https://", "wss://").replace("http://", "ws://") + "/v1/tts/ws"


async def check_voice_server(*, timeout: float = 5.0) -> dict[str, Any]:
    """Verify the voice server is reachable and report LLM/TTS readiness."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(f"{voice_server_base()}/health")
        r.raise_for_status()
        return r.json()


class AsyncTtsWebSocketClient:
    """Persistent TTS WebSocket — one connection, many synthesis requests."""

    def __init__(self) -> None:
        self._ws: Any | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        import websockets

        if self._ws is not None:
            return
        self._ws = await websockets.connect(voice_server_ws(), max_size=None)

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def _ensure_connected(self) -> None:
        if self._ws is None:
            await self.connect()
            return
        try:
            # websockets >= 13 uses ClientConnection.state; older uses .closed
            closed = getattr(self._ws, "closed", None)
            if closed is None:
                from websockets.protocol import State

                if self._ws.state != State.OPEN:
                    await self.connect()
            elif closed:
                self._ws = None
                await self.connect()
        except Exception:
            self._ws = None
            await self.connect()

    async def stream_audio(
        self,
        text: str,
        *,
        speaker: str | None = None,
        language: str | None = None,
        context_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        async with self._lock:
            await self._ensure_connected()
            assert self._ws is not None
            await self._ws.send(
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
                try:
                    msg = json.loads(await self._ws.recv()) # waits for next chunk of audio data
                except Exception:
                    self._ws = None
                    raise
                yield msg # immediately sends it forward to the client
                if msg.get("type") in ("done", "error"):
                    if msg.get("type") == "error":
                        raise RuntimeError(msg.get("message", "TTS error"))
                    break
