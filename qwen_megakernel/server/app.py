"""
Single-node voice server: megakernel LLM + Qwen3-TTS.

  POST /v1/chat/completions     OpenAI-compatible SSE (for Pipecat OpenAILLMService)
  WS   /v1/tts/ws               text → PCM chunks (for Pipecat TTSService)

Run:  python -m qwen_megakernel.server.app
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
import uuid
from functools import lru_cache
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from qwen_megakernel.server.llm import MegakernelLLM
from qwen_megakernel.server.tts import is_tts_available, stream_pcm_chunks

app = FastAPI(title="Qwen Megakernel Voice", version="0.2.0")


@lru_cache(maxsize=1)
def get_llm() -> MegakernelLLM:
    return MegakernelLLM()


# --- OpenAI-compatible chat (Pipecat LLM) -------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "qwen-megakernel"
    messages: list[ChatMessage]
    stream: bool = True
    max_tokens: int = Field(default=256, ge=1, le=2048)


def _sse_chat_chunk(content: str, *, finish: bool = False) -> str:
    delta: dict[str, Any] = {}
    if finish:
        delta = {}
        choice = {"index": 0, "delta": delta, "finish_reason": "stop"}
    else:
        choice = {"index": 0, "delta": {"content": content}, "finish_reason": None}
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "qwen-megakernel",
        "choices": [choice],
    }
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "llm": "megakernel",
        "tts": is_tts_available(),
    }


@app.post("/v1/chat/completions")
def chat_completions(body: ChatRequest):
    if not body.stream:
        text = "".join(
            get_llm().stream_chat(
                [m.model_dump() for m in body.messages],
                max_tokens=body.max_tokens,
            )
        )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }

    messages = [m.model_dump() for m in body.messages]

    def event_stream():
        for piece in get_llm().stream_chat(messages, max_tokens=body.max_tokens):
            yield _sse_chat_chunk(piece)
        yield _sse_chat_chunk("", finish=True)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- TTS WebSocket (Pipecat) --------------------------------------------------


class TtsRequest(BaseModel):
    text: str
    speaker: str | None = None
    language: str | None = None
    context_id: str | None = None


@app.websocket("/v1/tts/ws")
async def tts_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    if not is_tts_available():
        await websocket.send_json(
            {"type": "error", "message": "Install qwen-tts: pip install qwen-tts"}
        )
        await websocket.close()
        return

    while True:
        try:
            raw = await websocket.receive_json()
            req = TtsRequest.model_validate(raw)
        except WebSocketDisconnect:
            break
        except Exception as exc:
            await websocket.send_json({"type": "error", "message": str(exc)})
            continue

        context_id = req.context_id or uuid.uuid4().hex
        loop = asyncio.get_event_loop()

        try:
            chunks = await loop.run_in_executor(
                None,
                lambda: list(
                    stream_pcm_chunks(
                        req.text,
                        speaker=req.speaker,
                        language=req.language,
                    )
                ),
            )
            for pcm, sr in chunks:
                await websocket.send_json(
                    {
                        "type": "audio",
                        "context_id": context_id,
                        "sample_rate": sr,
                        "num_channels": 1,
                        "pcm_base64": base64.b64encode(pcm).decode("ascii"),
                    }
                )
            await websocket.send_json({"type": "done", "context_id": context_id})
        except Exception as exc:
            await websocket.send_json(
                {"type": "error", "message": str(exc), "context_id": context_id}
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Megakernel voice server (LLM + TTS)")
    parser.add_argument("--host", default=os.getenv("VOICE_SERVER_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("VOICE_SERVER_PORT", "8000"))
    )
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
