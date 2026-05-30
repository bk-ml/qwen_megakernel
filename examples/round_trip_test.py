#!/usr/bin/env python3
"""Scripted round-trip: WAV → STT → megakernel LLM → TTS → out.wav"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import wave
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from integrations.pipecat.voice_client import voice_server_base


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        if wf.getnchannels() > 1:
            pcm = pcm.reshape(-1, wf.getnchannels())[:, 0]
        return pcm.astype(np.float32) / 32768.0, sr


def save_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def transcribe(audio: np.ndarray, sample_rate: int) -> str:
    import whisper

    if sample_rate != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    return whisper.load_model(os.getenv("WHISPER_MODEL", "base")).transcribe(
        audio, fp16=False
    )["text"].strip()


def llm_stream(user_text: str) -> tuple[str, float, float]:
    """Return (reply, TTFT ms, decode tok/s)."""
    base = voice_server_base()
    messages = [
        {"role": "system", "content": "Reply in one short spoken sentence."},
        {"role": "user", "content": user_text},
    ]
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    tokens = 0
    with httpx.Client(timeout=120.0) as client:
        with client.stream(
            "POST",
            f"{base}/v1/chat/completions",
            json={"messages": messages, "stream": True, "max_tokens": 128},
        ) as r:
            r.raise_for_status()
            text = []
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta", {})
                if c := delta.get("content"):
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    tokens += 1
                    text.append(c)
    elapsed = time.perf_counter() - t0
    tok_s = tokens / elapsed if elapsed > 0 else 0.0
    return "".join(text), ttft_ms or 0.0, tok_s


def tts_pcm(text: str) -> tuple[bytes, int, float, float]:
    """Return (pcm, sample_rate, TTFC ms, RTF)."""
    import asyncio

    from integrations.pipecat.voice_client import AsyncTtsWebSocketClient

    async def _run():
        chunks = []
        sr = 24000
        t0 = time.perf_counter()
        ttfc_ms: float | None = None
        async for msg in AsyncTtsWebSocketClient().stream_audio(text):
            if msg.get("type") == "audio":
                if ttfc_ms is None:
                    ttfc_ms = (time.perf_counter() - t0) * 1000
                chunks.append(base64.b64decode(msg["pcm_base64"]))
                sr = int(msg["sample_rate"])
        synth_s = time.perf_counter() - t0
        pcm = b"".join(chunks)
        audio_s = len(pcm) / (2 * sr)
        rtf = synth_s / audio_s if audio_s > 0 else float("inf")
        return pcm, sr, ttfc_ms or 0.0, rtf

    return asyncio.run(_run())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, help="Input speech WAV")
    p.add_argument("--output", type=Path, default=Path("round_trip_out.wav"))
    p.add_argument("--mock", action="store_true", help="Skip STT; fixed user text")
    p.add_argument("--text", default="Hello, how are you?")
    args = p.parse_args()

    user = args.text if args.mock else transcribe(*load_wav(args.input))
    print(f"[STT] {user!r}")
    e2e_t0 = time.perf_counter()
    reply, llm_ttft_ms, decode_tok_s = llm_stream(user)
    print(f"[LLM] {reply!r}")
    pcm, sr, ttfc_ms, rtf = tts_pcm(reply)
    e2e_ms = (time.perf_counter() - e2e_t0) * 1000
    print(f"[TTS] {len(pcm)} bytes @ {sr} Hz")
    print(
        f"[perf] decode={decode_tok_s:.0f} tok/s  "
        f"LLM TTFT={llm_ttft_ms:.0f} ms  "
        f"TTS TTFC={ttfc_ms:.0f} ms  "
        f"RTF={rtf:.2f}  "
        f"E2E={e2e_ms:.0f} ms"
    )
    save_wav(args.output, pcm, sr)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
