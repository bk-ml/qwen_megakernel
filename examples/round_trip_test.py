#!/usr/bin/env python3
"""Scripted round-trip: WAV → STT → LLM → TTS → out.wav (with full mock support)"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import wave
from pathlib import Path

import httpx
import numpy as np

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


# -----------------------------
# WAV utils
# -----------------------------
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


# -----------------------------
# MOCK components
# -----------------------------
def mock_llm(user_text: str):
    reply = f"[MOCK LLM RESPONSE to: {user_text}]"
    return reply, 20.0, 1000.0  # TTFT ms, tok/s


def mock_tts(text: str):
    sr = 24000
    duration_s = 1.0

    t = np.linspace(0, duration_s, int(sr * duration_s), False)
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)  # 440 Hz beep

    pcm = (tone * 32767).astype(np.int16).tobytes()

    return pcm, sr, 30.0, 0.1


def mock_stt(_: np.ndarray, __: int) -> str:
    return "Hello, this is mocked STT input"


# -----------------------------
# REAL components
# -----------------------------
def transcribe(audio: np.ndarray, sample_rate: int) -> str:
    import whisper
    import librosa

    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

    model = whisper.load_model(os.getenv("WHISPER_MODEL", "base"))
    return model.transcribe(audio, fp16=False)["text"].strip()


def llm_stream(user_text: str):
    from integrations.pipecat.voice_client import voice_server_base

    base = voice_server_base()

    messages = [
        {"role": "system", "content": "Reply in one short spoken sentence."},
        {"role": "user", "content": user_text},
    ]

    t0 = time.perf_counter()
    ttft_ms = None
    tokens = 0
    text = []

    with httpx.Client(timeout=120.0) as client:
        with client.stream(
            "POST",
            f"{base}/v1/chat/completions",
            json={"messages": messages, "stream": True, "max_tokens": 128},
        ) as r:
            r.raise_for_status()
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


def tts_pcm(text: str):
    import asyncio
    from integrations.pipecat.voice_client import AsyncTtsWebSocketClient

    async def _run():
        chunks = []
        sr = 24000
        ttfc_ms = None
        t0 = time.perf_counter()

        async for msg in AsyncTtsWebSocketClient().stream_audio(text):
            if msg.get("type") == "audio":
                if ttfc_ms is None:
                    ttfc_ms = (time.perf_counter() - t0) * 1000
                chunks.append(base64.b64decode(msg["pcm_base64"]))
                sr = int(msg["sample_rate"])

        pcm = b"".join(chunks)
        audio_s = len(pcm) / (2 * sr)
        rtf = (time.perf_counter() - t0) / audio_s if audio_s > 0 else 0.0

        return pcm, sr, ttfc_ms or 0.0, rtf

    return asyncio.run(_run())


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path, default=Path("examples/mock/round_trip_out.wav"))
    p.add_argument("--mock", action="store_true")
    p.add_argument("--text", default="Hello, how are you?")
    args = p.parse_args()

    # ---------------- MOCK MODE ----------------
    if args.mock:
        print("🧪 RUNNING IN FULL MOCK MODE")

        user = args.text
        print(f"[STT-MOCK] {user}")

        reply, llm_ttft, tok_s = mock_llm(user)
        print(f"[LLM-MOCK] {reply}")

        pcm, sr, ttfc, rtf = mock_tts(reply)
        print(f"[TTS-MOCK] generated {len(pcm)} bytes")

        e2e = 120.0

    # ---------------- REAL MODE ----------------
    else:
        user = transcribe(*load_wav(args.input))
        print(f"[STT] {user}")

        reply, llm_ttft, tok_s = llm_stream(user)
        print(f"[LLM] {reply}")

        pcm, sr, ttfc, rtf = tts_pcm(reply)
        e2e = 0.0  # optional compute if needed

        print(f"[TTS] {len(pcm)} bytes @ {sr}")

    # ---------------- OUTPUT ----------------
    print(
        f"\n[PERF]\n"
        f"decode={tok_s:.1f} tok/s\n"
        f"LLM TTFT={llm_ttft:.1f} ms\n"
        f"TTS TTFC={ttfc:.1f} ms\n"
        f"RTF={rtf:.2f}\n"
    )

    save_wav(args.output, pcm, sr)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()