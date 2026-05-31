#!/usr/bin/env python3
"""Measure voice pipeline metrics against perf_targets."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from qwen_megakernel.perf_targets import TARGETS, check_max, check_min
from integrations.pipecat.voice_client import AsyncTtsWebSocketClient, voice_server_base


PROMPT = "Reply in one short spoken sentence."
USER_TEXT = "What is two plus two?"


def bench_decode_tok_s(*, tokens: int = 100, warmup: int = 3, runs: int = 5) -> float:
    import torch

    from qwen_megakernel.model import Decoder

    dec = Decoder(verbose=False)

    def run() -> None:
        dec.reset()
        dec.generate("Hello", max_tokens=tokens)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return tokens / (sum(times) / len(times))


async def bench_tts(text: str) -> tuple[float, float]:
    """Return (TTFC ms, RTF)."""
    client = AsyncTtsWebSocketClient()
    t0 = time.perf_counter()
    ttfc_ms: float | None = None
    pcm_bytes = 0
    sample_rate = 24000

    async for msg in client.stream_audio(text):
        if msg.get("type") != "audio":
            continue
        now = time.perf_counter()
        if ttfc_ms is None:
            ttfc_ms = (now - t0) * 1000
        pcm_bytes += len(base64.b64decode(msg["pcm_base64"]))
        sample_rate = int(msg.get("sample_rate", sample_rate))

    synth_s = time.perf_counter() - t0
    if ttfc_ms is None:
        raise RuntimeError("TTS returned no audio chunks")

    audio_s = pcm_bytes / (2 * sample_rate)  # int16 mono
    rtf = synth_s / audio_s if audio_s > 0 else float("inf")
    return ttfc_ms, rtf


async def bench_e2e(user_text: str) -> tuple[float, float, float]:
    """Return (E2E TTFC ms, LLM TTFT ms, decode tok/s over full reply)."""
    base = voice_server_base()
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": user_text},
    ]

    t0 = time.perf_counter()
    ttft_ms: float | None = None
    reply_parts: list[str] = []
    token_count = 0

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{base}/v1/chat/completions",
            json={"messages": messages, "stream": True, "max_tokens": 128},
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
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
                    reply_parts.append(c)
                    token_count += 1

    reply = "".join(reply_parts)
    llm_s = time.perf_counter() - t0
    decode_tok_s = token_count / llm_s if llm_s > 0 else 0.0

    tts_client = AsyncTtsWebSocketClient()
    tts_t0 = time.perf_counter()
    e2e_ttfc_ms: float | None = None

    async for msg in tts_client.stream_audio(reply):
        if msg.get("type") == "audio" and e2e_ttfc_ms is None:
            e2e_ttfc_ms = (time.perf_counter() - t0) * 1000
            break

    if e2e_ttfc_ms is None:
        raise RuntimeError("E2E run produced no TTS audio")

    return e2e_ttfc_ms, ttft_ms or 0.0, decode_tok_s


async def run_server_checks(*, user_text: str, skip_e2e: bool) -> bool:
    print("Voice server metrics")
    print("-" * 55)

    ttfc_ms, rtf = await bench_tts("Hello from the performance test.")
    ok = check_max("TTS TTFC", ttfc_ms, TARGETS.ttfc_ms_max, unit="ms")
    ok = check_max("TTS RTF", rtf, TARGETS.rtf_max) and ok

    if not skip_e2e:
        e2e_ms, llm_ttft_ms, stream_tok_s = await bench_e2e(user_text)
        ok = check_max("E2E latency", e2e_ms, TARGETS.e2e_latency_ms_max, unit="ms")
        check_max("LLM TTFT", llm_ttft_ms, TARGETS.ttfc_ms_max, unit="ms")
        check_min("Stream decode tok/s", stream_tok_s, TARGETS.decode_tok_s_min, unit="tok/s")

    return ok


def run_decode_check(*, tokens: int) -> bool:
    print("Megakernel decode")
    print("-" * 55)
    tok_s = bench_decode_tok_s(tokens=tokens)
    return check_min("Decode tok/s", tok_s, TARGETS.decode_tok_s_min, unit="tok/s")


async def main() -> None:
    p = argparse.ArgumentParser(description="Voice pipeline performance vs targets")
    p.add_argument("--text", default=USER_TEXT, help="User prompt for E2E run")
    p.add_argument("--tokens", type=int, default=100, help="Tokens for decode bench")
    p.add_argument(
        "--server-only",
        action="store_true",
        help="Skip in-process decode bench (no local GPU)",
    )
    p.add_argument(
        "--decode-only",
        action="store_true",
        help="Only run megakernel decode benchmark",
    )
    p.add_argument(
        "--skip-e2e",
        action="store_true",
        help="Skip end-to-end HTTP+TTS latency measurement",
    )
    args = p.parse_args()

    print("=" * 55)
    print("Performance targets")
    print("=" * 55)
    print(f"  Decode tok/s     ≥ {TARGETS.decode_tok_s_min:.0f}")
    print(f"  TTFC             ≤ {TARGETS.ttfc_ms_max:.0f} ms")
    print(f"  RTF              ≤ {TARGETS.rtf_max:.2f}")
    print(f"  E2E latency      ≤ {TARGETS.e2e_latency_ms_max:.0f} ms")
    print()

    ok = True
    if not args.server_only and not args.decode_only:
        try:
            ok = run_decode_check(tokens=args.tokens) and ok
            print()
        except Exception as exc:
            print(f"  Decode bench skipped: {exc}")
            print()

    if not args.decode_only:
        try:
            ok = await run_server_checks(user_text=args.text, skip_e2e=args.skip_e2e) and ok
        except Exception as exc:
            print(f"  Server metrics failed: {exc}")
            ok = False

    print()
   
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
