# Minimal voice pipeline (take-home)

## Architecture (1 service)

```text
┌─────────────────────────────────────────────────────────────┐
│  python -m qwen_megakernel.server.app   (port 8000)          │
│  ┌─────────────────────┐    ┌──────────────────────────┐   │
│  │ MegakernelLLM       │    │ Qwen3-TTS (in-process)   │   │
│  │ Decoder (0.6B bf16) │    │ text → PCM chunks        │   │
│  └──────────┬──────────┘    └────────────┬─────────────┘   │
│             │ POST /v1/chat/completions   │ WS /v1/tts/ws  │
└─────────────┼─────────────────────────────┼─────────────────┘
              │ SSE tokens                  │ audio chunks
              ▼                             ▼
┌─────────────────────────────────────────────────────────────┐
│  Pipecat (examples/pipecat_voice_agent.py)                  │
│  Mic → Whisper STT → OpenAILLMService → Qwen3TTSService →   │
│        speakers                                             │
└─────────────────────────────────────────────────────────────┘
```

## Full pipeline
```
You speak 🎤
   ↓
Pipecat (client)
   ↓
STT → text
   ↓
HTTP → /chat/completions (GPU server)
   ↓
Megakernel → text tokens
   ↓
Pipecat
   ↓
WebSocket → /tts/ws
   ↓
TTS → audio chunks
   ↓
Speaker 🔊
```


## Folder layout

```text
qwen_megakernel/
  model.py              # Decoder (megakernel) — do not change csrc
  server/
    app.py              # FastAPI entry (LLM + TTS routes)
    llm.py              # stream_text / stream_chat via Decoder.step()
    tts.py              # Qwen3-TTS PCM chunk streaming
  tts/
    synthesizer.py      # thin Qwen3TTSModel wrapper
integrations/pipecat/
  megakernel_llm.py     # OpenAILLMService → /v1/chat/completions
  qwen_tts.py           # TTSService → /v1/tts/ws
  voice_client.py       # VOICE_SERVER_URL helper
examples/
  pipecat_voice_agent.py
  round_trip_test.py
```

## Run on GPU host

```bash
uv pip install -r requirements-voice.txt
export MEGAKERNEL_MODEL=Qwen/Qwen3-0.6B
export QWEN3_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

python -m qwen_megakernel.server.app --port 8000
curl http://127.0.0.1:8000/health
```

## Pipecat client

```bash
uv pip install -r requirements-pipecat.txt
export VOICE_SERVER_URL=http://<gpu-host>:8000
python examples/pipecat_voice_agent.py
```

`OpenAILLMService` uses `base_url=$VOICE_SERVER_URL/v1` so LLM tokens stream over SSE without a custom LLM class in the hot path.

## Round-trip (no mic)

```bash
export VOICE_SERVER_URL=http://127.0.0.1:8000
python examples/round_trip_test.py --mock --text "What is two plus two?"
```

With WAV input:

```bash
python examples/round_trip_test.py --input samples/hello.wav --output reply.wav
```

## API summary

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | LLM + TTS readiness |
| `POST /v1/chat/completions` | OpenAI-compatible SSE (`stream: true`) |
| `WS /v1/tts/ws` | `{"text":"..."}` → `audio` + `done` messages |

## Environment

| Variable | Default |
|----------|---------|
| `VOICE_SERVER_URL` | `http://127.0.0.1:8000` |
| `MEGAKERNEL_MODEL` | `Qwen/Qwen3-0.6B` |
| `QWEN3_TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| `QWEN3_TTS_SPEAKER` | `serena` |
| `QWEN3_TTS_LANGUAGE` | `English` |

## Performance targets

SLOs for RTX 5090 (see `qwen_megakernel/perf_targets.py`):

| Metric | Target | Notes |
|--------|--------|-------|
| Decode tok/s | ≥ 800 | Megakernel LLM sustained decode |
| TTFC | ≤ 90 ms | Time to first TTS PCM chunk |
| RTF | ≤ 0.3 | TTS synthesis time ÷ audio duration |
| E2E latency | ≤ 2000 ms | User text → first audio chunk (LLM + TTS) |

Benchmark decode:

```bash
python -m qwen_megakernel.bench
```

Benchmark voice pipeline (server must be running):

```bash
python -m qwen_megakernel.server.app --port 8000
python examples/voice_perf_test.py
```
