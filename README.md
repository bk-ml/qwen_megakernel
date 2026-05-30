## Qwen 0.6B Megakernel for RTX 5090

This megakernel is aggressively optimized for Qwen3-0.6B (bf16) shapes to be run on an RTX 5090.

More details on this blogpost: https://blog.alpindale.net/posts/5090_decode_optimization/


| Backend      | tok/s  | ms/tok | Speedup |
|--------------|--------|--------|---------|
| PyTorch (HF) | 123.3  | 8.11   | 1.00x   |
| Megakernel   | 1036.3 | 0.99   | 8.40x   |

### Performance targets

| Metric | Target |
|--------|--------|
| Decode tok/s | ≥ 800 |
| TTFC (first audio chunk) | ≤ 90 ms |
| RTF (TTS) | ≤ 0.3 |
| End-to-end latency | ≤ 2000 ms |

Run `python -m qwen_megakernel.bench` for decode; with the voice server up, run `python examples/voice_perf_test.py` for TTFC, RTF, and E2E. See [docs/INTEGRATION.md](docs/INTEGRATION.md).

To use this:

```bash
uv pip install -r requirements.txt
python -m qwen_megakernel.bench
```

### Voice agent (Pipecat)

Single server: megakernel LLM + Qwen3-TTS. See [docs/INTEGRATION.md](docs/INTEGRATION.md).

```bash
uv pip install -r requirements-voice.txt && python -m qwen_megakernel.server.app
uv pip install -r requirements-pipecat.txt && VOICE_SERVER_URL=http://127.0.0.1:8000 python examples/pipecat_voice_agent.py
```

Not tested on any other GPU, and likely won't run or work. Needs at least CUDA 12.8.


### Credits
Based on Elliot Arledge's [MegaQwen](https://github.com/Infatoshi/MegaQwen) for the RTX 3090 GPU.
