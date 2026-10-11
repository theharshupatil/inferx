# InferX

An inference-systems lab for measuring latency and memory behavior, then testing optimizations under repeatable workloads.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
python -m inferx.benchmark --tokens 128 --batch-size 4
```

The current benchmark uses a small local PyTorch model to establish a measurement harness. It is not a production inference server and makes no speedup claims.

## Layout

- `kv_cache.py`: simple per-layer key/value cache container.
- `model.py`: tiny reference model for benchmarking.
- `benchmark.py`: warm-up, timed iterations, and summary output.
- `tests/`: correctness tests for cache growth and output shape.

## Metrics to add

- Time to first token (TTFT), inter-token latency, tokens/second.
- Peak allocated memory, batch size, sequence length, and hardware metadata.
- Baselines against an unoptimized implementation.
- Quality/correctness checks when introducing quantization or kernels.

## Next milestones

- [ ] Add prefill/decode separation and real autoregressive generation.
- [ ] Profile before optimizing.
- [ ] Compare static, dynamic, and paged KV-cache strategies.
- [ ] Add batching and scheduling experiments.
- [ ] Re-run every benchmark with fixed seeds and multiple trials.
