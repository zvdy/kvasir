# kvasir

An LLM inference engine written from scratch in Rust with hand-written CUDA kernels, targeting a single RTX 4070 (12 GB, sm_89). The goal is a mini-vLLM: paged KV cache, continuous batching, quantization, speculative decoding and an OpenAI-compatible server.

**Stage 1 (current):** correct batch-1 generation for Llama 3.2 1B.

| | status |
|---|---|
| GPU wrappers, cuBLAS GEMM | done |
| CUDA kernels (embedding, RMSNorm, RoPE, attention, SiLU·mul, KV append) with CPU-reference tests | done |
| Config, RoPE table, safetensors loader | done |
| Forward pass, KV cache, sampling, CLI | next |
| HF logit-matching test, tokens/sec vs roofline | next |

Design, data layouts and numeric deviations from HF are in [DESIGN.md](DESIGN.md).

## Requirements

- Rust (stable) and an NVIDIA GPU with driver support for CUDA
- CUDA toolkit with `nvcc` (found via `NVCC`, `CUDA_PATH`, `CUDA_HOME`, `/opt/cuda` or `/usr/local/cuda`)
- Llama 3.2 1B weights (gated on Hugging Face): `hf download meta-llama/Llama-3.2-1B --local-dir models/Llama-3.2-1B`

## Layout

```
engine/
  build.rs      compiles kernels/*.cu to PTX with nvcc (-arch=sm_89)
  kernels/      CUDA C++ kernels
  src/gpu/      device, buffers, kernel launch wrappers, cuBLAS GEMM
  src/model/    config, RoPE tables, weight loading
  tests/        per-kernel tests against CPU references, loader test
ref/            HF transformers reference dumper (planned)
```

## Commands

```
make build      # release build
make test       # all tests (needs a GPU)
make test-lib   # tests that don't need a GPU
make kernels    # kernel tests with error magnitudes
make check      # fmt + clippy + tests
```

No ML framework is used in the engine; candle, burn and tch are for reading only.
