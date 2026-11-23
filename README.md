# kvasir

[![ci](https://github.com/zvdy/kvasir/actions/workflows/ci.yml/badge.svg)](https://github.com/zvdy/kvasir/actions/workflows/ci.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/zvdy/kvasir/badge)](https://scorecard.dev/viewer/?uri=github.com/zvdy/kvasir)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

An LLM inference engine written from scratch in Rust with hand-written CUDA kernels, targeting a single RTX 4070 (12 GB, sm_89). The goal is a mini-vLLM: paged KV cache, continuous batching, quantization, speculative decoding and an OpenAI-compatible server.

**Stage 1 (current):** correct batch-1 generation for Llama 3.2 1B.

| | status |
|---|---|
| GPU wrappers, cuBLAS GEMM | done |
| CUDA kernels (embedding, RMSNorm, RoPE, attention, SiLU·mul, KV append) with CPU-reference tests | done |
| Config, RoPE table, safetensors loader | done |
| Forward pass, KV cache, sampling, `kvasir generate` CLI | done (tested on a synthetic model) |
| HF logit-matching test, tokens/sec vs roofline | next |

Design, data layouts and numeric deviations from HF are in [DESIGN.md](DESIGN.md).

## Requirements

- Rust (pinned in `rust-toolchain.toml`) and an NVIDIA GPU with driver support for CUDA
- CUDA toolkit 13.x with `nvcc` (found via `NVCC`, `CUDA_PATH`, `CUDA_HOME`, `/opt/cuda` or `/usr/local/cuda`)
- Llama 3.2 1B weights (gated on Hugging Face): `hf download meta-llama/Llama-3.2-1B --local-dir models/Llama-3.2-1B`

## Layout

```
engine/
  build.rs      compiles kernels/*.cu to PTX with nvcc (-arch=sm_89)
  kernels/      CUDA C++ kernels
  src/gpu/      device, buffers, kernel launch wrappers, cuBLAS GEMM
  src/model/    config, RoPE tables, weight loading
  tests/        per-kernel tests against CPU references, loader test (feature gpu-tests)
.agent/         backlog.yaml and per-run logs for agent maintenance
docs/adr/       architecture decision records
scripts/        doctor.sh, agent hooks
ref/            HF transformers reference dumper (planned)
```

## Commands

```
make doctor     # environment report
make build      # release build
make fmt        # format (the only target that writes files)
make ci         # fmt-check, clippy, CPU tests, cargo-deny, docs (what GitHub CI runs)
make ci-local   # ci + GPU tests (required before every PR)
make kernels    # kernel tests with error magnitudes
```

## Usage

```
cargo run --release -- generate --model models/Llama-3.2-1B --prompt "The capital of France is" \
    --max-tokens 64 --seed 0                      # greedy
cargo run --release -- generate --model models/Llama-3.2-1B --prompt "Once upon a time" \
    --max-tokens 128 --seed 42 --temperature 0.8 --top-p 0.95 --json
```

Text streams to stdout; prompt/generated token counts and prefill/decode tok/s go to stderr (`--json`: one JSON line).
`--ignore-eos` forces exactly `--max-tokens` tokens, `--prefill-chunk` sets the prompt tokens per forward call.

## Contributing and agent maintenance

See [CONTRIBUTING.md](CONTRIBUTING.md). The repo is also maintained by coding agents: [AGENTS.md](AGENTS.md) is the
runbook, `.agent/backlog.yaml` the ordered task list, and `/maintain` in Claude Code runs one task end to end
(branch, gate, review, PR). Decisions are recorded in [docs/adr](docs/adr).

## License

[Apache-2.0](LICENSE)

No ML framework is used in the engine; candle, burn and tch are for reading only.
