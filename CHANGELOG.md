# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/) (pre-1.0, minor versions may break).

## [Unreleased]

### Added

- `model::Workspace`: preallocated activation buffers sized by `WorkspaceDims` from `LlamaConfig`, max tokens per
  call and max LM-head rows; token-id upload and logits D2H reuse preallocated host buffers (S1-01).
- Host sampling (`engine/src/sampling.rs`): greedy, temperature and top-p with a seeded `StdRng`; no per-step heap
  allocation, nucleus found by partial selection instead of a full vocab sort.
- Agent maintenance harness: `AGENTS.md`, `.agent/backlog.yaml`, `/maintain` skill, `reviewer` subagent, `make doctor`.
- `make ci` / `make ci-local` gates; GPU tests gated behind the `gpu-tests` feature.
- Apache-2.0 license, governance docs, CI (CUDA container build, cargo-deny, DCO, OpenSSF Scorecard), ADRs.
- `kv::KvCache`: contiguous FP16 K/V cache `[layers, n_kv, max_seq, head_dim]` with per-layer views for `kv_append`
  and `attention`, length tracking and overflow errors (S1-02).
- `model::Llama`: forward pass over `Weights` + `KvCache` + `Workspace` for prefill, chunked prefill and decode;
  `forward(gpu, ids, Logits::Last | Logits::All)` returns FP32 logits without device allocation. `KvCache::from_config`
  now owns the RoPE cos/sin tables; `DeviceTensor::row_view(_mut)` give leading/ranged row views (S1-03).
- Kernel microbenchmarks (`make bench`) at Llama 3.2 1B decode/prefill shapes, compared against
  `bench/kernels-baseline.json` (fails on > 10% regression); `make bench-baseline` rewrites it.
- `kvasir generate --model <dir> --prompt <text> --max-tokens N --seed S [--temperature T --top-p P]`: chunked
  prefill, streamed output, stops at any EOS id from config/generation_config, prints prefill and decode tok/s to
  stderr (`--json` for one JSON line, `--ignore-eos` for fixed-length runs). `Llama::max_seq_limit` (S1-05).
- End-to-end forward bench (`make bench-forward`, also run by `make bench`): synthetic full-size Llama 3.2 1B,
  prefill 128 tokens and decode at ctx 128/512/2048 (wall and device time, % of the `decode_bytes` roofline),
  compared against `bench/forward-baseline.json` (> 10% regression fails). Decode: 5.67–5.83 ms/token (171–176
  tok/s, 84–86% of roofline); prefill 128 tokens: 7.65 ms (I-01).

### Changed

- Attention rewritten (A-01): flash-decoding for `T = 1` (one block per KV head × context split, GQA heads share
  each K/V load, coalesced 16-byte loads, online softmax, last-block split merge) and an FA2-style tensor-core
  (`mma.sync`) kernel for causal prefill. Decode at ctx 2048: 311 → 14.0 µs (3% → 60% of DRAM bandwidth); prefill
  T=128: 178 → 7.0 µs. The ≈11.9k kv-length cap is gone. **Breaking:** `Gpu::attention` takes a new
  `scratch: &mut CudaViewMut<f32>` (size `kernels::attention_scratch_len(n_q, head_dim)`, zeroed once at allocation)
  and supports head_dim 64/128 with GQA groups 1/2/4/8.
- The forward pass uses the A-01 attention (I-01): `Workspace` owns the zeroed `attn_scratch`, `KvCache::attention`
  takes it as a parameter, `Llama::new` rejects head shapes without an attention variant instead of the old
  shared-memory kv-length cap, and `Llama::max_seq_limit` now returns the kernels' i32 bound. `Gpu::attention`
  returns a `Shape` error for misaligned views. Forward and CLI test models use head_dim 64 (tolerances unchanged).
- Decode (`T = 1`) projections use hand-written FP16 GEMV kernels instead of cuBLAS (P-01, ADR 0007, amends 0002):
  `gemm::gemv_f16 / gemv_f32 / gemv_add / gemv_silu_mul` with an optional fused RMSNorm prologue (`gemm::Norm`),
  SwiGLU fused into gate/up and the residual add fused into o/down; 5 launches fewer per layer. Decode 2.8–3.8%
  faster (ctx 2048: 5.83 → 5.67 ms/token wall, 176 tok/s, 87% of roofline); GEMV kernel bench entries added and
  baselines regenerated. `Llama::new` now requires hidden / q_dim / intermediate to be multiples of 8. Prefill
  still uses cuBLAS.
- CI split into parallel `fmt`, `clippy`, `test`, `build` (GPU test binaries, nvcc), `doc`, `deny`, `actionlint` jobs
  behind one required `ci-pass` check; shared setup in `.github/actions/setup`.
- New `agent-pr` check (`scripts/check-agent-pr.sh`, `make check-agent-pr`): agent PRs need a run log with local
  `make test-gpu` results, a CHANGELOG entry and an updated backlog status (ADR 0006).
- Push hook checks the branch of the directory a push runs in, so agents in worktrees can push and `git -C` cannot bypass it.
- Harness allows parallel agent runs in separate worktrees for non-overlapping tasks (`/maintain parallel`).
- Backlog: S1-09 batch-1 comparison against llama.cpp and vLLM.
- CI installs Rust with a SHA-pinned action instead of `curl | sh` (Scorecard Pinned-Dependencies).
- cudarc pinned to the `cuda-13040` feature instead of detecting the toolkit version (CUDA 13.5 is unknown to cudarc 0.19).
- Rust toolchain pinned via `rust-toolchain.toml`.

### Fixed

- Tests import `rand::RngExt` (moved there in rand 0.10); main CI was red after the Dependabot bump.

### Security

- rustix 1.1.5 → 1.1.6 (RUSTSEC-2026-0486).

## [0.1.0] - initial import

- CUDA kernels (embedding, RMSNorm, RoPE, attention, SiLU·mul, KV append, residual add, BF16→FP16) with CPU-reference
  tests, cuBLAS GEMM, Llama config, RoPE tables, safetensors loader.
