# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/) (pre-1.0, minor versions may break).

## [Unreleased]

### Added

- Agent maintenance harness: `AGENTS.md`, `.agent/backlog.yaml`, `/maintain` skill, `reviewer` subagent, `make doctor`.
- `make ci` / `make ci-local` gates; GPU tests gated behind the `gpu-tests` feature.
- Apache-2.0 license, governance docs, CI (CUDA container build, cargo-deny, DCO, OpenSSF Scorecard), ADRs.
- Kernel microbenchmarks (`make bench`) at Llama 3.2 1B decode/prefill shapes, compared against
  `bench/kernels-baseline.json` (fails on > 10% regression); `make bench-baseline` rewrites it.

### Changed

- Attention rewritten (A-01): flash-decoding for `T = 1` (one block per KV head × context split, GQA heads share
  each K/V load, coalesced 16-byte loads, online softmax, last-block split merge) and an FA2-style tensor-core
  (`mma.sync`) kernel for causal prefill. Decode at ctx 2048: 311 → 14.0 µs (3% → 60% of DRAM bandwidth); prefill
  T=128: 178 → 7.0 µs. The ≈11.9k kv-length cap is gone. **Breaking:** `Gpu::attention` takes a new
  `scratch: &mut CudaViewMut<f32>` (size `kernels::attention_scratch_len(n_q, head_dim)`, zeroed once at allocation)
  and supports head_dim 64/128 with GQA groups 1/2/4/8.

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
