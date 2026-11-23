# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/) (pre-1.0, minor versions may break).

## [Unreleased]

### Added

- `model::Workspace`: preallocated activation buffers sized by `WorkspaceDims` from `LlamaConfig`, max tokens per
  call and max LM-head rows; token-id upload and logits D2H reuse preallocated host buffers (S1-01).
- Agent maintenance harness: `AGENTS.md`, `.agent/backlog.yaml`, `/maintain` skill, `reviewer` subagent, `make doctor`.
- `make ci` / `make ci-local` gates; GPU tests gated behind the `gpu-tests` feature.
- Apache-2.0 license, governance docs, CI (CUDA container build, cargo-deny, DCO, OpenSSF Scorecard), ADRs.
- `kv::KvCache`: contiguous FP16 K/V cache `[layers, n_kv, max_seq, head_dim]` with per-layer views for `kv_append`
  and `attention`, length tracking and overflow errors (S1-02).
- `model::Llama`: forward pass over `Weights` + `KvCache` + `Workspace` for prefill, chunked prefill and decode;
  `forward(gpu, ids, Logits::Last | Logits::All)` returns FP32 logits without device allocation. `KvCache::from_config`
  now owns the RoPE cos/sin tables; `DeviceTensor::row_view(_mut)` give leading/ranged row views (S1-03).

### Changed

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
