# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/) (pre-1.0, minor versions may break).

## [Unreleased]

### Added

- Agent maintenance harness: `AGENTS.md`, `.agent/backlog.yaml`, `/maintain` skill, `reviewer` subagent, `make doctor`.
- `make ci` / `make ci-local` gates; GPU tests gated behind the `gpu-tests` feature.
- Apache-2.0 license, governance docs, CI (CUDA container build, cargo-deny, DCO, OpenSSF Scorecard), ADRs.

### Changed

- cudarc pinned to the `cuda-13040` feature instead of detecting the toolkit version (CUDA 13.5 is unknown to cudarc 0.19).
- Rust toolchain pinned via `rust-toolchain.toml`.

## [0.1.0] - initial import

- CUDA kernels (embedding, RMSNorm, RoPE, attention, SiLU·mul, KV append, residual add, BF16→FP16) with CPU-reference
  tests, cuBLAS GEMM, Llama config, RoPE tables, safetensors loader.
